#!/usr/bin/env python3
"""
ExpertFlow v23 — Large Quantized Cache + Batched Miss Loading
=============================================================
v22 results: 9.6s/tok best, 36-38% hit rate with 400-expert budget.
Each decode pass: ~712 unique experts. After trim to 400, ~312 overlap.

v23 changes:
  1. Budget 700 experts (~7.5GB) — covers nearly a full decode pass
     After trim: 700/712 = 98% of last pass retained!
  2. Batch cache miss loading: collect all misses per layer, single mx.eval
  3. Skip attention pinning to save ~8GB for cache headroom
     (attention is only 0.2-0.3s — not worth 8GB of RAM)
  4. Lower wired limit to leave more for mmap/cache

Memory budget:
  - Cache: 700 × 11MB = 7.5GB (peak ~8GB during pass before trim)
  - No attention pin: save ~8GB
  - Net: same RAM pressure but much higher hit rate
"""

import os, sys, time, json, subprocess, traceback
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
from collections import OrderedDict


def free_gb():
    try:
        out = subprocess.check_output(["vm_stat"], timeout=2).decode()
        f = i = 0
        for l in out.split("\n"):
            if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
            elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
        return (f + i) * 16384 / 1e9
    except:
        return -1


def get_moe(layer):
    if hasattr(layer, 'block_sparse_moe'):
        return layer.block_sparse_moe
    elif hasattr(layer, 'mlp'):
        return layer.mlp
    return None


def is_moe(layer):
    m = get_moe(layer)
    return m is not None and hasattr(m, 'gate') and hasattr(m, 'switch_mlp')


class SmartExpertCache:
    def __init__(self, budget=700):
        self.cache = OrderedDict()
        self.budget = budget
        self.token_hits = 0
        self.token_misses = 0
        self.total_hits = 0
        self.total_misses = 0

    def get(self, key):
        if key in self.cache:
            self.token_hits += 1
            self.total_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.token_misses += 1
        self.total_misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value

    def trim(self):
        evicted = 0
        while len(self.cache) > self.budget:
            self.cache.popitem(last=False)
            evicted += 1
        return evicted

    def reset_token_stats(self):
        self.token_hits = 0
        self.token_misses = 0

    @property
    def token_hit_rate(self):
        t = self.token_hits + self.token_misses
        return self.token_hits / t * 100 if t > 0 else 0

    @property
    def total_hit_rate(self):
        t = self.total_hits + self.total_misses
        return self.total_hits / t * 100 if t > 0 else 0

    def stats(self):
        return (f"{len(self.cache)} cached | "
                f"tok {self.token_hit_rate:.0f}% (all {self.total_hit_rate:.0f}%)")


def streaming_moe_forward(moe_module, x, expert_cache, use_cache=True):
    B, S, H = x.shape

    gates_out = moe_module.gate(x)
    if isinstance(gates_out, tuple):
        inds, scores = gates_out
    else:
        k = moe_module.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates_out, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates_out, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
    mx.eval(inds, scores)

    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp
    gs_val = mlp.gate_proj.group_size
    bits = mlp.gate_proj.bits
    layer_idx = getattr(moe_module, '_ef_layer_idx', -1)

    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    inds_list = inds.reshape(B * S, topk).tolist()
    scores_flat = scores.reshape(B * S, topk)
    x_flat = x.reshape(B * S, H)

    # Collect unique experts needed and check cache
    needed = set()
    for t in range(B * S):
        for k_i in range(topk):
            needed.add(inds_list[t][k_i])

    # Batch load cache misses
    expert_weights = {}
    miss_evals = []
    for eidx in needed:
        cache_key = (layer_idx, eidx)
        if use_cache:
            cached = expert_cache.get(cache_key)
        else:
            cached = None

        if cached is not None:
            expert_weights[eidx] = cached
        else:
            # Load quantized weights (lazy — not eval'd yet)
            parts = []
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                w = proj["weight"][eidx]
                s = proj["scales"][eidx]
                b_arr = proj.get("biases")
                b = b_arr[eidx] if b_arr is not None else None
                parts.extend([w, s, b])
                miss_evals.extend([p for p in [w, s, b] if p is not None])
            expert_weights[eidx] = tuple(parts)
            if use_cache:
                expert_cache.put(cache_key, tuple(parts))

    # Single batched eval for ALL cache misses in this layer
    if miss_evals:
        mx.eval(*miss_evals)

    # Compute expert outputs
    with mx.stream(mx.cpu):
        token_outs = []
        for t in range(B * S):
            x_t = x_flat[t:t+1]
            expert_results = []

            for k_i in range(topk):
                eidx = inds_list[t][k_i]
                score = scores_flat[t, k_i]
                gw, gs, gb, uw, us, ub, dw, ds, db = expert_weights[eidx]

                g = mx.quantized_matmul(x_t, gw, gs, gb, transpose=True, group_size=gs_val, bits=bits)
                u = mx.quantized_matmul(x_t, uw, us, ub, transpose=True, group_size=gs_val, bits=bits)
                out = mx.quantized_matmul(nn.silu(g) * u, dw, ds, db, transpose=True, group_size=gs_val, bits=bits)
                expert_results.append(out * score)

            combined = expert_results[0]
            for er in expert_results[1:]:
                combined = combined + er
            token_outs.append(combined)

        routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
        result = (routed + shared) if shared is not None else routed
        mx.eval(result)

    return result


def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def generate(model, tokenizer, prompt, max_tokens, cache_budget=700, verbose=True):
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    moe_indices = []
    for i, layer in enumerate(model.model.layers):
        if is_moe(layer):
            moe_indices.append(i)
            get_moe(layer)._ef_layer_idx = i

    expert_cache = SmartExpertCache(budget=cache_budget)
    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_indices)} MoE)")
        print(f"  Cache: {cache_budget} experts (~{cache_budget * 11 / 1024:.1f}GB)")

    for step in range(max_tokens):
        t_token = time.time()
        expert_cache.reset_token_stats()

        if step == 0:
            ids = mx.array([input_ids])
        else:
            ids = mx.array([[generated_ids[-1]]])

        x = model.model.embed_tokens(ids)
        mx.eval(x)

        seq_len = ids.shape[1]
        cache_offset = kv_caches[0].offset if not kv_caches[0].empty() else 0
        mask = create_mask(seq_len, cache_offset)

        attn_time = 0
        moe_time = 0

        for i, layer in enumerate(model.model.layers):
            t_a = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)
            attn_time += time.time() - t_a

            t_m = time.time()
            h = layer.post_attention_layernorm(x)
            moe_mod = get_moe(layer)
            if i in moe_indices:
                h = streaming_moe_forward(moe_mod, h, expert_cache, use_cache=(step > 0))
            else:
                h = moe_mod(h)
                mx.eval(h)
            x = x + h
            mx.eval(x)
            moe_time += time.time() - t_m

            if verbose and (i + 1) % 10 == 0:
                mem = free_gb()
                print(f"[L{i+1} {mem:.0f}G]", end=" ", flush=True)

        pre_trim = len(expert_cache.cache)
        evicted = expert_cache.trim()

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)

        next_id = int(mx.argmax(logits[0, 0]).item())
        generated_ids.append(next_id)

        dt = time.time() - t_token
        token_times.append(dt)

        if verbose:
            try:
                text = tokenizer.decode([next_id])
            except:
                text = f"[{next_id}]"
            mode = "prefill" if step == 0 else "decode"
            print(f"\n  ✅ Token {step+1} ({mode}): {text!r} | "
                  f"{dt:.1f}s ({1/dt:.3f} tok/s) | "
                  f"attn={attn_time:.1f}s moe={moe_time:.1f}s | "
                  f"{expert_cache.stats()} | "
                  f"trim {pre_trim}→{len(expert_cache.cache)}", flush=True)

    try:
        output_text = tokenizer.decode(generated_ids)
    except:
        output_text = str(generated_ids)

    prefill_time = token_times[0] if token_times else 0
    decode_times = token_times[1:] if len(token_times) > 1 else []
    decode_avg = sum(decode_times) / len(decode_times) if decode_times else 0

    return {
        "prompt": prompt,
        "output": output_text,
        "tokens": len(generated_ids),
        "prefill_s": round(prefill_time, 2),
        "decode_avg_s": round(decode_avg, 2),
        "decode_tok_s": round(1/decode_avg, 4) if decode_avg > 0 else 0,
        "total_s": round(sum(token_times), 1),
        "cache_hit_rate": round(expert_cache.total_hit_rate, 1),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--cache-budget", type=int, default=700)
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v23 — Large Cache + Batched Loading")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(4 * 1024**3))
    try:
        mx.set_wired_limit(int(60 * 1024**3))  # Lower to leave room for cache
        print("  Wired: 60GB")
    except:
        pass

    import mlx_lm
    print("  Loading (lazy)...", flush=True)
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"  Loaded. Free: {free_gb():.0f}G")

    try:
        results = generate(model, tokenizer, args.prompt, args.max_tokens,
                          cache_budget=args.cache_budget)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        traceback.print_exc()
        return

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {results['prompt']}{results['output']}")
    print(f"  PREFILL: {results['prefill_s']}s")
    print(f"  DECODE:  {results['decode_avg_s']}s/tok ({results['decode_tok_s']} tok/s)")
    print(f"  Cache: {results['cache_hit_rate']}% hit")
    print(f"  Total: {results['total_s']}s for {results['tokens']} tokens")

    outfile = os.path.expanduser(f"~/dev/expertflow/v23_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
