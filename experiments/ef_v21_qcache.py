#!/usr/bin/env python3
"""
ExpertFlow v21 — Quantized Expert Cache
========================================
Target: 1 tok/s on GLM-4.5 (685B, 198GB, 128GB RAM). Currently 0.09 tok/s.

Key insight: Cache experts in QUANTIZED form (~11MB each, not 84MB dequantized).
  - 500 cached experts = 5.5GB RAM (feasible)
  - MoE expert usage follows power law — top 20% handle 60-80% of activations
  - Cache hit: read from RAM (microseconds) vs NVMe page fault (~5ms)
  - Expected: 60-80% cache hit rate → 3-5x speedup → ~0.3-0.5 tok/s

Cache strategy:
  - LRU eviction with frequency-weighted scoring
  - Key: (layer_idx, expert_idx)
  - Value: (gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b)
  - All values stay as MLX quantized arrays (uint32 + float16 scales)

Combined with: pinned attention, KV cache, CPU quantized_matmul.
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


def pin_attention_weights(model):
    print("  Pinning attention...", end=" ", flush=True)
    t0 = time.time()
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn, name):
                proj = getattr(attn, name)
                if hasattr(proj, 'weight'):
                    mx.eval(proj.weight.sum())
        for ln in ['input_layernorm', 'post_attention_layernorm']:
            if hasattr(layer, ln):
                mx.eval(getattr(layer, ln).weight.sum())
    mx.eval(model.model.embed_tokens.weight.sum())
    if hasattr(model.lm_head, 'weight'):
        mx.eval(model.lm_head.weight.sum())
    if hasattr(model.model, 'norm'):
        mx.eval(model.model.norm.weight.sum())
    print(f"done ({time.time()-t0:.1f}s, {free_gb():.0f}G free)")


class QuantizedExpertCache:
    """
    LRU cache storing expert weights in QUANTIZED form.
    Each entry: ~11MB (3 projections × weight + scales + biases in int4/float16)
    vs ~84MB dequantized. 500 entries = ~5.5GB RAM.
    """
    def __init__(self, max_entries=600):
        self.cache = OrderedDict()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self.total_bytes = 0
        self.ENTRY_BYTES = 11 * 1024 * 1024  # ~11MB estimate per expert

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        while len(self.cache) >= self.max_entries:
            self.cache.popitem(last=False)
            self.total_bytes -= self.ENTRY_BYTES
        self.cache[key] = value
        self.total_bytes += self.ENTRY_BYTES

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0

    @property
    def size_gb(self):
        return self.total_bytes / 1024**3

    def stats(self):
        return (f"{len(self.cache)}/{self.max_entries} experts, "
                f"{self.hit_rate:.0f}% hit, {self.size_gb:.1f}GB")


def load_expert_quantized(mlp, eidx):
    """
    Load one expert's quantized weights from mmap and materialize them.
    Returns a tuple of (weight, scales, biases) for each projection.
    The mx.eval forces the data into RAM — subsequent access is free.
    """
    parts = []
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp, proj_name)
        w = proj["weight"][eidx]
        s = proj["scales"][eidx]
        b_arr = proj.get("biases")
        b = b_arr[eidx] if b_arr is not None else None
        parts.extend([w, s, b])

    # Single eval materializes all parts into RAM
    to_eval = [p for p in parts if p is not None]
    mx.eval(*to_eval)

    return tuple(parts)


def expert_qmm_from_cache(x_t, cached_parts, group_size, bits):
    """
    Run SwiGLU expert computation using cached quantized weights.
    cached_parts = (gw, gs, gb, uw, us, ub, dw, ds, db)
    """
    gw, gs, gb, uw, us, ub, dw, ds, db = cached_parts
    g = mx.quantized_matmul(x_t, gw, gs, gb, transpose=True, group_size=group_size, bits=bits)
    u = mx.quantized_matmul(x_t, uw, us, ub, transpose=True, group_size=group_size, bits=bits)
    return mx.quantized_matmul(nn.silu(g) * u, dw, ds, db, transpose=True, group_size=group_size, bits=bits)


def streaming_moe_forward(moe_module, x, expert_cache):
    B, S, H = x.shape

    # Gate
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
    group_size = mlp.gate_proj.group_size
    bits = mlp.gate_proj.bits

    # Shared experts
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    inds_list = inds.reshape(B * S, topk).tolist()
    scores_flat = scores.reshape(B * S, topk)
    x_flat = x.reshape(B * S, H)

    # Get layer index from the module (we'll pass it as attribute)
    layer_idx = getattr(moe_module, '_ef_layer_idx', -1)

    with mx.stream(mx.cpu):
        token_outs = []
        for t in range(B * S):
            x_t = x_flat[t:t+1]
            expert_results = []

            for k_i in range(topk):
                eidx = inds_list[t][k_i]
                score = scores_flat[t, k_i]
                cache_key = (layer_idx, eidx)

                # Try cache first
                cached = expert_cache.get(cache_key)
                if cached is None:
                    # Cache miss: load from mmap and cache
                    cached = load_expert_quantized(mlp, eidx)
                    expert_cache.put(cache_key, cached)

                out = expert_qmm_from_cache(x_t, cached, group_size, bits)
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


def generate(model, tokenizer, prompt, max_tokens, cache_entries=600, verbose=True):
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    moe_indices = []
    for i, layer in enumerate(model.model.layers):
        if is_moe(layer):
            moe_indices.append(i)
            # Tag MoE module with layer index for cache keys
            get_moe(layer)._ef_layer_idx = i

    pin_attention_weights(model)

    expert_cache = QuantizedExpertCache(max_entries=cache_entries)

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_indices)} MoE)")
        print(f"  Expert cache: {cache_entries} entries (~{cache_entries * 11 / 1024:.1f}GB)")
        print(f"  Free: {free_gb():.0f}G")

    for step in range(max_tokens):
        t_token = time.time()

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
                h = streaming_moe_forward(moe_mod, h, expert_cache)
            else:
                h = moe_mod(h)
                mx.eval(h)
            x = x + h
            mx.eval(x)
            moe_time += time.time() - t_m

            if verbose and (i + 1) % 10 == 0:
                mem = free_gb()
                print(f"[L{i+1} {mem:.0f}G]", end=" ", flush=True)

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
                  f"cache: {expert_cache.stats()}", flush=True)

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
        "cache_hit_rate": round(expert_cache.hit_rate, 1),
        "cache_entries": len(expert_cache.cache),
        "cache_gb": round(expert_cache.size_gb, 2),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--cache-entries", type=int, default=600,
                   help="Max quantized experts to cache (600 ≈ 6.4GB)")
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v21 — Quantized Expert Cache")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(4 * 1024**3))
    try:
        mx.set_wired_limit(int(80 * 1024**3))
        print("  Wired: 80GB")
    except:
        pass

    import mlx_lm
    print("  Loading (lazy)...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    try:
        results = generate(model, tokenizer, args.prompt, args.max_tokens,
                          cache_entries=args.cache_entries)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        traceback.print_exc()
        return

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {results['prompt']}{results['output']}")
    print(f"  PREFILL: {results['prefill_s']}s")
    print(f"  DECODE:  {results['decode_avg_s']}s/tok ({results['decode_tok_s']} tok/s)")
    print(f"  Cache: {results['cache_entries']} experts, {results['cache_hit_rate']}% hit, {results['cache_gb']}GB")
    print(f"  Total: {results['total_s']}s for {results['tokens']} tokens")

    outfile = os.path.expanduser(f"~/dev/expertflow/v21_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
