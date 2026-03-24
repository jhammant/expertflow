#!/usr/bin/env python3
"""
ExpertFlow v9 — GPU Attention + CPU MoE + Native KV Cache
==========================================================
v7 findings: KV cache works, correct output, 11ms dense, 3400ms MoE.
v8 finding: Manual indexing on GPU = OOM killed.
ef_fixed finding: CPU MoE = 47s decode (but no KV cache, so repeated output).

v9 strategy:
  - GPU: attention + KV cache (11ms/layer, native, correct)
  - CPU: MoE expert dequant + matmul (avoids GPU OOM on expert weights)
  - LRU cache: keep hot expert weights in RAM across tokens
  - Only process last token on decode (KV cache handles history)

Target: Combine v7's correct KV caching with CPU-mode MoE.
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


class ExpertCache:
    """LRU cache for dequantized expert weights."""
    def __init__(self, max_experts=400):
        self.cache = OrderedDict()
        self.max_experts = max_experts
        self.hits = 0
        self.misses = 0

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
        while len(self.cache) >= self.max_experts:
            self.cache.popitem(last=False)
        self.cache[key] = value

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0


def dequant_expert(proj, eidx):
    """Dequant one expert from a QuantizedSwitchLinear, on CPU."""
    w = proj["weight"][eidx]
    s = proj["scales"][eidx]
    b = proj.get("biases")
    b = b[eidx] if b is not None else None
    return mx.dequantize(w, s, b, group_size=proj.group_size, bits=proj.bits)


def streaming_moe(moe_module, x, layer_idx, expert_cache):
    """
    MoE with CPU expert dequant + matmul. Gate runs wherever default device is.
    """
    B, S, H = x.shape

    # Route (fast — just a linear + topk)
    inds, scores = moe_module.gate(x)
    mx.eval(inds, scores)

    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp

    # Shared experts (dense, fast)
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    x_flat = x.reshape(B * S, H)
    inds_2d = inds.reshape(B * S, topk)
    scores_2d = scores.reshape(B * S, topk)

    # Collect unique experts needed
    unique_experts = set()
    for t in range(B * S):
        for k in range(topk):
            unique_experts.add(int(inds_2d[t, k].item()))

    # Load/dequant needed experts on CPU
    expert_weights = {}
    for eidx in unique_experts:
        key = (layer_idx, eidx)
        cached = expert_cache.get(key)
        if cached is not None:
            expert_weights[eidx] = cached
        else:
            with mx.stream(mx.cpu):
                gw = dequant_expert(mlp.gate_proj, eidx)
                uw = dequant_expert(mlp.up_proj, eidx)
                dw = dequant_expert(mlp.down_proj, eidx)
                mx.eval(gw, uw, dw)
            expert_weights[eidx] = (gw, uw, dw)
            expert_cache.put(key, (gw, uw, dw))

    # Compute expert outputs on CPU
    with mx.stream(mx.cpu):
        token_outs = []
        for t in range(B * S):
            x_t = x_flat[t:t+1]
            out = mx.zeros((1, H))

            for k in range(topk):
                eidx = int(inds_2d[t, k].item())
                score = scores_2d[t, k]
                gw, uw, dw = expert_weights[eidx]

                g = x_t @ gw.T
                u = x_t @ uw.T
                expert_out = (nn.silu(g) * u) @ dw.T
                out = out + expert_out * score

            token_outs.append(out)

        routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)

        if shared is not None:
            result = routed + shared
        else:
            result = routed

        mx.eval(result)

    return result


def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def generate(model, tokenizer, prompt, max_tokens, cache_size=400, verbose=True):
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]
    expert_cache = ExpertCache(max_experts=cache_size)

    moe_layers = set()
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
            moe_layers.add(i)

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_layers)} MoE, {num_layers - len(moe_layers)} dense)")
        print(f"  Expert cache: {cache_size} experts")
        print(f"  Strategy: GPU attention + CPU MoE + KV cache")

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
        dense_time = 0

        for i, layer in enumerate(model.model.layers):
            # Attention with KV cache (GPU)
            t_a = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)
            attn_time += time.time() - t_a

            # MLP/MoE
            t_m = time.time()
            h = layer.post_attention_layernorm(x)

            if i in moe_layers:
                h = streaming_moe(layer.mlp, h, i, expert_cache)
            else:
                h = layer.mlp(h)
                mx.eval(h)

            x = x + h
            mx.eval(x)

            if i in moe_layers:
                moe_time += time.time() - t_m
            else:
                dense_time += time.time() - t_m

            if verbose and (i + 1) % 10 == 0:
                print(f"[L{i+1}/{num_layers}]", end=" ", flush=True)

        # Final logits
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
            mem = free_gb()
            print(f"\n  ✅ Token {step+1} ({mode}): {text!r} | "
                  f"{dt:.1f}s ({1/dt:.3f} tok/s) | "
                  f"attn={attn_time:.1f}s moe={moe_time:.1f}s dense={dense_time:.2f}s | "
                  f"cache {expert_cache.hit_rate:.0f}% hit | "
                  f"mem {mem:.0f}G", flush=True)

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
        "expert_cache_hit_rate": round(expert_cache.hit_rate, 1),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=5)
    p.add_argument("--cache-size", type=int, default=400)
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v9 — GPU Attn + CPU MoE + KV Cache")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(8 * 1024**3))
    try:
        mx.set_wired_limit(int(80 * 1024**3))
        print("  Wired: 80GB")
    except:
        pass

    import mlx_lm
    print("  Loading model (lazy)...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    try:
        results = generate(model, tokenizer, args.prompt, args.max_tokens, args.cache_size)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        traceback.print_exc()
        return

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {results['prompt']}{results['output']}")
    print(f"  PREFILL: {results['prefill_s']}s")
    print(f"  DECODE:  {results['decode_avg_s']}s/tok ({results['decode_tok_s']} tok/s)")
    print(f"  Expert cache: {results['expert_cache_hit_rate']}% hit")
    print(f"  Total: {results['total_s']}s for {results['tokens']} tokens")

    outfile = os.path.expanduser(
        f"~/dev/expertflow/v9_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
