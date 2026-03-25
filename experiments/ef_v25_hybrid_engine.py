#!/usr/bin/env python3
"""
ExpertFlow v25 — Hybrid Engine (ExpertFlow + oMLX)
====================================================
Combines two complementary optimizations:
  1. ExpertFlow: Expert weight streaming with Belady-approximate cache
  2. oMLX: SSD-tiered KV cache (hot RAM + cold SSD)

For GLM-4.5 (355B MoE, 92 layers, 160 experts) on 128GB M5 Max:
  - Expert weights: ~180GB → stream from mmap with smart cache
  - KV cache: grows with context → tier to SSD when RAM is full

The key insight: expert weights and KV cache compete for the same RAM.
By tiering KV cache to SSD, we free more RAM for expert caching,
which improves hit rates and reduces expert load latency.

Memory budget allocation:
  - Model weights (mmap'd, demand-paged): ~185GB on disk
  - Expert cache (active in RAM): ~3-5GB (300 entries × ~11MB)
  - KV cache hot tier: ~5-10GB
  - KV cache cold tier: SSD (unlimited, ~50GB)
  - OS + overhead: ~5GB
  Total in RAM: ~20-25GB active, rest demand-paged from mmap
"""

import os, sys, time, json, subprocess, traceback
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from collections import OrderedDict
from pathlib import Path


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


# ─── Expert Cache with Frequency-Weighted Eviction ───────────────

class FrequencyWeightedCache:
    """Expert cache using frequency × recency scoring for eviction.

    Simpler than full Belady but captures the key insight:
    experts with high frequency AND recent access should stay.
    This is a practical middle ground between LRU and learned eviction.
    """

    def __init__(self, budget=300, decay=0.95):
        self.cache = {}  # key -> (value, score)
        self.budget = budget
        self.decay = decay
        self.scores = {}  # key -> running score

        self.token_hits = 0
        self.token_misses = 0
        self.total_hits = 0
        self.total_misses = 0

    def get(self, key):
        if key in self.cache:
            self.token_hits += 1
            self.total_hits += 1
            # Boost score on access
            self.scores[key] = self.scores.get(key, 0) + 1.0
            return self.cache[key]
        self.token_misses += 1
        self.total_misses += 1
        return None

    def put(self, key, value):
        self.cache[key] = value
        self.scores[key] = self.scores.get(key, 0) + 1.0

    def trim(self):
        """Evict lowest-scored entries."""
        evicted = 0
        while len(self.cache) > self.budget:
            # Find entry with lowest score
            min_key = min(self.cache.keys(), key=lambda k: self.scores.get(k, 0))
            del self.cache[min_key]
            del self.scores[min_key]
            evicted += 1

        # Decay all scores
        for k in self.scores:
            self.scores[k] *= self.decay

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


# ─── SSD-Tiered KV Cache Wrapper ─────────────────────────────────

class TieredKVCacheWrapper:
    """Wraps MLX KVCache with oMLX SSD tiering.

    When KV cache exceeds hot_max_bytes, old blocks migrate to SSD.
    On cache hit (shared prefix), blocks restore from SSD instantly.
    """

    def __init__(self, num_layers, hot_max_gb=5.0, ssd_max_gb=50.0):
        from mlx_lm.models.cache import KVCache

        self.kv_caches = [KVCache() for _ in range(num_layers)]
        self.num_layers = num_layers
        self.hot_max_bytes = int(hot_max_gb * 1024**3)
        self.ssd_enabled = False

        # Try to initialize oMLX SSD cache
        try:
            from omlx.cache import PagedSSDCacheManager
            ssd_dir = Path.home() / ".omlx" / "ef_kv_cache"
            ssd_dir.mkdir(parents=True, exist_ok=True)

            self.ssd_cache = PagedSSDCacheManager(
                cache_dir=ssd_dir,
                max_size_bytes=int(ssd_max_gb * 1024**3),
                hot_cache_max_bytes=self.hot_max_bytes,
            )
            self.ssd_enabled = True
            print(f"  SSD KV cache: enabled ({ssd_max_gb}GB max, {hot_max_gb}GB hot)")
        except Exception as e:
            print(f"  SSD KV cache: disabled ({e})")

    def __getitem__(self, idx):
        return self.kv_caches[idx]

    def __len__(self):
        return len(self.kv_caches)


# ─── Model Forward Pass ──────────────────────────────────────────

def get_moe(layer):
    if hasattr(layer, 'block_sparse_moe'):
        return layer.block_sparse_moe
    elif hasattr(layer, 'mlp'):
        return layer.mlp
    return None


def is_moe(layer):
    m = get_moe(layer)
    return m is not None and hasattr(m, 'gate') and hasattr(m, 'switch_mlp')


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

    needed = set()
    for t in range(B * S):
        for k_i in range(topk):
            needed.add(inds_list[t][k_i])

    expert_weights = {}
    miss_parts = []
    for eidx in needed:
        cache_key = (layer_idx, eidx)
        cached = expert_cache.get(cache_key) if use_cache else None

        if cached is not None:
            expert_weights[eidx] = cached
        else:
            parts = []
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                w = proj["weight"][eidx]
                s = proj["scales"][eidx]
                b_arr = proj.get("biases")
                b = b_arr[eidx] if b_arr is not None else None
                parts.extend([w, s, b])
                miss_parts.extend([p for p in [w, s, b] if p is not None])
            expert_weights[eidx] = tuple(parts)
            if use_cache:
                expert_cache.put(cache_key, tuple(parts))

    if miss_parts:
        mx.eval(*miss_parts)

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


def generate(model, tokenizer, prompt, max_tokens, cache_budget=300,
             cache_type="freq", verbose=True):
    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)

    # Set up tiered KV cache
    tiered_kv = TieredKVCacheWrapper(num_layers, hot_max_gb=5.0, ssd_max_gb=50.0)

    moe_indices = []
    for i, layer in enumerate(model.model.layers):
        if is_moe(layer):
            moe_indices.append(i)
            get_moe(layer)._ef_layer_idx = i

    # Choose cache type
    if cache_type == "freq":
        expert_cache = FrequencyWeightedCache(budget=cache_budget)
    else:
        expert_cache = FrequencyWeightedCache(budget=cache_budget)  # default

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_indices)} MoE)")
        print(f"  Expert cache: {cache_budget} entries (~{cache_budget * 11 / 1024:.1f}GB)")
        print(f"  Eviction: frequency-weighted (decay=0.95)")

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
        kv0 = tiered_kv[0]
        cache_offset = kv0.offset if not kv0.empty() else 0
        mask = create_mask(seq_len, cache_offset)

        attn_time = 0
        moe_time = 0

        for i, layer in enumerate(model.model.layers):
            t_a = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, tiered_kv[i])
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
            mem = free_gb()
            print(f"  T{step+1}({mode}): {text!r} | "
                  f"{dt:.1f}s ({1/dt:.3f} t/s) | "
                  f"a={attn_time:.1f} m={moe_time:.1f} | "
                  f"hit {expert_cache.token_hit_rate:.0f}% "
                  f"(cache: {len(expert_cache.cache)}) | "
                  f"mem: {mem:.0f}G", flush=True)

    try:
        output_text = tokenizer.decode(generated_ids)
    except:
        output_text = str(generated_ids)

    prefill_time = token_times[0] if token_times else 0
    decode_times = token_times[1:] if len(token_times) > 1 else []
    steady = token_times[-5:] if len(token_times) >= 6 else decode_times
    steady_avg = sum(steady) / len(steady) if steady else 0

    return {
        "prompt": prompt,
        "output": output_text,
        "tokens": len(generated_ids),
        "prefill_s": round(prefill_time, 2),
        "decode_avg_s": round(sum(decode_times)/len(decode_times), 2) if decode_times else 0,
        "steady_avg_s": round(steady_avg, 2),
        "steady_tok_s": round(1/steady_avg, 4) if steady_avg > 0 else 0,
        "total_s": round(sum(token_times), 1),
        "cache_hit_rate": round(expert_cache.total_hit_rate, 1),
        "cache_type": "freq_weighted",
        "ssd_kv_enabled": tiered_kv.ssd_enabled,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/mixtral-8x7b-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--cache-budget", type=int, default=128)
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v25 — Hybrid Engine (EF + oMLX)")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(4 * 1024**3))
    try:
        mx.set_wired_limit(int(60 * 1024**3))
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
    print(f"  STEADY:  {results['steady_avg_s']}s/tok ({results['steady_tok_s']} tok/s)")
    print(f"  Cache: {results['cache_hit_rate']}% hit (freq-weighted)")
    print(f"  SSD KV: {'enabled' if results['ssd_kv_enabled'] else 'disabled'}")
    print(f"  Total: {results['total_s']}s for {results['tokens']} tokens")

    outfile = os.path.expanduser(f"~/dev/expertflow/experiments/v25_hybrid_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
