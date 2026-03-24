#!/usr/bin/env python3
"""
ExpertFlow Benchmark — Fits-in-RAM MoE Model
=============================================
Compare native mlx_lm vs ExpertFlow layer-by-layer approaches.
Uses Mixtral-8x7B-4bit (~26GB) which fits entirely in 128GB RAM.

Benchmarks:
  1. Native mlx_lm.generate (baseline — best possible speed)
  2. ExpertFlow v12: layer-by-layer + KV cache + quantized_matmul per expert
  3. ExpertFlow v7-style: layer-by-layer + KV cache + native SwitchGLU

This lets us measure ExpertFlow overhead vs native on a model that fits,
before scaling to models that DON'T fit.
"""

import os, sys, time, json, subprocess, traceback
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

MODEL = os.path.expanduser("~/models/mixtral-8x7b-4bit")
PROMPT = "The capital of France is"
MAX_TOKENS = 20


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


# ═══════════════════════════════════════════════════════════
# Benchmark 1: Native mlx_lm baseline
# ═══════════════════════════════════════════════════════════
def bench_native(model, tokenizer, prompt, max_tokens):
    """Native mlx_lm.stream_generate — gold standard speed."""
    import mlx_lm

    tokens = []
    t_start = time.time()
    first_token_time = None

    for resp in mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if first_token_time is None:
            first_token_time = time.time() - t_start
        tokens.append(resp.text)

    total = time.time() - t_start
    n = len(tokens)
    output = "".join(tokens)

    return {
        "approach": "native_mlx_lm",
        "output": output,
        "tokens": n,
        "prefill_s": round(first_token_time, 3),
        "total_s": round(total, 3),
        "tok_s": round(n / total, 2) if total > 0 else 0,
        "decode_tok_s": round((n-1) / (total - first_token_time), 2) if n > 1 and total > first_token_time else 0,
    }


# ═══════════════════════════════════════════════════════════
# Benchmark 2: ExpertFlow layer-by-layer + native MoE
# ═══════════════════════════════════════════════════════════
def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def get_moe_module(layer):
    """Get the MoE/MLP module regardless of model architecture."""
    if hasattr(layer, 'block_sparse_moe'):
        return layer.block_sparse_moe  # Mixtral
    elif hasattr(layer, 'mlp'):
        return layer.mlp  # GLM, DeepSeek, Qwen
    return None


def is_moe_layer(layer):
    moe = get_moe_module(layer)
    return moe is not None and hasattr(moe, 'gate') and hasattr(moe, 'switch_mlp')


def bench_layerwise_native(model, tokenizer, prompt, max_tokens):
    """Layer-by-layer with native MoE (SwitchGLU/gather_qmm) + KV cache."""
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    generated = []
    t_start = time.time()
    first_token_time = None

    for step in range(max_tokens):
        if step == 0:
            ids = mx.array([input_ids])
        else:
            ids = mx.array([[generated[-1]]])

        x = model.model.embed_tokens(ids)
        mx.eval(x)

        seq_len = ids.shape[1]
        offset = kv_caches[0].offset if not kv_caches[0].empty() else 0
        mask = create_mask(seq_len, offset)

        for i, layer in enumerate(model.model.layers):
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)

            h = layer.post_attention_layernorm(x)
            moe = get_moe_module(layer)
            h = moe(h)
            x = x + h
            mx.eval(x)

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)

        next_id = int(mx.argmax(logits[0, 0]).item())
        generated.append(next_id)

        if first_token_time is None:
            first_token_time = time.time() - t_start

    total = time.time() - t_start
    n = len(generated)
    try:
        output = tokenizer.decode(generated)
    except:
        output = str(generated)

    return {
        "approach": "layerwise_native_moe",
        "output": output,
        "tokens": n,
        "prefill_s": round(first_token_time, 3),
        "total_s": round(total, 3),
        "tok_s": round(n / total, 2) if total > 0 else 0,
        "decode_tok_s": round((n-1) / (total - first_token_time), 2) if n > 1 and total > first_token_time else 0,
    }


# ═══════════════════════════════════════════════════════════
# Benchmark 3: ExpertFlow layer-by-layer + quantized_matmul per expert
# ═══════════════════════════════════════════════════════════
def expert_qmatmul(x, proj, eidx):
    w = proj["weight"][eidx]
    s = proj["scales"][eidx]
    b = proj.get("biases")
    b = b[eidx] if b is not None else None
    return mx.quantized_matmul(x, w, s, b, transpose=True,
                                group_size=proj.group_size, bits=proj.bits)


def streaming_moe_qmatmul(moe_module, x):
    B, S, H = x.shape

    # Gate — handle both Mixtral-style (Linear→argpartition) and GLM-style (MoEGate)
    gates = moe_module.gate(x)
    if isinstance(gates, tuple):
        # GLM/DeepSeek-style gate returns (inds, scores) directly
        inds, scores = gates
    else:
        # Mixtral-style: raw logits → argpartition + softmax
        k = moe_module.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
    mx.eval(inds, scores)

    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp

    # Shared experts
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    x_flat = x.reshape(B * S, H)
    inds_2d = inds.reshape(B * S, topk)
    scores_2d = scores.reshape(B * S, topk)

    token_outs = []
    for t in range(B * S):
        x_t = x_flat[t:t+1]
        out = mx.zeros((1, H))
        for k in range(topk):
            eidx = int(inds_2d[t, k].item())
            score = scores_2d[t, k]
            g = expert_qmatmul(x_t, mlp.gate_proj, eidx)
            u = expert_qmatmul(x_t, mlp.up_proj, eidx)
            expert_out = expert_qmatmul(nn.silu(g) * u, mlp.down_proj, eidx)
            out = out + expert_out * score
        token_outs.append(out)

    routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
    if shared is not None:
        result = routed + shared
    else:
        result = routed
    mx.eval(result)
    return result


def bench_layerwise_qmatmul(model, tokenizer, prompt, max_tokens):
    """Layer-by-layer with per-expert quantized_matmul + KV cache."""
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    moe_layers = set()
    for i, layer in enumerate(model.model.layers):
        if is_moe_layer(layer):
            moe_layers.add(i)

    generated = []
    t_start = time.time()
    first_token_time = None

    for step in range(max_tokens):
        if step == 0:
            ids = mx.array([input_ids])
        else:
            ids = mx.array([[generated[-1]]])

        x = model.model.embed_tokens(ids)
        mx.eval(x)

        seq_len = ids.shape[1]
        offset = kv_caches[0].offset if not kv_caches[0].empty() else 0
        mask = create_mask(seq_len, offset)

        for i, layer in enumerate(model.model.layers):
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)

            h = layer.post_attention_layernorm(x)
            moe = get_moe_module(layer)
            if i in moe_layers:
                h = streaming_moe_qmatmul(moe, h)
            else:
                h = moe(h)
                mx.eval(h)
            x = x + h
            mx.eval(x)

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)

        next_id = int(mx.argmax(logits[0, 0]).item())
        generated.append(next_id)

        if first_token_time is None:
            first_token_time = time.time() - t_start

    total = time.time() - t_start
    n = len(generated)
    try:
        output = tokenizer.decode(generated)
    except:
        output = str(generated)

    return {
        "approach": "layerwise_qmatmul",
        "output": output,
        "tokens": n,
        "prefill_s": round(first_token_time, 3),
        "total_s": round(total, 3),
        "tok_s": round(n / total, 2) if total > 0 else 0,
        "decode_tok_s": round((n-1) / (total - first_token_time), 2) if n > 1 and total > first_token_time else 0,
    }


def main():
    print("=" * 60)
    print("  ExpertFlow Benchmark — Mixtral 8x7B 4-bit")
    print("=" * 60)
    print(f"  Model: {os.path.basename(MODEL)}")
    print(f"  Free:  {free_gb():.1f} GB")
    print(f"  Prompt: {PROMPT!r}")
    print(f"  Tokens: {MAX_TOKENS}")

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(8 * 1024**3))

    import mlx_lm
    print("\n  Loading model...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  Layers: {len(model.model.layers)}")

    # Warm up
    print("\n  Warmup...", flush=True)
    _ = mlx_lm.generate(model, tokenizer, "Hi", max_tokens=2)
    print(f"  Warm. Free: {free_gb():.1f} GB")

    results = []

    # Benchmark 1: Native
    print(f"\n{'─'*60}")
    print("  BENCH 1: Native mlx_lm.stream_generate()")
    print(f"{'─'*60}")
    r1 = bench_native(model, tokenizer, PROMPT, MAX_TOKENS)
    results.append(r1)
    print(f"  Output: {r1['output'][:60]}...")
    print(f"  {r1['tok_s']} tok/s overall | {r1['decode_tok_s']} tok/s decode | prefill {r1['prefill_s']}s")

    # Benchmark 2: Layer-wise + native MoE
    print(f"\n{'─'*60}")
    print("  BENCH 2: Layer-wise + native MoE (gather_qmm)")
    print(f"{'─'*60}")
    r2 = bench_layerwise_native(model, tokenizer, PROMPT, MAX_TOKENS)
    results.append(r2)
    print(f"  Output: {r2['output'][:60]}...")
    print(f"  {r2['tok_s']} tok/s overall | {r2['decode_tok_s']} tok/s decode | prefill {r2['prefill_s']}s")

    # Benchmark 3: Layer-wise + quantized_matmul per expert
    print(f"\n{'─'*60}")
    print("  BENCH 3: Layer-wise + quantized_matmul per expert")
    print(f"{'─'*60}")
    r3 = bench_layerwise_qmatmul(model, tokenizer, PROMPT, MAX_TOKENS)
    results.append(r3)
    print(f"  Output: {r3['output'][:60]}...")
    print(f"  {r3['tok_s']} tok/s overall | {r3['decode_tok_s']} tok/s decode | prefill {r3['prefill_s']}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Approach':<30} {'tok/s':>8} {'decode t/s':>10} {'prefill':>10}")
    print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*10}")
    for r in results:
        name = r['approach'][:30]
        print(f"  {name:<30} {r['tok_s']:>8} {r['decode_tok_s']:>10} {r['prefill_s']:>9}s")
    print(f"{'='*60}")

    # Overhead calculation
    if r1['decode_tok_s'] > 0:
        overhead_native_moe = (1 - r2['decode_tok_s'] / r1['decode_tok_s']) * 100
        overhead_qmatmul = (1 - r3['decode_tok_s'] / r1['decode_tok_s']) * 100
        print(f"  Layer-wise native MoE overhead: {overhead_native_moe:.1f}%")
        print(f"  Layer-wise qmatmul overhead: {overhead_qmatmul:.1f}%")

    outfile = os.path.expanduser(f"~/dev/expertflow/bench_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")


if __name__ == "__main__":
    main()
