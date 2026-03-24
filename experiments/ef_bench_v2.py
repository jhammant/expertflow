#!/usr/bin/env python3
"""
ExpertFlow Benchmark v2 — Optimization Iterations
==================================================
Baseline: layer-wise native MoE = 36.2 tok/s (43.6% overhead vs native 64.2).

Optimizations to try:
  1. Reduce eval frequency: eval every N layers instead of every layer
  2. Skip eval on attention-only step (combine attn+mlp then eval)
  3. Use model.__call__ directly with cache (native forward, no per-layer)
  4. Profile: is overhead from eval() sync, Python loop, or memory?
"""

import os, sys, time, json, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

MODEL = os.path.expanduser("~/models/mixtral-8x7b-4bit")
PROMPT = "The capital of France is"
MAX_TOKENS = 30


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


def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def get_moe(layer):
    if hasattr(layer, 'block_sparse_moe'):
        return layer.block_sparse_moe
    elif hasattr(layer, 'mlp'):
        return layer.mlp
    return None


# ═══ Native baseline ═══
def bench_native(model, tokenizer):
    import mlx_lm
    tokens = []
    t_start = time.time()
    ft = None
    for resp in mlx_lm.stream_generate(model, tokenizer, PROMPT, max_tokens=MAX_TOKENS):
        if ft is None: ft = time.time() - t_start
        tokens.append(resp.text)
    total = time.time() - t_start
    n = len(tokens)
    return {
        "name": "native",
        "output": "".join(tokens),
        "tokens": n,
        "prefill_s": round(ft, 4),
        "total_s": round(total, 3),
        "decode_tok_s": round((n-1) / (total - ft), 2) if n > 1 else 0,
    }


# ═══ Layer-wise eval every layer (v7 approach) ═══
def bench_eval_every_layer(model, tokenizer):
    from mlx_lm.models.cache import KVCache
    input_ids = tokenizer.encode(PROMPT)
    n_layers = len(model.model.layers)
    kv = [KVCache() for _ in range(n_layers)]
    gen = []
    t_start = time.time()
    ft = None

    for step in range(MAX_TOKENS):
        ids = mx.array([input_ids]) if step == 0 else mx.array([[gen[-1]]])
        x = model.model.embed_tokens(ids)
        mx.eval(x)
        offset = kv[0].offset if not kv[0].empty() else 0
        mask = create_mask(ids.shape[1], offset)

        for i, layer in enumerate(model.model.layers):
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv[i])
            x = x + h
            mx.eval(x)
            h = layer.post_attention_layernorm(x)
            h = get_moe(layer)(h)
            x = x + h
            mx.eval(x)

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)
        gen.append(int(mx.argmax(logits[0, 0]).item()))
        if ft is None: ft = time.time() - t_start

    total = time.time() - t_start
    n = len(gen)
    return {
        "name": "eval_every_layer",
        "output": tokenizer.decode(gen),
        "tokens": n,
        "prefill_s": round(ft, 4),
        "total_s": round(total, 3),
        "decode_tok_s": round((n-1) / (total - ft), 2) if n > 1 else 0,
    }


# ═══ Eval once per layer (combine attn+mlp) ═══
def bench_eval_combined(model, tokenizer):
    from mlx_lm.models.cache import KVCache
    input_ids = tokenizer.encode(PROMPT)
    n_layers = len(model.model.layers)
    kv = [KVCache() for _ in range(n_layers)]
    gen = []
    t_start = time.time()
    ft = None

    for step in range(MAX_TOKENS):
        ids = mx.array([input_ids]) if step == 0 else mx.array([[gen[-1]]])
        x = model.model.embed_tokens(ids)
        mx.eval(x)
        offset = kv[0].offset if not kv[0].empty() else 0
        mask = create_mask(ids.shape[1], offset)

        for i, layer in enumerate(model.model.layers):
            # Full layer forward — single eval
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv[i])
            x = x + h
            h = layer.post_attention_layernorm(x)
            h = get_moe(layer)(h)
            x = x + h
            mx.eval(x)  # One eval per layer instead of two

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)
        gen.append(int(mx.argmax(logits[0, 0]).item()))
        if ft is None: ft = time.time() - t_start

    total = time.time() - t_start
    n = len(gen)
    return {
        "name": "eval_combined",
        "output": tokenizer.decode(gen),
        "tokens": n,
        "prefill_s": round(ft, 4),
        "total_s": round(total, 3),
        "decode_tok_s": round((n-1) / (total - ft), 2) if n > 1 else 0,
    }


# ═══ Eval every N layers ═══
def bench_eval_every_n(model, tokenizer, n_eval=4):
    from mlx_lm.models.cache import KVCache
    input_ids = tokenizer.encode(PROMPT)
    n_layers = len(model.model.layers)
    kv = [KVCache() for _ in range(n_layers)]
    gen = []
    t_start = time.time()
    ft = None

    for step in range(MAX_TOKENS):
        ids = mx.array([input_ids]) if step == 0 else mx.array([[gen[-1]]])
        x = model.model.embed_tokens(ids)
        mx.eval(x)
        offset = kv[0].offset if not kv[0].empty() else 0
        mask = create_mask(ids.shape[1], offset)

        for i, layer in enumerate(model.model.layers):
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv[i])
            x = x + h
            h = layer.post_attention_layernorm(x)
            h = get_moe(layer)(h)
            x = x + h
            if (i + 1) % n_eval == 0 or i == n_layers - 1:
                mx.eval(x)

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)
        gen.append(int(mx.argmax(logits[0, 0]).item()))
        if ft is None: ft = time.time() - t_start

    total = time.time() - t_start
    n = len(gen)
    return {
        "name": f"eval_every_{n_eval}",
        "output": tokenizer.decode(gen),
        "tokens": n,
        "prefill_s": round(ft, 4),
        "total_s": round(total, 3),
        "decode_tok_s": round((n-1) / (total - ft), 2) if n > 1 else 0,
    }


# ═══ Native model.__call__ with cache (no per-layer iteration) ═══
def bench_model_call(model, tokenizer):
    from mlx_lm.models.cache import make_prompt_cache
    input_ids = tokenizer.encode(PROMPT)
    cache = make_prompt_cache(model)
    gen = []
    t_start = time.time()
    ft = None

    for step in range(MAX_TOKENS):
        ids = mx.array([input_ids]) if step == 0 else mx.array([[gen[-1]]])
        logits = model(ids, cache=cache)
        logits = logits[:, -1, :]
        mx.eval(logits)
        gen.append(int(mx.argmax(logits[0]).item()))
        if ft is None: ft = time.time() - t_start

    total = time.time() - t_start
    n = len(gen)
    return {
        "name": "model_call_with_cache",
        "output": tokenizer.decode(gen),
        "tokens": n,
        "prefill_s": round(ft, 4),
        "total_s": round(total, 3),
        "decode_tok_s": round((n-1) / (total - ft), 2) if n > 1 else 0,
    }


def main():
    print("=" * 60)
    print("  ExpertFlow Bench v2 — Optimization Sweep")
    print("=" * 60)
    print(f"  Model: {os.path.basename(MODEL)}")
    print(f"  Free:  {free_gb():.1f} GB")
    print(f"  Prompt: {PROMPT!r}")
    print(f"  Tokens: {MAX_TOKENS}")

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(8 * 1024**3))

    import mlx_lm
    print("\n  Loading model...", flush=True)
    model, tokenizer = mlx_lm.load(MODEL)

    # Warm up
    _ = mlx_lm.generate(model, tokenizer, "Hi", max_tokens=2)
    print(f"  Ready. Free: {free_gb():.1f} GB\n")

    benchmarks = [
        ("Native mlx_lm", lambda: bench_native(model, tokenizer)),
        ("model(ids, cache)", lambda: bench_model_call(model, tokenizer)),
        ("Eval combined (1/layer)", lambda: bench_eval_combined(model, tokenizer)),
        ("Eval every layer (2/layer)", lambda: bench_eval_every_layer(model, tokenizer)),
        ("Eval every 4 layers", lambda: bench_eval_every_n(model, tokenizer, 4)),
        ("Eval every 8 layers", lambda: bench_eval_every_n(model, tokenizer, 8)),
        ("Eval every 16 layers", lambda: bench_eval_every_n(model, tokenizer, 16)),
    ]

    results = []
    for name, fn in benchmarks:
        print(f"  Running: {name}...", end=" ", flush=True)
        r = fn()
        results.append(r)
        correct = r['output'][:20] == results[0]['output'][:20] if results else True
        mark = "✓" if correct else "✗"
        print(f"{r['decode_tok_s']} tok/s  {mark}", flush=True)

    print(f"\n{'='*60}")
    print(f"  {'Approach':<30} {'decode t/s':>10} {'prefill':>8} {'overhead':>8}")
    print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*8}")
    baseline = results[0]['decode_tok_s']
    for r in results:
        overhead = (1 - r['decode_tok_s'] / baseline) * 100 if baseline > 0 else 0
        print(f"  {r['name']:<30} {r['decode_tok_s']:>10} {r['prefill_s']:>7}s {overhead:>7.1f}%")
    print(f"{'='*60}")

    outfile = os.path.expanduser(f"~/dev/expertflow/bench2_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")


if __name__ == "__main__":
    main()
