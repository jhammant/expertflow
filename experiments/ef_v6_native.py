#!/usr/bin/env python3
"""
ExpertFlow v6 — Native MLX-LM Generation
=========================================
Strategy: Use mlx_lm's built-in generate with lazy loading.
MLX-LM already handles KV cache, attention, and MoE dispatch natively.
With lazy=True, weights are mmap'd from disk — only active pages in RAM.

Approach 1: Pure native mlx_lm.stream_generate()
Approach 2: If OOM, monkey-patch MoE layers to reduce peak memory
"""

import os, sys, time, json, subprocess, traceback
os.environ["MLX_LAZY_INITIALIZATION"] = "1"


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


def try_native_generate(model_path, prompt, max_tokens):
    """Approach 1: Pure native mlx_lm generation."""
    import mlx.core as mx
    import mlx_lm

    # Conservative memory limits — leave headroom for OS
    mx.set_memory_limit(int(100 * 1024**3))  # 100 GB max GPU/unified
    mx.set_cache_limit(int(8 * 1024**3))     # 8 GB compute cache
    try:
        mx.set_wired_limit(int(80 * 1024**3))
    except:
        pass

    print(f"  Loading model (lazy)...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_path, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  Free memory: {free_gb():.1f} GB", flush=True)

    print(f"\n  Generating (native mlx_lm)...", flush=True)
    print(f"  Prompt: {prompt!r}", flush=True)

    t_start = time.time()
    tokens_out = []
    try:
        for resp in mlx_lm.stream_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
        ):
            tokens_out.append(resp.text)
            elapsed = time.time() - t_start
            tps = resp.generation_tps if hasattr(resp, 'generation_tps') else 0
            print(f"  Token {resp.generation_tokens}: {resp.text!r} "
                  f"({elapsed:.1f}s, {tps:.3f} tok/s, "
                  f"peak {resp.peak_memory:.1f}GB)", flush=True)
    except Exception as e:
        error_msg = str(e)
        print(f"\n  NATIVE FAILED: {error_msg[:200]}", flush=True)
        return None, error_msg

    total_time = time.time() - t_start
    output = "".join(tokens_out)
    return {
        "approach": "native",
        "prompt": prompt,
        "output": output,
        "tokens": len(tokens_out),
        "total_s": round(total_time, 1),
        "tok_s": round(len(tokens_out) / total_time, 4) if total_time > 0 else 0,
    }, None


def try_hybrid_generate(model_path, prompt, max_tokens):
    """
    Approach 2: Native attention + KV cache, but monkey-patch MoE
    to stream experts with lower peak memory.

    Key insight: Keep mlx_lm's cache/attention infrastructure intact.
    Only intercept the MoE forward pass to control memory.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(8 * 1024**3))
    try:
        mx.set_wired_limit(int(80 * 1024**3))
    except:
        pass

    print(f"  Loading model (lazy)...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_path, lazy=True)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s", flush=True)

    # Count MoE vs dense layers
    moe_count = 0
    dense_count = 0
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
            moe_count += 1
        else:
            dense_count += 1
    print(f"  Layers: {moe_count} MoE + {dense_count} dense = {moe_count + dense_count} total")

    # Monkey-patch MoE layers with streaming version
    original_moe_calls = {}
    patched_count = 0

    def make_streaming_moe(moe_module, layer_idx):
        """Create a streaming MoE forward that processes experts sequentially
        and evaluates eagerly to bound memory."""
        original_call = moe_module.__call__

        def streaming_call(x):
            # Use gate to get routing
            inds, scores = moe_module.gate(x)
            mx.eval(inds, scores)

            B, S, H = x.shape
            ne = inds.shape[-1]  # num experts per token

            # Shared experts (dense — run normally)
            shared = None
            if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
                shared = moe_module.shared_experts(x)

            # Routed experts — use native SwitchGLU but eval eagerly
            y = moe_module.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
            mx.eval(y)

            if shared is not None:
                y = y + shared
                mx.eval(y)

            return y

        return streaming_call

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
            original_moe_calls[i] = layer.mlp.__call__
            layer.mlp.__call__ = make_streaming_moe(layer.mlp, i)
            patched_count += 1

    print(f"  Patched {patched_count} MoE layers for streaming")

    # Now use native generate with KV cache
    print(f"\n  Generating (hybrid: native cache + streaming MoE)...", flush=True)
    print(f"  Prompt: {prompt!r}", flush=True)

    t_start = time.time()
    tokens_out = []
    try:
        for resp in mlx_lm.stream_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
        ):
            tokens_out.append(resp.text)
            elapsed = time.time() - t_start
            tps = resp.generation_tps if hasattr(resp, 'generation_tps') else 0
            print(f"  Token {resp.generation_tokens}: {resp.text!r} "
                  f"({elapsed:.1f}s, {tps:.3f} tok/s, "
                  f"peak {resp.peak_memory:.1f}GB)", flush=True)
    except Exception as e:
        error_msg = str(e)
        print(f"\n  HYBRID FAILED: {error_msg[:300]}", flush=True)
        traceback.print_exc()
        return None, error_msg

    total_time = time.time() - t_start
    output = "".join(tokens_out)

    # Restore original MoE calls
    for i, orig in original_moe_calls.items():
        model.model.layers[i].mlp.__call__ = orig

    return {
        "approach": "hybrid",
        "prompt": prompt,
        "output": output,
        "tokens": len(tokens_out),
        "total_s": round(total_time, 1),
        "tok_s": round(len(tokens_out) / total_time, 4) if total_time > 0 else 0,
        "moe_layers": moe_count,
        "dense_layers": dense_count,
    }, None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=5)
    p.add_argument("--approach", choices=["native", "hybrid", "auto"], default="auto")
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v6 — Native MLX-LM + Streaming MoE")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")
    print(f"  Approach: {args.approach}")

    results = None
    error = None

    if args.approach in ("native", "auto"):
        print(f"\n{'─'*60}")
        print(f"  ATTEMPT 1: Native mlx_lm.stream_generate()")
        print(f"{'─'*60}")
        results, error = try_native_generate(args.model, args.prompt, args.max_tokens)

    if results is None and args.approach in ("hybrid", "auto"):
        print(f"\n{'─'*60}")
        print(f"  ATTEMPT 2: Hybrid (native cache + streaming MoE)")
        print(f"{'─'*60}")
        results, error = try_hybrid_generate(args.model, args.prompt, args.max_tokens)

    print(f"\n{'='*60}")
    if results:
        print(f"  SUCCESS [{results['approach']}]")
        print(f"  Output: {results['prompt']}{results['output']}")
        print(f"  Tokens: {results['tokens']}")
        print(f"  Speed:  {results['tok_s']} tok/s")
        print(f"  Total:  {results['total_s']}s")

        outfile = os.path.expanduser(
            f"~/dev/expertflow/v6_{results['approach']}_{time.strftime('%H%M%S')}.json"
        )
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved:  {outfile}")
    else:
        print(f"  FAILED: {error[:200] if error else 'unknown'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
