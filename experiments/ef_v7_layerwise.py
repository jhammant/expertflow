#!/usr/bin/env python3
"""
ExpertFlow v7 — Layer-wise Forward with Native KV Cache
========================================================
Strategy:
  - Use mlx_lm's native KVCache objects for proper attention caching
  - Iterate layers manually with mx.eval() after each for memory control
  - During decode, only process the LAST token (KV cache handles history)
  - Native SwitchGLU/gather_qmm for MoE dispatch (no manual dequant)
  - Eager eval per layer prevents Metal OOM from graph explosion

This should give us:
  1. Proper KV caching (no re-processing the full sequence)
  2. Bounded memory (eval per layer frees intermediates)
  3. Native MLX performance for matmuls
"""

import os, sys, time, json, subprocess, traceback
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn


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
    """Create causal attention mask."""
    if seq_len == 1:
        return None  # Single token decode — no mask needed
    # Prefill: standard causal mask
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def layer_forward(layer, x, mask, cache):
    """Forward one decoder layer with KV cache."""
    # Attention with KV cache
    h = layer.input_layernorm(x)
    h = layer.self_attn(h, mask, cache)
    x = x + h

    # MLP / MoE
    h = layer.post_attention_layernorm(x)
    h = layer.mlp(h)
    x = x + h

    return x


def generate(model, tokenizer, prompt, max_tokens, verbose=True):
    """
    Generate tokens using layer-by-layer forward with native KV cache.
    """
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)

    # Create KV cache for each layer
    kv_caches = [KVCache() for _ in range(num_layers)]

    generated_ids = []
    token_times = []
    layer_times_log = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers}")

    for step in range(max_tokens):
        t_token = time.time()

        if step == 0:
            # Prefill: process entire prompt
            ids = mx.array([input_ids])
        else:
            # Decode: only the new token (cache has history)
            ids = mx.array([[generated_ids[-1]]])

        # Embed
        x = model.model.embed_tokens(ids)
        mx.eval(x)

        # Attention mask
        seq_len = ids.shape[1]
        cache_offset = kv_caches[0].offset if not kv_caches[0].empty() else 0
        mask = create_mask(seq_len, cache_offset)

        # Layer-by-layer forward with eager eval
        step_layer_times = []
        for i, layer in enumerate(model.model.layers):
            t_layer = time.time()
            x = layer_forward(layer, x, mask, kv_caches[i])
            mx.eval(x)
            # Also eval the cache state to materialize it
            if not kv_caches[i].empty():
                mx.eval(kv_caches[i].state)
            lt = time.time() - t_layer
            step_layer_times.append(lt)

        layer_times_log.append(step_layer_times)

        # Final norm + logits
        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)

        # Greedy sample
        next_id = int(mx.argmax(logits[0, 0]).item())
        generated_ids.append(next_id)

        dt = time.time() - t_token
        token_times.append(dt)

        if verbose:
            try:
                text = tokenizer.decode([next_id])
            except:
                text = f"[{next_id}]"

            avg_layer = sum(step_layer_times) / len(step_layer_times)
            mode = "prefill" if step == 0 else "decode"
            mem = free_gb()
            print(f"  Token {step+1} ({mode}): {text!r} | "
                  f"{dt:.1f}s ({1/dt:.3f} tok/s) | "
                  f"avg layer {avg_layer*1000:.0f}ms | "
                  f"mem {mem:.0f}G free", flush=True)

    # Decode full output
    try:
        output_text = tokenizer.decode(generated_ids)
    except:
        output_text = str(generated_ids)

    prefill_time = token_times[0] if token_times else 0
    decode_times = token_times[1:] if len(token_times) > 1 else []
    decode_avg = sum(decode_times) / len(decode_times) if decode_times else 0

    # Layer time analysis
    if len(layer_times_log) > 1:
        decode_layer_times = layer_times_log[1]  # First decode step
        moe_times = []
        dense_times = []
        for i, lt in enumerate(decode_layer_times):
            layer = model.model.layers[i]
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
                moe_times.append(lt)
            else:
                dense_times.append(lt)
        avg_moe = sum(moe_times) / len(moe_times) if moe_times else 0
        avg_dense = sum(dense_times) / len(dense_times) if dense_times else 0
    else:
        avg_moe = avg_dense = 0

    return {
        "prompt": prompt,
        "output": output_text,
        "tokens_generated": len(generated_ids),
        "prefill_s": round(prefill_time, 2),
        "decode_avg_s": round(decode_avg, 2),
        "decode_tok_s": round(1/decode_avg, 4) if decode_avg > 0 else 0,
        "total_s": round(sum(token_times), 1),
        "avg_moe_layer_ms": round(avg_moe * 1000, 1),
        "avg_dense_layer_ms": round(avg_dense * 1000, 1),
        "num_layers": num_layers,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=5)
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v7 — Layer-wise + Native KV Cache")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")

    # Memory config
    mx.set_memory_limit(int(110 * 1024**3))
    mx.set_cache_limit(int(8 * 1024**3))
    try:
        mx.set_wired_limit(int(80 * 1024**3))
        print("  Wired: 80GB")
    except:
        pass

    # Load model
    import mlx_lm
    print("  Loading model (lazy)...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # Generate
    try:
        results = generate(model, tokenizer, args.prompt, args.max_tokens)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        traceback.print_exc()
        return

    # Print results
    print(f"\n{'='*60}")
    print(f"  OUTPUT: {results['prompt']}{results['output']}")
    print(f"  PREFILL: {results['prefill_s']}s")
    print(f"  DECODE:  {results['decode_avg_s']}s/tok ({results['decode_tok_s']} tok/s)")
    print(f"  MoE layers:   {results['avg_moe_layer_ms']}ms avg")
    print(f"  Dense layers: {results['avg_dense_layer_ms']}ms avg")
    print(f"  Total: {results['total_s']}s for {results['tokens_generated']} tokens")

    outfile = os.path.expanduser(
        f"~/dev/expertflow/v7_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
