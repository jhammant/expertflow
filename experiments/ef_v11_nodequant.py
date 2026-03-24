#!/usr/bin/env python3
"""
ExpertFlow v11 — Skip Dequant: Use gather_qmm per expert
=========================================================
v10 profiling: dequant=36s (78%), matmul=8s (18%), attn=1.2s.

Key insight: mx.gather_qmm operates on QUANTIZED weights directly.
Instead of: dequant → float16 matmul (expensive dequant)
Do: gather_qmm with single expert index (no dequant, compute on quantized)

The native SwitchGLU already does this, but it builds a graph touching all experts.
Solution: Call gather_qmm PER LAYER with only the active expert indices,
then mx.eval() per layer to prevent graph explosion.
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


def streaming_moe(moe_module, x, layer_idx):
    """
    MoE using native gather_qmm (no dequant!) but eval'd per-layer.
    The key difference from v7: we eval the MoE result immediately,
    so the computation graph doesn't grow across layers.
    """
    # This just calls the native MoE forward — which uses gather_qmm internally
    # The per-layer eval in the main loop prevents graph explosion
    return moe_module(x)


def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def generate(model, tokenizer, prompt, max_tokens, verbose=True):
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    moe_layer_set = set()
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
            moe_layer_set.add(i)

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_layer_set)} MoE)")
        print(f"  Strategy: native gather_qmm + per-layer eval + KV cache")

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
            # Attention
            t_a = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)
            attn_time += time.time() - t_a

            # MLP/MoE
            t_m = time.time()
            h = layer.post_attention_layernorm(x)
            h = layer.mlp(h)  # Native forward (uses gather_qmm for MoE)
            x = x + h
            mx.eval(x)  # KEY: eval per layer to bound memory
            if i in moe_layer_set:
                moe_time += time.time() - t_m

            if verbose and (i + 1) % 10 == 0:
                mem = free_gb()
                print(f"[L{i+1} {mem:.0f}G]", end=" ", flush=True)

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
            moe_per_layer = moe_time / len(moe_layer_set) * 1000 if moe_layer_set else 0
            print(f"\n  ✅ Token {step+1} ({mode}): {text!r} | "
                  f"{dt:.1f}s ({1/dt:.3f} tok/s) | "
                  f"attn={attn_time:.1f}s moe={moe_time:.1f}s "
                  f"({moe_per_layer:.0f}ms/layer)", flush=True)

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
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=5)
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v11 — Native gather_qmm + KV Cache")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")

    # Tighter memory limits to force aggressive page eviction
    mx.set_memory_limit(int(90 * 1024**3))   # 90GB (leave 38GB for OS)
    mx.set_cache_limit(int(4 * 1024**3))      # 4GB compute cache
    try:
        mx.set_wired_limit(int(60 * 1024**3))  # 60GB wired
        print("  Wired: 60GB")
    except:
        pass

    import mlx_lm
    print("  Loading (lazy)...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    try:
        results = generate(model, tokenizer, args.prompt, args.max_tokens)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        traceback.print_exc()
        return

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {results['prompt']}{results['output']}")
    print(f"  PREFILL: {results['prefill_s']}s")
    print(f"  DECODE:  {results['decode_avg_s']}s/tok ({results['decode_tok_s']} tok/s)")
    print(f"  Total: {results['total_s']}s for {results['tokens']} tokens")

    outfile = os.path.expanduser(
        f"~/dev/expertflow/v11_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
