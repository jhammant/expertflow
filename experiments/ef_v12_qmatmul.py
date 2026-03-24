#!/usr/bin/env python3
"""
ExpertFlow v12 — Quantized Matmul per Expert (No Dequant)
==========================================================
v10 profiling: dequant=36s (78%), matmul=8s (18%).

Key optimization: mx.quantized_matmul operates on quantized int4 weights
directly — NO dequantization needed. We extract just the active expert's
quantized weight slice and do quantized_matmul.

This should eliminate the 36s dequant cost entirely.
Each expert slice: ~1MB quantized (vs ~28MB dequantized float16).
8 experts × 3 projections × 89 layers = ~2.1GB I/O per token.
At 7.4 GB/s NVMe → ~0.3s theoretical minimum.

Combined with KV cache from v7 (working, correct output).
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


def expert_qmatmul(x, proj, eidx):
    """
    Quantized matmul for a single expert — no dequantization.
    x: [1, hidden_dim]
    proj: QuantizedSwitchLinear with weight[num_experts, out_dim, packed_in]
    Returns: [1, out_dim]
    """
    w = proj["weight"][eidx]       # [out_dim, packed_in] — still quantized int4
    s = proj["scales"][eidx]       # [out_dim, n_groups]
    b = proj.get("biases")
    b = b[eidx] if b is not None else None
    return mx.quantized_matmul(
        x, w, s, b,
        transpose=True,
        group_size=proj.group_size,
        bits=proj.bits,
    )


def streaming_moe(moe_module, x, layer_idx, profiler):
    """
    MoE forward using quantized_matmul per active expert.
    No dequantization. Only reads quantized data for active experts.
    """
    B, S, H = x.shape

    # Gate
    t0 = time.time()
    inds, scores = moe_module.gate(x)
    mx.eval(inds, scores)
    profiler['gate'] += time.time() - t0

    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp

    # Shared experts
    t0 = time.time()
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)
    profiler['shared'] += time.time() - t0

    # Routed experts via quantized_matmul
    t0 = time.time()
    x_flat = x.reshape(B * S, H)
    inds_2d = inds.reshape(B * S, topk)
    scores_2d = scores.reshape(B * S, topk)

    token_outs = []
    for t in range(B * S):
        x_t = x_flat[t:t+1]  # [1, H]
        out = mx.zeros((1, H))

        for k in range(topk):
            eidx = int(inds_2d[t, k].item())
            score = scores_2d[t, k]

            # SwiGLU with quantized_matmul — no dequant!
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
    profiler['qmatmul'] += time.time() - t0

    return result


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
        print(f"  Strategy: quantized_matmul per expert (no dequant) + KV cache")

    for step in range(max_tokens):
        t_token = time.time()
        profiler = {'gate': 0, 'shared': 0, 'qmatmul': 0}

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

        for i, layer in enumerate(model.model.layers):
            # Attention with KV cache
            t_a = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)
            attn_time += time.time() - t_a

            # MLP/MoE
            h = layer.post_attention_layernorm(x)
            if i in moe_layer_set:
                h = streaming_moe(layer.mlp, h, i, profiler)
            else:
                h = layer.mlp(h)
                mx.eval(h)
            x = x + h
            mx.eval(x)

            if verbose and (i + 1) % 10 == 0:
                mem = free_gb()
                print(f"[L{i+1} {mem:.0f}G]", end=" ", flush=True)

        # Logits
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
            moe_total = profiler['gate'] + profiler['shared'] + profiler['qmatmul']
            print(f"\n  ✅ Token {step+1} ({mode}): {text!r} | "
                  f"{dt:.1f}s ({1/dt:.3f} tok/s) | "
                  f"attn={attn_time:.1f}s | "
                  f"gate={profiler['gate']:.1f}s "
                  f"shared={profiler['shared']:.1f}s "
                  f"qmm={profiler['qmatmul']:.1f}s "
                  f"moe_total={moe_total:.1f}s", flush=True)

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
    print("  ExpertFlow v12 — Quantized Matmul (No Dequant) + KV Cache")
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
        f"~/dev/expertflow/v12_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
