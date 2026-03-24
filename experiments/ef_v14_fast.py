#!/usr/bin/env python3
"""
ExpertFlow v14 — Fastest Oversized MoE Engine
==============================================
Benchmark insights:
  - Fits-in-RAM: native gather_qmm is fastest (64 tok/s)
  - Oversized (198GB/128GB): per-expert access is 5.6x faster than native
    because gather_qmm pages in ALL 160 experts, per-expert only pages 8
  - v9 CPU dequant: 56s/tok decode (dequant=36s, matmul=8s, rest=I/O)
  - quantized_matmul eliminates 36s dequant cost

v14 approach:
  - GPU attention + native KV cache (proven 0.4s)
  - Per-expert quantized_matmul (no dequant, ~1/5 the compute of v9)
  - Batch eval per MoE layer (not per expert)
  - Minimize Python loop overhead
  - No expert caching (dequantized cache = OOM, quantized slices are mmap'd)
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


def get_moe(layer):
    if hasattr(layer, 'block_sparse_moe'):
        return layer.block_sparse_moe
    elif hasattr(layer, 'mlp'):
        return layer.mlp
    return None


def is_moe(layer):
    m = get_moe(layer)
    return m is not None and hasattr(m, 'gate') and hasattr(m, 'switch_mlp')


def expert_qmm(x, proj, eidx):
    """Single expert quantized matmul — no dequantization."""
    w = proj["weight"][eidx]
    s = proj["scales"][eidx]
    b = proj.get("biases")
    b = b[eidx] if b is not None else None
    return mx.quantized_matmul(x, w, s, b, transpose=True,
                                group_size=proj.group_size, bits=proj.bits)


def streaming_moe_forward(moe_module, x, is_mixtral=False):
    """
    MoE forward using per-expert quantized_matmul on CPU.
    CPU mode avoids Metal command buffer overhead for mmap page faults.
    """
    B, S, H = x.shape

    # Gate routing (fast, small matmul)
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

    # Shared experts (dense, small — GPU is fine)
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    # Per-token expert computation on CPU
    # CPU handles mmap page faults more efficiently than Metal
    with mx.stream(mx.cpu):
        x_flat = x.reshape(B * S, H)
        inds_2d = inds.reshape(B * S, topk)
        scores_2d = scores.reshape(B * S, topk)

        token_outs = []
        for t in range(B * S):
            x_t = x_flat[t:t+1]
            out = mx.zeros((1, H))

            for k_i in range(topk):
                eidx = int(inds_2d[t, k_i].item())
                score = scores_2d[t, k_i]

                g = expert_qmm(x_t, mlp.gate_proj, eidx)
                u = expert_qmm(x_t, mlp.up_proj, eidx)
                expert_out = expert_qmm(nn.silu(g) * u, mlp.down_proj, eidx)
                out = out + expert_out * score

            token_outs.append(out)

        routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
        result = (routed + shared) if shared is not None else routed
        mx.eval(result)

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

    # Detect architecture
    is_mixtral = hasattr(model.model.layers[0], 'block_sparse_moe')

    moe_count = sum(1 for l in model.model.layers if is_moe(l))
    dense_count = num_layers - moe_count

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({moe_count} MoE, {dense_count} dense)")
        print(f"  Arch: {'Mixtral' if is_mixtral else 'GLM/DeepSeek'}")

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
            # Attention with KV cache
            t_a = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)
            attn_time += time.time() - t_a

            # MLP/MoE
            t_m = time.time()
            h = layer.post_attention_layernorm(x)

            if is_moe(layer):
                h = streaming_moe_forward(get_moe(layer), h, is_mixtral)
            else:
                h = get_moe(layer)(h)
                mx.eval(h)

            x = x + h
            mx.eval(x)
            moe_time += time.time() - t_m

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
            mem = free_gb()
            print(f"\n  ✅ Token {step+1} ({mode}): {text!r} | "
                  f"{dt:.1f}s ({1/dt:.3f} tok/s) | "
                  f"attn={attn_time:.1f}s moe={moe_time:.1f}s | "
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
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=5)
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v14 — Per-Expert QMM + KV Cache")
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
        f"~/dev/expertflow/v14_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
