#!/usr/bin/env python3
"""
ExpertFlow v15 — Batched Expert Matmul + Prefetch
==================================================
v14 results: 11.4s/tok steady-state on GLM-4.5 (198GB/128GB)
  - moe=10.6s: 89 layers × 8 experts × 3 projections = 2136 qmm calls
  - Each call: ~5ms (mmap page-in + quantized matmul)
  - attn=0.7s: fast, GPU with KV cache

v15 optimizations:
  1. Batch expert matmuls: For single-token decode, all 8 experts see the
     same input x. Pre-slice all 8 experts' weights, then do 8 qmm calls
     without re-evaluating inds each time.
  2. Minimize Python overhead: pre-extract all expert indices before the loop.
  3. Single mx.eval per MoE layer (batch all expert outputs).
  4. Try: thread-based prefetch of next layer's weights.
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


def streaming_moe_forward(moe_module, x, is_mixtral=False):
    """
    Optimized MoE: CPU quantized_matmul with minimized Python overhead.
    """
    B, S, H = x.shape

    # Gate routing
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

    # Shared experts
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    # Pre-extract all expert indices (avoid repeated .item() calls in loop)
    inds_flat = inds.reshape(B * S, topk)
    scores_flat = scores.reshape(B * S, topk)

    # Materialize indices to Python ints in one go
    inds_np = inds_flat.tolist()  # List[List[int]]
    x_flat = x.reshape(B * S, H)

    # Expert compute on CPU
    with mx.stream(mx.cpu):
        token_outs = []
        for t in range(B * S):
            x_t = x_flat[t:t+1]

            # Build all 8 expert outputs in the graph, eval once
            expert_results = []
            for k_i in range(topk):
                eidx = inds_np[t][k_i]
                score = scores_flat[t, k_i]

                # 3 quantized matmuls per expert (gate, up, down)
                gw = mlp.gate_proj["weight"][eidx]
                gs = mlp.gate_proj["scales"][eidx]
                gb = mlp.gate_proj.get("biases")
                gb = gb[eidx] if gb is not None else None

                uw = mlp.up_proj["weight"][eidx]
                us = mlp.up_proj["scales"][eidx]
                ub = mlp.up_proj.get("biases")
                ub = ub[eidx] if ub is not None else None

                dw = mlp.down_proj["weight"][eidx]
                ds = mlp.down_proj["scales"][eidx]
                db = mlp.down_proj.get("biases")
                db = db[eidx] if db is not None else None

                g = mx.quantized_matmul(x_t, gw, gs, gb, transpose=True,
                                         group_size=mlp.gate_proj.group_size,
                                         bits=mlp.gate_proj.bits)
                u = mx.quantized_matmul(x_t, uw, us, ub, transpose=True,
                                         group_size=mlp.up_proj.group_size,
                                         bits=mlp.up_proj.bits)
                activated = nn.silu(g) * u
                expert_out = mx.quantized_matmul(activated, dw, ds, db, transpose=True,
                                                  group_size=mlp.down_proj.group_size,
                                                  bits=mlp.down_proj.bits)
                expert_results.append(expert_out * score)

            # Sum all expert outputs
            out = expert_results[0]
            for er in expert_results[1:]:
                out = out + er
            token_outs.append(out)

        routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
        result = (routed + shared) if shared is not None else routed
        # Single eval for entire MoE layer
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

    is_mixtral = hasattr(model.model.layers[0], 'block_sparse_moe')
    moe_count = sum(1 for l in model.model.layers if is_moe(l))

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({moe_count} MoE)")

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
                  f"attn={attn_time:.1f}s moe={moe_time:.1f}s", flush=True)

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
    print("  ExpertFlow v15 — Batched Expert QMM + KV Cache")
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
        f"~/dev/expertflow/v15_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
