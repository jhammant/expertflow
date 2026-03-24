#!/usr/bin/env python3
"""
ExpertFlow v20 — Micro-Profile MoE Layer Breakdown
===================================================
v17 steady-state: 10.7s/tok = 120ms per MoE layer.
But WHERE is the 120ms spent?

Profile a single decode token's MoE layers with fine-grained timing:
  - Python overhead (loop, int conversion, dict access)
  - Weight slicing (proj["weight"][eidx] — triggers mmap page fault)
  - quantized_matmul compute
  - mx.eval() sync
  - Gate routing
  - Shared experts
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


def pin_attention_weights(model):
    print("  Pinning attention...", end=" ", flush=True)
    t0 = time.time()
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn, name):
                proj = getattr(attn, name)
                if hasattr(proj, 'weight'):
                    mx.eval(proj.weight.sum())
        for ln in ['input_layernorm', 'post_attention_layernorm']:
            if hasattr(layer, ln):
                mx.eval(getattr(layer, ln).weight.sum())
    mx.eval(model.model.embed_tokens.weight.sum())
    if hasattr(model.lm_head, 'weight'):
        mx.eval(model.lm_head.weight.sum())
    if hasattr(model.model, 'norm'):
        mx.eval(model.model.norm.weight.sum())
    print(f"done ({time.time()-t0:.1f}s)")


def profiled_moe_forward(moe_module, x, prof):
    """MoE with fine-grained per-operation timing."""
    B, S, H = x.shape

    # Gate
    t0 = time.time()
    gates_out = moe_module.gate(x)
    if isinstance(gates_out, tuple):
        inds, scores = gates_out
    else:
        k = moe_module.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates_out, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates_out, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
    mx.eval(inds, scores)
    prof['gate'] += time.time() - t0

    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp

    # Shared experts
    t0 = time.time()
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)
    prof['shared'] += time.time() - t0

    # Index extraction
    t0 = time.time()
    inds_list = inds.reshape(B * S, topk).tolist()
    scores_flat = scores.reshape(B * S, topk)
    x_flat = x.reshape(B * S, H)
    prof['index_extract'] += time.time() - t0

    with mx.stream(mx.cpu):
        for t in range(B * S):
            x_t = x_flat[t:t+1]
            expert_results = []

            for k_i in range(topk):
                eidx = inds_list[t][k_i]
                score = scores_flat[t, k_i]

                # Weight slicing (triggers mmap page faults)
                t0 = time.time()
                gw, gs = mlp.gate_proj["weight"][eidx], mlp.gate_proj["scales"][eidx]
                gb = mlp.gate_proj.get("biases")
                gb = gb[eidx] if gb is not None else None
                uw, us = mlp.up_proj["weight"][eidx], mlp.up_proj["scales"][eidx]
                ub = mlp.up_proj.get("biases")
                ub = ub[eidx] if ub is not None else None
                dw, ds = mlp.down_proj["weight"][eidx], mlp.down_proj["scales"][eidx]
                db = mlp.down_proj.get("biases")
                db = db[eidx] if db is not None else None
                prof['weight_slice'] += time.time() - t0

                # Quantized matmul (graph construction — no actual compute yet)
                t0 = time.time()
                gs_val, bits = mlp.gate_proj.group_size, mlp.gate_proj.bits
                g = mx.quantized_matmul(x_t, gw, gs, gb, transpose=True, group_size=gs_val, bits=bits)
                u = mx.quantized_matmul(x_t, uw, us, ub, transpose=True, group_size=gs_val, bits=bits)
                out = mx.quantized_matmul(nn.silu(g) * u, dw, ds, db, transpose=True, group_size=gs_val, bits=bits)
                expert_results.append(out * score)
                prof['graph_build'] += time.time() - t0

            # Sum experts
            t0 = time.time()
            combined = expert_results[0]
            for er in expert_results[1:]:
                combined = combined + er
            prof['sum'] += time.time() - t0

        # Eval (actual compute + I/O)
        t0 = time.time()
        routed = combined.reshape(B, S, H)  # Single token, so no concat needed
        result = (routed + shared) if shared is not None else routed
        mx.eval(result)
        prof['eval'] += time.time() - t0

    return result


def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v20 — MoE Micro-Profile")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(4 * 1024**3))
    try:
        mx.set_wired_limit(int(80 * 1024**3))
    except:
        pass

    import mlx_lm
    print("  Loading (lazy)...", flush=True)
    model, tokenizer = mlx_lm.load(args.model, lazy=True)

    from mlx_lm.models.cache import KVCache
    input_ids = tokenizer.encode(args.prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]
    moe_indices = [i for i, l in enumerate(model.model.layers) if is_moe(l)]

    pin_attention_weights(model)

    # Run prefill first (token 1)
    print("\n  Running prefill...", flush=True)
    ids = mx.array([input_ids])
    x = model.model.embed_tokens(ids)
    mx.eval(x)
    mask = create_mask(ids.shape[1], 0)
    for i, layer in enumerate(model.model.layers):
        h = layer.input_layernorm(x)
        h = layer.self_attn(h, mask, kv_caches[i])
        x = x + h
        mx.eval(x)
        h = layer.post_attention_layernorm(x)
        moe_mod = get_moe(layer)
        if i in moe_indices:
            with mx.stream(mx.cpu):
                h = moe_mod(x)  # Use native for prefill
                mx.eval(h)
        else:
            h = moe_mod(h)
            mx.eval(h)
        x = x + h
        mx.eval(x)
        if (i+1) % 10 == 0:
            print(f"[L{i+1}]", end=" ", flush=True)

    x_norm = model.model.norm(x[:, -1:, :])
    logits = model.lm_head(x_norm)
    mx.eval(logits)
    token1 = int(mx.argmax(logits[0, 0]).item())
    print(f"\n  Prefill done: {tokenizer.decode([token1])!r}")

    # Now profile decode tokens 2-4
    for decode_step in range(3):
        print(f"\n  --- Decode token {decode_step + 2} ---")
        prof = {
            'gate': 0, 'shared': 0, 'index_extract': 0,
            'weight_slice': 0, 'graph_build': 0, 'sum': 0, 'eval': 0,
        }

        ids = mx.array([[token1]])
        x = model.model.embed_tokens(ids)
        mx.eval(x)
        mask = create_mask(1, kv_caches[0].offset)

        t_total = time.time()
        attn_total = 0

        for i, layer in enumerate(model.model.layers):
            t_a = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)
            attn_total += time.time() - t_a

            h = layer.post_attention_layernorm(x)
            moe_mod = get_moe(layer)
            if i in moe_indices:
                h = profiled_moe_forward(moe_mod, h, prof)
            else:
                h = moe_mod(h)
                mx.eval(h)
            x = x + h
            mx.eval(x)

        x_norm = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x_norm)
        mx.eval(logits)
        token1 = int(mx.argmax(logits[0, 0]).item())

        dt = time.time() - t_total
        moe_total = sum(prof.values())

        print(f"  Token: {tokenizer.decode([token1])!r} | {dt:.1f}s total")
        print(f"  Attention: {attn_total:.2f}s ({attn_total/dt*100:.0f}%)")
        print(f"  MoE total: {moe_total:.2f}s ({moe_total/dt*100:.0f}%)")
        print(f"  MoE breakdown (across {len(moe_indices)} layers):")
        for k, v in sorted(prof.items(), key=lambda x: -x[1]):
            per_layer = v / len(moe_indices) * 1000
            pct = v / moe_total * 100 if moe_total > 0 else 0
            print(f"    {k:<15} {v:.3f}s ({pct:5.1f}%) = {per_layer:.1f}ms/layer")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
