#!/usr/bin/env python3
"""
ExpertFlow v19 — CPU gather_qmm (Batched Expert Dispatch)
==========================================================
v17 steady-state: 10.7s/tok = 120ms/MoE layer = 2136 individual qmm calls.
Per-expert Python loop: 8 experts × 3 projections × 89 layers = massive overhead.

Hypothesis: Use SwitchGLU's native gather_qmm on CPU stream.
  - gather_qmm batches all 8 expert matmuls into ONE kernel call
  - CPU stream avoids Metal page-fault overhead
  - Eliminates Python loop over experts (8 iterations → 1 call)
  - The full weight tensor IS referenced, but CPU mmap should handle it

If this works, we go from 2136 Python-level calls to 267 (89 layers × 3 projections).
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


def pin_attention_weights(model, verbose=True):
    if verbose:
        print("  Pinning attention weights...", end=" ", flush=True)
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
    if verbose:
        print(f"done ({time.time()-t0:.1f}s, {free_gb():.0f}G free)")


def cpu_native_moe_forward(moe_module, x):
    """
    Run the NATIVE MoE forward (SwitchGLU with gather_qmm) on CPU stream.
    gather_qmm batches all expert matmuls into one kernel — no Python loop.
    CPU stream avoids Metal command buffer overhead for mmap page faults.
    """
    with mx.stream(mx.cpu):
        result = moe_module(x)
        mx.eval(result)
    return result


def per_expert_moe_forward(moe_module, x):
    """Original per-expert approach for comparison."""
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
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    inds_list = inds.reshape(B * S, topk).tolist()
    scores_flat = scores.reshape(B * S, topk)
    x_flat = x.reshape(B * S, H)

    with mx.stream(mx.cpu):
        token_outs = []
        for t in range(B * S):
            x_t = x_flat[t:t+1]
            expert_results = []
            for k_i in range(topk):
                eidx = inds_list[t][k_i]
                score = scores_flat[t, k_i]
                gw, gs = mlp.gate_proj["weight"][eidx], mlp.gate_proj["scales"][eidx]
                gb = mlp.gate_proj.get("biases")
                gb = gb[eidx] if gb is not None else None
                uw, us = mlp.up_proj["weight"][eidx], mlp.up_proj["scales"][eidx]
                ub = mlp.up_proj.get("biases")
                ub = ub[eidx] if ub is not None else None
                dw, ds = mlp.down_proj["weight"][eidx], mlp.down_proj["scales"][eidx]
                db = mlp.down_proj.get("biases")
                db = db[eidx] if db is not None else None
                gs_val, bits = mlp.gate_proj.group_size, mlp.gate_proj.bits
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


def generate(model, tokenizer, prompt, max_tokens, use_cpu_gather=True, verbose=True):
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]
    moe_indices = [i for i, l in enumerate(model.model.layers) if is_moe(l)]

    pin_attention_weights(model, verbose)

    approach = "cpu_gather_qmm" if use_cpu_gather else "per_expert_qmm"

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_indices)} MoE)")
        print(f"  Approach: {approach}")

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
            moe_mod = get_moe(layer)

            if i in moe_indices:
                if use_cpu_gather:
                    h = cpu_native_moe_forward(moe_mod, h)
                else:
                    h = per_expert_moe_forward(moe_mod, h)
            else:
                h = moe_mod(h)
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
        "approach": approach,
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
    p.add_argument("--max-tokens", type=int, default=7)
    p.add_argument("--per-expert", action="store_true", help="Use per-expert approach (baseline)")
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v19 — CPU gather_qmm vs per-expert")
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

    results = generate(model, tokenizer, args.prompt, args.max_tokens,
                      use_cpu_gather=not args.per_expert)

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {results['prompt']}{results['output']}")
    print(f"  APPROACH: {results['approach']}")
    print(f"  PREFILL: {results['prefill_s']}s")
    print(f"  DECODE:  {results['decode_avg_s']}s/tok ({results['decode_tok_s']} tok/s)")
    print(f"  Total: {results['total_s']}s for {results['tokens']} tokens")

    outfile = os.path.expanduser(f"~/dev/expertflow/v19_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
