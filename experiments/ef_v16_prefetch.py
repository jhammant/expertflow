#!/usr/bin/env python3
"""
ExpertFlow v16 — Prefetched Expert Loading
===========================================
v14/v15: 11.4s/tok steady-state. moe=10.7s = 120ms/layer = 5ms/qmm.
Bottleneck: NVMe page faults per expert weight access.

v16: Prefetch next layer's expert weights while computing current layer.
  - Use a background thread to touch (fault in) the weight pages
  - By the time the compute reaches them, they're already in RAM
  - This overlaps I/O with compute, potentially halving MoE time

Also try: using madvise-like hints via accessing weight pages ahead.
"""

import os, sys, time, json, subprocess, traceback, threading
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


def prefetch_expert_weights(mlp, expert_indices):
    """
    Touch expert weight pages to fault them into RAM.
    Called from a background thread while the current layer computes.
    """
    for eidx in expert_indices:
        # Accessing the weight slice triggers mmap page-in
        # We don't need the result — just need to touch the pages
        try:
            _ = mlp.gate_proj["weight"][eidx]
            _ = mlp.up_proj["weight"][eidx]
            _ = mlp.down_proj["weight"][eidx]
            # Force evaluation to actually trigger page faults
            mx.eval(mlp.gate_proj["weight"][eidx].sum())
        except:
            pass


def streaming_moe_forward(moe_module, x, is_mixtral=False,
                          next_moe=None, next_x=None):
    """
    MoE with CPU quantized_matmul.
    Optionally prefetches next layer's weights in background.
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

    # Pre-extract indices
    inds_flat = inds.reshape(B * S, topk)
    scores_flat = scores.reshape(B * S, topk)
    inds_list = inds_flat.tolist()

    # Start prefetch for NEXT layer if available
    prefetch_thread = None
    if next_moe is not None and next_x is not None:
        # Predict next layer's experts (run gate on next layer's input)
        try:
            next_gates = next_moe.gate(next_x)
            if isinstance(next_gates, tuple):
                next_inds, _ = next_gates
            else:
                nk = next_moe.num_experts_per_tok
                next_inds = mx.argpartition(-next_gates, kth=nk - 1, axis=-1)[..., :nk]
            mx.eval(next_inds)
            next_experts = set(next_inds.reshape(-1).tolist())

            # Prefetch in background
            prefetch_thread = threading.Thread(
                target=prefetch_expert_weights,
                args=(next_moe.switch_mlp, next_experts),
                daemon=True,
            )
            prefetch_thread.start()
        except:
            pass

    # Shared experts
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    # Expert compute on CPU
    with mx.stream(mx.cpu):
        x_flat = x.reshape(B * S, H)
        token_outs = []

        for t in range(B * S):
            x_t = x_flat[t:t+1]
            expert_results = []

            for k_i in range(topk):
                eidx = inds_list[t][k_i]
                score = scores_flat[t, k_i]

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

            out = expert_results[0]
            for er in expert_results[1:]:
                out = out + er
            token_outs.append(out)

        routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
        result = (routed + shared) if shared is not None else routed
        mx.eval(result)

    # Wait for prefetch to complete
    if prefetch_thread is not None:
        prefetch_thread.join(timeout=0.1)

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

    is_mixtral_arch = hasattr(model.model.layers[0], 'block_sparse_moe')
    moe_count = sum(1 for l in model.model.layers if is_moe(l))

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({moe_count} MoE)")
        print(f"  Prefetch: enabled (1-layer lookahead)")

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

            # MoE with prefetch
            t_m = time.time()
            h = layer.post_attention_layernorm(x)

            if is_moe(layer):
                # Look ahead: find next MoE layer for prefetch
                next_moe_module = None
                next_h = None
                for j in range(i + 1, num_layers):
                    if is_moe(model.model.layers[j]):
                        next_moe_module = get_moe(model.model.layers[j])
                        # next layer's input would be current output after MoE
                        # We approximate with current post-attention output
                        next_h = h  # Approximation
                        break

                h = streaming_moe_forward(
                    get_moe(layer), h, is_mixtral_arch,
                    next_moe=next_moe_module, next_x=next_h,
                )
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
    print("  ExpertFlow v16 — Prefetched Expert Loading")
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
        f"~/dev/expertflow/v16_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
