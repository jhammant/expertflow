#!/usr/bin/env python3
"""Profile MoE forward pass: native vs streaming vs GPU-cached."""
import os, time
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

def free_gb():
    import subprocess
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
    return layer.mlp


def is_moe(layer):
    m = get_moe(layer)
    return m is not None and hasattr(m, 'gate') and hasattr(m, 'switch_mlp')


def profile_native_forward(model, tokenizer, prompt, n_tokens=20):
    """Use mlx_lm native generate — lets MLX handle everything."""
    import mlx_lm
    print("\n=== Profile: NATIVE (mlx_lm.generate) ===")
    input_ids = tokenizer.encode(prompt)
    t0 = time.time()
    result = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=n_tokens, verbose=False)
    dt = time.time() - t0
    tok_s = n_tokens / dt
    print(f"  {n_tokens} tokens in {dt:.1f}s = {tok_s:.2f} tok/s")
    print(f"  Output: {result[:100]}")
    return tok_s


def profile_streaming_cpu(model, tokenizer, prompt, n_tokens=20):
    """Current streaming path: quantized_matmul on CPU per expert."""
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.base import create_causal_mask

    print("\n=== Profile: STREAMING CPU (quantized_matmul per expert) ===")
    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    generated = []
    times_detail = []

    for step in range(n_tokens):
        ids = mx.array([input_ids]) if step == 0 else mx.array([[generated[-1]]])
        x = model.model.embed_tokens(ids)
        mx.eval(x)

        seq_len = ids.shape[1]
        offset = kv_caches[0].offset if not kv_caches[0].empty() else 0
        mask = create_causal_mask(seq_len, offset) if seq_len > 1 else None

        t_attn = 0
        t_moe = 0
        t_gate = 0
        t_expert = 0

        for i, layer in enumerate(model.model.layers):
            ta = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)
            t_attn += time.time() - ta

            tm = time.time()
            h = layer.post_attention_layernorm(x)

            moe = get_moe(layer)
            if is_moe(layer):
                B, S, H = h.shape
                # Time gate
                tg = time.time()
                gates_out = moe.gate(h)
                if isinstance(gates_out, tuple):
                    inds, scores = gates_out
                else:
                    k = moe.num_experts_per_tok
                    inds = mx.stop_gradient(mx.argpartition(-gates_out, kth=k-1, axis=-1)[..., :k])
                    scores = mx.take_along_axis(gates_out, inds, axis=-1)
                    scores = mx.softmax(scores, axis=-1, precise=True)
                mx.eval(inds, scores)
                t_gate += time.time() - tg

                # Time expert compute
                te = time.time()
                topk = inds.shape[-1]
                mlp = moe.switch_mlp
                gs_val = mlp.gate_proj.group_size
                bits = mlp.gate_proj.bits

                shared = None
                if hasattr(moe, 'shared_experts') and moe.shared_experts is not None:
                    shared = moe.shared_experts(h)
                    mx.eval(shared)

                inds_list = inds.reshape(B*S, topk).tolist()
                scores_flat = scores.reshape(B*S, topk)
                x_flat = h.reshape(B*S, H)

                with mx.stream(mx.cpu):
                    token_outs = []
                    for t in range(B*S):
                        x_t = x_flat[t:t+1]
                        expert_results = []
                        for k_i in range(topk):
                            eidx = inds_list[t][k_i]
                            score = scores_flat[t, k_i]
                            gw, gs2 = mlp.gate_proj["weight"][eidx], mlp.gate_proj["scales"][eidx]
                            gb = mlp.gate_proj.get("biases")
                            gb = gb[eidx] if gb is not None else None
                            uw, us = mlp.up_proj["weight"][eidx], mlp.up_proj["scales"][eidx]
                            ub = mlp.up_proj.get("biases")
                            ub = ub[eidx] if ub is not None else None
                            dw, ds = mlp.down_proj["weight"][eidx], mlp.down_proj["scales"][eidx]
                            db = mlp.down_proj.get("biases")
                            db = db[eidx] if db is not None else None
                            g = mx.quantized_matmul(x_t, gw, gs2, gb, transpose=True, group_size=gs_val, bits=bits)
                            u = mx.quantized_matmul(x_t, uw, us, ub, transpose=True, group_size=gs_val, bits=bits)
                            out = mx.quantized_matmul(nn.silu(g)*u, dw, ds, db, transpose=True, group_size=gs_val, bits=bits)
                            expert_results.append(out * score)
                        combined = expert_results[0]
                        for er in expert_results[1:]:
                            combined = combined + er
                        token_outs.append(combined)
                    routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
                    result = (routed + shared) if shared is not None else routed
                    mx.eval(result)
                h = result
                t_expert += time.time() - te
            else:
                h = moe(h)
                mx.eval(h)

            x = x + h
            mx.eval(x)
            t_moe += time.time() - tm

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)
        next_id = int(mx.argmax(logits[0, 0]).item())
        generated.append(next_id)

        if step > 0:  # skip prefill
            times_detail.append({
                'attn': t_attn, 'moe_total': t_moe,
                'gate': t_gate, 'expert': t_expert,
            })

    if times_detail:
        avg = {k: sum(d[k] for d in times_detail)/len(times_detail)
               for k in times_detail[0]}
        total = avg['attn'] + avg['moe_total']
        print(f"  Avg decode: {total:.3f}s/tok = {1/total:.2f} tok/s")
        print(f"    attn:    {avg['attn']*1000:.1f}ms ({avg['attn']/total*100:.0f}%)")
        print(f"    moe:     {avg['moe_total']*1000:.1f}ms ({avg['moe_total']/total*100:.0f}%)")
        print(f"      gate:  {avg['gate']*1000:.1f}ms")
        print(f"      expert:{avg['expert']*1000:.1f}ms")
    return 1/total if times_detail else 0


def profile_native_moe(model, tokenizer, prompt, n_tokens=20):
    """Use native MoE forward on GPU (no streaming, no CPU)."""
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.base import create_causal_mask

    print("\n=== Profile: NATIVE MOE (GPU, model's own forward) ===")
    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    generated = []
    decode_times = []

    for step in range(n_tokens):
        t0 = time.time()
        ids = mx.array([input_ids]) if step == 0 else mx.array([[generated[-1]]])
        x = model.model.embed_tokens(ids)
        mx.eval(x)

        seq_len = ids.shape[1]
        offset = kv_caches[0].offset if not kv_caches[0].empty() else 0
        mask = create_causal_mask(seq_len, offset) if seq_len > 1 else None

        for i, layer in enumerate(model.model.layers):
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)

            h = layer.post_attention_layernorm(x)
            moe = get_moe(layer)
            h = moe(h)
            mx.eval(h)
            x = x + h
            mx.eval(x)

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)
        next_id = int(mx.argmax(logits[0, 0]).item())
        generated.append(next_id)

        if step > 0:
            decode_times.append(time.time() - t0)

    if decode_times:
        avg = sum(decode_times) / len(decode_times)
        print(f"  Avg decode: {avg:.3f}s/tok = {1/avg:.2f} tok/s")
    return 1/avg if decode_times else 0


def main():
    import mlx_lm
    model_path = os.path.expanduser("~/models/mixtral-8x7b-4bit")

    print("=" * 60)
    print("  MoE Forward Pass Profiler")
    print("=" * 60)
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Free:  {free_gb():.1f} GB")

    mx.set_memory_limit(int(80 * 1024**3))
    mx.set_cache_limit(int(4 * 1024**3))

    model, tokenizer = mlx_lm.load(model_path, lazy=True)
    prompt = "Explain the history of Paris, France"

    # Test 1: Native mlx_lm.generate (gold standard)
    native_tps = profile_native_forward(model, tokenizer, prompt, n_tokens=20)

    # Test 2: Native MoE forward on GPU (our layer loop, native MoE)
    gpu_tps = profile_native_moe(model, tokenizer, prompt, n_tokens=20)

    # Test 3: Streaming CPU (current ExpertFlow path)
    cpu_tps = profile_streaming_cpu(model, tokenizer, prompt, n_tokens=20)

    print(f"\n{'='*60}")
    print(f"  SUMMARY:")
    print(f"    mlx_lm.generate (native):  {native_tps:.2f} tok/s")
    print(f"    Native MoE (GPU loop):     {gpu_tps:.2f} tok/s")
    print(f"    Streaming CPU (ExpertFlow): {cpu_tps:.2f} tok/s")
    print(f"    Overhead: {native_tps/max(cpu_tps,0.01):.1f}x vs streaming")
    print("=" * 60)


if __name__ == "__main__":
    main()
