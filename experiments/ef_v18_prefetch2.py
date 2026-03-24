#!/usr/bin/env python3
"""
ExpertFlow v18 — Stacked Tensor Prefetch + Pinned Attention
============================================================
v17: pinned attn → token 2 attn: 10.9→1.7s, steady-state 10.7s/tok

Expert weights are stacked: [160, out_dim, packed_in] per projection.
Each expert slice: ~3.8MB. 3 projections × 8 experts = ~91MB per layer.
At 7.4 GB/s NVMe, prefetching = ~12ms — easily fits within current layer's compute.

v18: File-based prefetch using calculated byte offsets into stacked tensors.
Uses Python threads for I/O (GIL released during reads).
"""

import os, sys, time, json, subprocess, traceback, struct
import concurrent.futures
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


# ═══ Pin Attention Weights ═══

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
        if hasattr(layer, 'input_layernorm'):
            mx.eval(layer.input_layernorm.weight.sum())
        if hasattr(layer, 'post_attention_layernorm'):
            mx.eval(layer.post_attention_layernorm.weight.sum())
    mx.eval(model.model.embed_tokens.weight.sum())
    if hasattr(model.lm_head, 'weight'):
        mx.eval(model.lm_head.weight.sum())
    if hasattr(model.model, 'norm'):
        mx.eval(model.model.norm.weight.sum())
    if verbose:
        print(f"done ({time.time()-t0:.1f}s, {free_gb():.0f}G free)")


# ═══ Stacked Tensor Prefetcher ═══

class StackedExpertPrefetcher:
    """
    Prefetch specific expert slices from stacked weight tensors.
    Reads raw bytes at calculated offsets to warm the OS page cache.
    """

    def __init__(self, model_path, max_workers=4):
        self.model_path = model_path
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending = []

        # Build lookup: (layer_idx, proj_name) -> (filepath, data_start, shape, n_experts, per_expert_bytes)
        self.tensor_info = {}
        self._build_tensor_info()

    def _build_tensor_info(self):
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            return

        with open(index_path) as f:
            idx = json.load(f)

        # Group by file
        files_to_parse = set()
        tensor_to_file = {}
        for tensor_name, filename in idx.get("weight_map", {}).items():
            if 'switch_mlp' in tensor_name and '.weight' in tensor_name:
                filepath = os.path.join(self.model_path, filename)
                tensor_to_file[tensor_name] = filepath
                files_to_parse.add(filepath)

        # Parse safetensors headers
        file_headers = {}
        for filepath in files_to_parse:
            try:
                with open(filepath, 'rb') as f:
                    header_size = struct.unpack('<Q', f.read(8))[0]
                    header = json.loads(f.read(header_size))
                    data_start = 8 + header_size
                file_headers[filepath] = (header, data_start)
            except:
                pass

        # Build per-layer per-projection lookup
        for tensor_name, filepath in tensor_to_file.items():
            if filepath not in file_headers:
                continue
            header, data_start = file_headers[filepath]
            if tensor_name not in header:
                continue

            info = header[tensor_name]
            shape = info['shape']
            offsets = info['data_offsets']
            abs_start = data_start + offsets[0]
            total_bytes = offsets[1] - offsets[0]
            n_experts = shape[0]
            per_expert = total_bytes // n_experts

            # Parse layer index and projection name
            # e.g. "model.layers.3.mlp.switch_mlp.gate_proj.weight"
            parts = tensor_name.split('.')
            try:
                layer_idx = int(parts[2])
                proj_name = parts[5]  # gate_proj, up_proj, down_proj
            except (IndexError, ValueError):
                continue

            self.tensor_info[(layer_idx, proj_name)] = (
                filepath, abs_start, n_experts, per_expert
            )

    def prefetch_experts(self, layer_idx, expert_indices):
        """Submit async reads for specific expert slices."""
        futures = []
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            key = (layer_idx, proj)
            if key not in self.tensor_info:
                continue
            filepath, abs_start, n_experts, per_expert = self.tensor_info[key]

            for eidx in expert_indices:
                if eidx >= n_experts:
                    continue
                offset = abs_start + eidx * per_expert
                futures.append(
                    self.executor.submit(self._read_range, filepath, offset, per_expert)
                )

        # Also prefetch scales (smaller, but needed)
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            scales_key_name = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}.scales"
            # Scales are much smaller — prefetching the whole thing is fine

        self.pending = futures

    def _read_range(self, filepath, offset, length):
        """Read bytes to warm page cache."""
        try:
            with open(filepath, 'rb') as f:
                f.seek(offset)
                # Read in 64KB chunks (page-aligned)
                remaining = length
                while remaining > 0:
                    chunk = min(remaining, 65536)
                    f.read(chunk)
                    remaining -= chunk
        except:
            pass

    def wait(self):
        if self.pending:
            concurrent.futures.wait(self.pending, timeout=1.0)
            self.pending = []

    def shutdown(self):
        self.executor.shutdown(wait=False)

    @property
    def available(self):
        return len(self.tensor_info) > 0


# ═══ Streaming MoE ═══

def streaming_moe_forward(moe_module, x):
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

                gs_val = mlp.gate_proj.group_size
                bits = mlp.gate_proj.bits

                g = mx.quantized_matmul(x_t, gw, gs, gb, transpose=True,
                                         group_size=gs_val, bits=bits)
                u = mx.quantized_matmul(x_t, uw, us, ub, transpose=True,
                                         group_size=gs_val, bits=bits)
                out = mx.quantized_matmul(nn.silu(g) * u, dw, ds, db, transpose=True,
                                           group_size=gs_val, bits=bits)
                expert_results.append(out * score)

            combined = expert_results[0]
            for er in expert_results[1:]:
                combined = combined + er
            token_outs.append(combined)

        routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
        result = (routed + shared) if shared is not None else routed
        mx.eval(result)

    return result, inds_list


def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def generate(model, tokenizer, prompt, max_tokens, model_path=None, verbose=True):
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    moe_indices = [i for i, l in enumerate(model.model.layers) if is_moe(l)]

    # Optimizations
    pin_attention_weights(model, verbose)

    prefetcher = StackedExpertPrefetcher(model_path) if model_path else None
    if verbose and prefetcher:
        print(f"  Prefetcher: {len(prefetcher.tensor_info)} projection tensors mapped "
              f"({'active' if prefetcher.available else 'inactive'})")

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_indices)} MoE)")

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

            if i in moe_indices:
                # Wait for prefetch of THIS layer
                if prefetcher and prefetcher.available:
                    prefetcher.wait()

                h, expert_inds = streaming_moe_forward(get_moe(layer), h)

                # Start prefetch for NEXT MoE layer using current experts
                if prefetcher and prefetcher.available:
                    current_experts = set()
                    for t_inds in expert_inds:
                        current_experts.update(t_inds)

                    # Find next MoE layer
                    for mi in moe_indices:
                        if mi > i:
                            prefetcher.prefetch_experts(mi, current_experts)
                            break
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

    if prefetcher:
        prefetcher.shutdown()

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
    p.add_argument("--max-tokens", type=int, default=7)
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v18 — Pinned Attn + Stacked Prefetch")
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
        results = generate(model, tokenizer, args.prompt, args.max_tokens,
                          model_path=args.model)
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
        f"~/dev/expertflow/v18_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
