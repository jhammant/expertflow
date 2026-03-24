#!/usr/bin/env python3
"""
ExpertFlow v17 — Pinned Attention + File-Based Prefetch
========================================================
v14 steady-state: 11.4s/tok (attn=0.7s, moe=10.7s)
Token 2 cold-start: 39.6s (attn=10.9s! — attention weights evicted)

Optimization 1: Pin attention weights in RAM
  - Pre-evaluate all attention layer weights before generation
  - Forces them into page cache; frequent access keeps them there
  - Should eliminate the 10.9s cold-start on token 2

Optimization 2: File-based prefetch for expert weights
  - After computing gate (know which experts needed), predict next layer's
  - Use Python threads to read raw file bytes (warms OS page cache)
  - Threads do plain file I/O (no MLX) — GIL released during reads
  - When MLX later accesses the same mmap regions, pages already in RAM
"""

import os, sys, time, json, subprocess, traceback
import concurrent.futures
import struct
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


# ═══ Optimization 1: Pin Attention Weights ═══

def pin_attention_weights(model, verbose=True):
    """
    Pre-evaluate all attention layer weights to force them into RAM.
    This prevents the 10s cold-start on token 2 caused by attention
    weight eviction during prefill's MoE processing.
    """
    if verbose:
        print("  Pinning attention weights...", end=" ", flush=True)
    t0 = time.time()

    for i, layer in enumerate(model.model.layers):
        # Touch attention weights
        attn = layer.self_attn
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn, name):
                proj = getattr(attn, name)
                if hasattr(proj, 'weight'):
                    mx.eval(proj.weight.sum())

        # Touch layernorm weights
        if hasattr(layer, 'input_layernorm'):
            mx.eval(layer.input_layernorm.weight.sum())
        if hasattr(layer, 'post_attention_layernorm'):
            mx.eval(layer.post_attention_layernorm.weight.sum())

    # Also pin embedding and output head
    mx.eval(model.model.embed_tokens.weight.sum())
    if hasattr(model, 'lm_head'):
        if hasattr(model.lm_head, 'weight'):
            mx.eval(model.lm_head.weight.sum())
    if hasattr(model.model, 'norm'):
        mx.eval(model.model.norm.weight.sum())

    dt = time.time() - t0
    if verbose:
        print(f"done ({dt:.1f}s, {free_gb():.0f}G free)", flush=True)


# ═══ Optimization 2: File-Based Expert Prefetch ═══

class ExpertPrefetcher:
    """
    Prefetches expert weight pages by reading raw bytes from safetensors files.
    Uses a thread pool — Python threads release GIL during file I/O,
    so reads happen concurrently with MLX computation.
    The OS page cache is shared, so these reads warm pages for MLX's mmap.
    """
    def __init__(self, model_path, max_workers=4):
        self.model_path = model_path
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.tensor_map = {}  # tensor_name -> (file_path, offset, length)
        self._build_tensor_map()
        self.pending = []

    def _build_tensor_map(self):
        """Parse safetensors index to find tensor locations."""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            # Single-file model
            for f in os.listdir(self.model_path):
                if f.endswith('.safetensors'):
                    self._parse_safetensors_header(os.path.join(self.model_path, f))
            return

        with open(index_path) as f:
            index = json.load(f)

        # Group tensors by file
        file_tensors = {}
        for tensor_name, filename in index.get("weight_map", {}).items():
            filepath = os.path.join(self.model_path, filename)
            if filepath not in file_tensors:
                file_tensors[filepath] = []
            file_tensors[filepath].append(tensor_name)

        # Parse each safetensors file header to get offsets
        for filepath in file_tensors:
            self._parse_safetensors_header(filepath)

    def _parse_safetensors_header(self, filepath):
        """Read safetensors header to get tensor byte offsets."""
        try:
            with open(filepath, 'rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(header_size))

            data_start = 8 + header_size
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                offsets = info.get('data_offsets', [0, 0])
                start = data_start + offsets[0]
                length = offsets[1] - offsets[0]
                self.tensor_map[name] = (filepath, start, length)
        except Exception:
            pass

    def prefetch_experts(self, layer_idx, expert_indices, proj_names=None):
        """
        Submit prefetch jobs for specific experts at a given layer.
        Reads raw bytes from safetensors files to warm OS page cache.
        """
        if proj_names is None:
            proj_names = ['gate_proj', 'up_proj', 'down_proj']

        futures = []
        for eidx in expert_indices:
            for proj in proj_names:
                # Try different naming conventions
                for pattern in [
                    f"model.layers.{layer_idx}.mlp.experts.{eidx}.{proj}.weight",
                    f"model.layers.{layer_idx}.block_sparse_moe.experts.{eidx}.{proj}.weight",
                    f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}.weight",
                ]:
                    if pattern in self.tensor_map:
                        filepath, offset, length = self.tensor_map[pattern]
                        futures.append(
                            self.executor.submit(self._read_bytes, filepath, offset, length)
                        )
                        break
        self.pending = futures

    def _read_bytes(self, filepath, offset, length):
        """Read bytes from file to warm page cache. Data is discarded."""
        try:
            with open(filepath, 'rb') as f:
                f.seek(offset)
                f.read(min(length, 4 * 1024 * 1024))  # Read up to 4MB
        except:
            pass

    def wait(self):
        """Wait for pending prefetch jobs to complete."""
        if self.pending:
            concurrent.futures.wait(self.pending, timeout=0.5)
            self.pending = []

    def shutdown(self):
        self.executor.shutdown(wait=False)


# ═══ Streaming MoE ═══

def streaming_moe_forward(moe_module, x, is_mixtral=False):
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

    inds_flat = inds.reshape(B * S, topk)
    scores_flat = scores.reshape(B * S, topk)
    inds_list = inds_flat.tolist()
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

    # Return expert indices for prefetch prediction
    return result, inds_list


def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def generate(model, tokenizer, prompt, max_tokens, model_path=None,
             pin_attn=True, prefetch=True, verbose=True):
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    is_mixtral_arch = hasattr(model.model.layers[0], 'block_sparse_moe')

    # Build MoE layer index
    moe_indices = []
    for i, layer in enumerate(model.model.layers):
        if is_moe(layer):
            moe_indices.append(i)

    # Pin attention weights
    if pin_attn:
        pin_attention_weights(model, verbose)

    # Setup prefetcher
    prefetcher = None
    if prefetch and model_path:
        prefetcher = ExpertPrefetcher(model_path, max_workers=4)
        if verbose:
            print(f"  Prefetcher: {len(prefetcher.tensor_map)} tensors mapped")

    generated_ids = []
    token_times = []

    if verbose:
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_indices)} MoE)")

    # Track expert usage for prefetch prediction
    prev_experts_by_layer = {}

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

            # MoE
            t_m = time.time()
            h = layer.post_attention_layernorm(x)

            if i in moe_indices:
                # Wait for any pending prefetch for THIS layer
                if prefetcher:
                    prefetcher.wait()

                h, expert_inds = streaming_moe_forward(get_moe(layer), h, is_mixtral_arch)

                # Track experts used and prefetch for next MoE layer
                current_experts = set()
                for token_inds in expert_inds:
                    current_experts.update(token_inds)
                prev_experts_by_layer[i] = current_experts

                # Prefetch: submit read jobs for the NEXT MoE layer
                # Use current layer's experts as prediction (hot experts repeat)
                if prefetcher:
                    next_moe_idx = None
                    for mi in moe_indices:
                        if mi > i:
                            next_moe_idx = mi
                            break
                    if next_moe_idx is not None:
                        # Predict: use same experts as current layer
                        prefetcher.prefetch_experts(next_moe_idx, current_experts)
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
        "pin_attn": pin_attn,
        "prefetch": prefetch,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=5)
    p.add_argument("--no-pin", action="store_true", help="Disable attention pinning")
    p.add_argument("--no-prefetch", action="store_true", help="Disable expert prefetch")
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v17 — Pinned Attn + Expert Prefetch")
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
        results = generate(
            model, tokenizer, args.prompt, args.max_tokens,
            model_path=args.model,
            pin_attn=not args.no_pin,
            prefetch=not args.no_prefetch,
        )
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
        f"~/dev/expertflow/v17_{time.strftime('%H%M%S')}.json"
    )
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
