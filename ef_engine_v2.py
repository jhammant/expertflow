#!/usr/bin/env python3
"""
ExpertFlow Engine v2 — Dynamic Expert Inference for DeepSeek V3.1
=================================================================
Runs a 378GB model on 128GB RAM by monkey-patching mlx-lm's
QuantizedSwitchLinear to slice expert weights before GPU eval.

The key trick: intercept gather_qmm calls and replace them with
manual slice → dequantize → matmul on only the needed expert rows.
"""

import os, sys, time, json, gc, subprocess
from collections import OrderedDict
from functools import wraps

os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

MODEL_PATH = os.path.expanduser("~/models/deepseek-v3.1-4bit")
MAX_EXPERT_CACHE = 128  # LRU entries
GROUP_SIZE = 64
BITS = 4

def free_gb():
    out = subprocess.check_output(["vm_stat"]).decode()
    f = i = 0
    for l in out.split("\n"):
        if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
        elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
    return (f + i) * 16384 / 1e9


class ExpertLRU:
    """LRU cache for dequantized expert weight slices."""
    def __init__(self, max_size=MAX_EXPERT_CACHE):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, val):
        self.cache[key] = val
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_size:
            evicted_key, evicted_val = self.cache.popitem(last=False)
            del evicted_val

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t * 100 if t > 0 else 0


CACHE = ExpertLRU()


def patched_switch_linear_call(self, x, indices, sorted_indices=False):
    """
    Replacement for QuantizedSwitchLinear.__call__ that loads only
    the needed expert slices instead of the full [256,...] tensor.
    
    Original uses mx.gather_qmm which needs full weight tensor on GPU.
    We instead:
    1. Get unique expert indices needed
    2. For each unique expert, slice weight/scales/biases (still lazy/mmap'd)
    3. Dequantize just those slices
    4. Compute matmul manually
    5. Scatter results back
    """
    # Get the unique expert indices we need
    # indices shape: [batch, topk] or flat
    idx_flat = indices.reshape(-1)
    mx.eval(idx_flat)
    unique_list = sorted(set(idx_flat.tolist()))
    
    # Get weight, scales, biases
    weight = self["weight"]       # [num_experts, out_dim, packed_in]
    scales = self["scales"]       # [num_experts, out_dim, num_groups]
    biases = self.get("biases")   # [num_experts, out_dim, num_groups] or None
    has_bias = "bias" in self      # per-expert output bias
    
    num_experts = weight.shape[0]
    out_dim = weight.shape[1]
    
    # For each unique expert, get or cache the dequantized weights
    expert_weights = {}
    for eidx in unique_list:
        eidx = int(eidx)
        cache_key = (id(self), eidx)
        cached = CACHE.get(cache_key)
        if cached is not None:
            expert_weights[eidx] = cached
        else:
            # Slice this expert's quantized weights
            w_slice = weight[eidx]      # [out_dim, packed_in]
            s_slice = scales[eidx]      # [out_dim, num_groups]
            b_slice = biases[eidx] if biases is not None else None
            
            # Dequantize
            w_deq = mx.dequantize(w_slice, s_slice, b_slice,
                                   group_size=self.group_size, bits=self.bits)
            mx.eval(w_deq)
            
            expert_weights[eidx] = w_deq
            CACHE.put(cache_key, w_deq)
    
    # Now compute: for each token, apply its assigned expert
    # x shape: [batch, 1, 1, in_dim] (after expand_dims in SwitchGLU)
    # indices shape: [batch, topk]
    
    # Flatten for processing
    orig_shape = x.shape
    if x.ndim == 4:
        # [batch, 1, 1, in_dim] -> [batch, in_dim]
        x_2d = x.reshape(-1, x.shape[-1])
    elif x.ndim == 3:
        x_2d = x.reshape(-1, x.shape[-1])
    else:
        x_2d = x
    
    idx_2d = indices.reshape(-1)  # [batch * topk]
    mx.eval(idx_2d)
    n_tokens = x_2d.shape[0]
    n_indices = idx_2d.shape[0]
    
    # Each index corresponds to a token (indices may repeat tokens for topk)
    # If more indices than tokens, tokens cycle
    outputs = []
    for i in range(n_indices):
        eidx = int(idx_2d[i].item())
        token_idx = i % n_tokens
        w = expert_weights[eidx]  # [out_dim, in_dim]
        xi = x_2d[token_idx:token_idx+1]  # [1, in_dim]
        out = xi @ w.T  # [1, out_dim]
        
        if has_bias:
            bias_slice = self["bias"][eidx:eidx+1]
            mx.eval(bias_slice)
            out = out + bias_slice
        
        outputs.append(out)
    
    result = mx.concatenate(outputs, axis=0)
    
    # Reshape to match expected output from gather_qmm
    # gather_qmm returns [batch, topk, 1, out_dim] when x has expand_dims
    result = result.reshape(*indices.shape, 1, out_dim)
    
    # Eval to free intermediate computation graph
    mx.eval(result)
    
    return result


def monkey_patch_model(model):
    """Replace all QuantizedSwitchLinear.__call__ with our dynamic version."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedSwitchLinear):
            # Bind our patched method
            import types
            module.__call__ = types.MethodType(
                lambda self, x, indices, sorted_indices=False: 
                    patched_switch_linear_call(self, x, indices, sorted_indices),
                module
            )
            count += 1
    
    return count


def monkey_patch_v2(model):
    """
    Alternative: patch at the class level so ALL instances use our version.
    """
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    
    # Save original
    QuantizedSwitchLinear._original_call = QuantizedSwitchLinear.__call__
    QuantizedSwitchLinear.__call__ = patched_switch_linear_call
    
    # Count instances
    count = sum(1 for _, m in model.named_modules() 
                if isinstance(m, QuantizedSwitchLinear))
    return count


def main():
    print("=" * 60, flush=True)
    print("  ExpertFlow Engine v2", flush=True)
    print("  DeepSeek V3.1 (378GB) on 128GB M5 Max", flush=True)
    print("=" * 60, flush=True)
    print(f"  Free memory: {free_gb():.1f} GB", flush=True)
    
    # Set memory limits
    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(5 * 1024**3)
    print(f"  MLX memory limit: 20 GB, cache: 5 GB", flush=True)
    
    # Load model lazy
    import mlx_lm
    print(f"\n  Loading model (lazy)...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL_PATH, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s | Free: {free_gb():.1f} GB", flush=True)
    
    # Monkey-patch expert layers
    print(f"\n  Patching expert layers for dynamic loading...", flush=True)
    count = monkey_patch_v2(model)
    print(f"  Patched {count} QuantizedSwitchLinear layers", flush=True)
    
    # Generate!
    prompt = "What is the capital of France?"
    print(f"\n  Prompt: {prompt!r}", flush=True)
    print(f"  Generating...", flush=True)
    print(f"  Output: ", end="", flush=True)
    
    t_start = time.time()
    token_times = []
    full_text = ""
    
    try:
        for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=20):
            t_now = time.time()
            if token_times:
                token_times.append(t_now - t_start - sum(token_times))
            else:
                token_times.append(t_now - t_start)
            
            text = resp.text if hasattr(resp, "text") else str(resp)
            full_text += text
            sys.stdout.write(text)
            sys.stdout.flush()
    except Exception as e:
        print(f"\n  Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    total = time.time() - t_start
    print(flush=True)
    
    if token_times:
        ttft = token_times[0]
        decode_times = token_times[1:]
        avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
        tps = 1.0 / avg_decode if avg_decode > 0 else 0
        
        print(f"\n  --- Results ---", flush=True)
        print(f"  TTFT: {ttft:.1f}s", flush=True)
        print(f"  Tokens: {len(token_times)}", flush=True)
        print(f"  Decode: {tps:.2f} tok/s", flush=True)
        print(f"  Total: {total:.1f}s", flush=True)
        print(f"  Cache: {CACHE.hit_rate:.1f}% ({CACHE.hits} hits, {CACHE.misses} misses)", flush=True)
        print(f"  Free: {free_gb():.1f} GB", flush=True)
        
        with open(os.path.expanduser("~/dev/expertflow/deepseek-inference.json"), "w") as f:
            json.dump({
                "model": "DeepSeek V3.1 4-bit",
                "size_gb": 378, "ram_gb": 128,
                "prompt": prompt, "response": full_text,
                "ttft_s": round(ttft, 1),
                "tokens": len(token_times),
                "decode_tok_s": round(tps, 2),
                "total_s": round(total, 1),
                "cache_hit_rate": round(CACHE.hit_rate, 1),
                "token_times": [round(t, 3) for t in token_times],
            }, f, indent=2)
        
        print(f"\n  🔥 DeepSeek V3.1 inference COMPLETE!", flush=True)
    else:
        print(f"  No tokens generated", flush=True)


if __name__ == "__main__":
    main()
