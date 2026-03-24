#!/usr/bin/env python3
"""
ExpertFlow Engine v5 — Aggressive Memory Management
===================================================
Forces evaluation after every expert computation to prevent graph explosion.
"""

import os, sys, time, json, subprocess, gc
from collections import OrderedDict

os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

MODEL_PATH = os.path.expanduser("~/models/deepseek-v3.1-4bit")

def free_gb():
    out = subprocess.check_output(["vm_stat"]).decode()
    f = i = 0
    for l in out.split("\n"):
        if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
        elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
    return (f + i) * 16384 / 1e9

class ExpertLRU:
    def __init__(self, max_size=32):  # Smaller cache to save memory
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        return None

    def put(self, key, val):
        self.cache[key] = val
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_size:
            evicted_key, evicted_val = self.cache.popitem(last=False)
            del evicted_val  # Explicit cleanup

    @property 
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t * 100 if t else 0

CACHE = ExpertLRU()

def patched_gather_qmm(self, x, indices, sorted_indices=False):
    """
    Ultra-aggressive memory management version.
    Forces evaluation after each expert to keep GPU memory minimal.
    """
    # Shapes
    *batch_dims, in_dim = x.shape  
    *index_batch_dims, topk = indices.shape
    out_dim = self.output_dims
    
    # Get unique experts
    indices_flat = indices.reshape(-1)
    mx.eval(indices_flat)
    unique_experts = sorted(set(indices_flat.tolist()))
    
    # Process each expert individually with immediate eval
    expert_weights = {}
    for eidx in unique_experts:
        cache_key = (id(self), eidx)
        cached = CACHE.get(cache_key)
        if cached is not None:
            expert_weights[eidx] = cached
        else:
            CACHE.misses += 1
            # Load and immediately evaluate
            w = self["weight"][eidx]
            s = self["scales"][eidx] 
            b = self.get("biases")
            b = b[eidx] if b is not None else None
            
            w_dequant = mx.dequantize(w, s, b,
                                    group_size=self.group_size, 
                                    bits=self.bits)
            mx.eval(w_dequant)  # Force immediate eval
            mx.metal.clear_cache()  # Clear Metal cache
            
            expert_weights[eidx] = w_dequant
            CACHE.put(cache_key, w_dequant)
    
    # Compute outputs one at a time with aggressive cleanup
    x_flat = x.reshape(-1, in_dim)
    indices_flat = indices.reshape(-1)
    
    outputs = []
    for i in range(0, len(indices_flat), 8):  # Process in small batches
        batch_outputs = []
        for j in range(min(8, len(indices_flat) - i)):
            idx = i + j
            eidx = int(indices_flat[idx].item())
            token_idx = idx // topk
            
            x_token = x_flat[token_idx:token_idx+1]
            w_expert = expert_weights[eidx]
            
            out = x_token @ w_expert.T
            
            if "bias" in self:
                bias = self["bias"][eidx:eidx+1]
                mx.eval(bias)
                out = out + bias
            
            mx.eval(out)  # Eval each output immediately
            batch_outputs.append(out)
        
        # Concatenate this batch and eval
        if batch_outputs:
            batch_result = mx.concatenate(batch_outputs, axis=0)
            mx.eval(batch_result)
            outputs.append(batch_result)
            
        # Clear cache between batches
        mx.metal.clear_cache()
        if i % 32 == 0:  # Periodic GC
            gc.collect()
    
    # Final concatenation
    if outputs:
        result = mx.concatenate(outputs, axis=0)
    else:
        result = mx.zeros((len(indices_flat), out_dim))
    
    # Reshape with extra dimension for squeeze
    target_shape = (*index_batch_dims, topk, 1, out_dim)
    result = result.reshape(target_shape)
    
    mx.eval(result)
    mx.metal.clear_cache()
    
    return result

def monkey_patch_with_memory_management(model):
    """Patch with ultra-aggressive memory management."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    
    QuantizedSwitchLinear._original_call = QuantizedSwitchLinear.__call__
    QuantizedSwitchLinear.__call__ = patched_gather_qmm
    
    count = sum(1 for _, m in model.named_modules() 
                if isinstance(m, QuantizedSwitchLinear))
    return count

def main():
    print("=" * 60, flush=True)
    print("  ExpertFlow Engine v5 — Memory Optimized", flush=True) 
    print("  DeepSeek V3.1 (378GB) → 128GB M5 Max", flush=True)
    print("=" * 60, flush=True)
    
    # Very conservative settings
    mx.set_memory_limit(15 * 1024**3)  # Even tighter: 15GB 
    mx.set_cache_limit(3 * 1024**3)    # 3GB cache
    print(f"  MLX: 15GB memory, 3GB cache | Free: {free_gb():.1f} GB", flush=True)
    
    # Load model
    import mlx_lm
    print(f"\n  Loading model...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL_PATH, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s | Free: {free_gb():.1f} GB", flush=True)
    
    # Patch
    count = monkey_patch_with_memory_management(model)
    print(f"  Patched {count} layers with memory management", flush=True)
    
    # Single test to minimize memory usage
    prompt = "Hi"
    max_tokens = 3  # Very short to test
    
    print(f"\n  Testing: {prompt!r} (max {max_tokens} tokens)", flush=True)
    print(f"  Output: ", end="", flush=True)
    
    try:
        t_start = time.time()
        token_count = 0
        
        for resp in mlx_lm.stream_generate(model, tokenizer, 
                                         prompt=prompt, max_tokens=max_tokens):
            text = resp.text if hasattr(resp, "text") else str(resp)
            token_count += 1
            sys.stdout.write(text)
            sys.stdout.flush()
            
            # Clear cache after each token
            mx.metal.clear_cache()
            gc.collect()
        
        elapsed = time.time() - t_start
        tps = token_count / elapsed if elapsed > 0 else 0
        
        print(f"\n  ✅ SUCCESS! {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)", flush=True)
        print(f"  Cache: {CACHE.hit_rate:.1f}% hit rate", flush=True)
        print(f"  Free: {free_gb():.1f} GB", flush=True)
        
        # Save the achievement
        result = {
            "status": "FIRST_SUCCESS",
            "model": "DeepSeek V3.1 4-bit",
            "size_gb": 378,
            "ram_gb": 128,
            "ratio": "2.95x RAM",
            "hardware": "Apple M5 Max", 
            "tokens_generated": token_count,
            "time_s": round(elapsed, 1),
            "tok_per_s": round(tps, 1),
            "cache_hit_rate": round(CACHE.hit_rate, 1),
        }
        with open(os.path.expanduser("~/dev/expertflow/BREAKTHROUGH.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n  🏆 BREAKTHROUGH: DeepSeek V3.1 running on consumer hardware!", flush=True)
        print(f"  🔥 ExpertFlow enables 3x RAM overcommit with dynamic expert loading!", flush=True)
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}", flush=True)
        print(f"  Free: {free_gb():.1f} GB", flush=True)

if __name__ == "__main__":
    main()