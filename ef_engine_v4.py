#!/usr/bin/env python3
"""
ExpertFlow Engine v4 — Correct gather_qmm Behavior
===================================================
Fixed to return the exact same shape as mx.gather_qmm would.
"""

import os, sys, time, json, subprocess
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
    def __init__(self, max_size=64):
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
            self.cache.popitem(last=False)

    @property 
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t * 100 if t else 0

CACHE = ExpertLRU()

def patched_gather_qmm(self, x, indices, sorted_indices=False):
    """
    Replacement for gather_qmm that mimics the exact behavior:
    
    Input:
    - x: [batch, seq, 1, 1, in_dim] (from expand_dims(-2, -3))
    - indices: [batch, seq, topk]
    
    Output: 
    - [batch, seq, topk, 1, out_dim] — to be squeezed by SwitchGLU
    
    Original gather_qmm gathers from weight[indices] and does quantized matmul.
    We do the same but load only needed expert rows.
    """
    # Get shapes
    *batch_dims, in_dim = x.shape  
    *index_batch_dims, topk = indices.shape
    out_dim = self.output_dims
    
    # Get unique expert indices across the entire batch
    indices_flat = indices.reshape(-1)
    mx.eval(indices_flat)
    unique_experts = sorted(set(indices_flat.tolist()))
    
    # Load needed expert weights into cache
    expert_weights = {}
    for eidx in unique_experts:
        cache_key = (id(self), eidx)
        cached = CACHE.get(cache_key)
        if cached is not None:
            expert_weights[eidx] = cached
        else:
            CACHE.misses += 1
            # Load quantized weights for this expert
            w = self["weight"][eidx]      # [out_dim, packed_in_dim]
            s = self["scales"][eidx]      # [out_dim, num_groups]  
            b = self.get("biases")
            b = b[eidx] if b is not None else None
            
            # Dequantize this expert's weights
            w_dequant = mx.dequantize(w, s, b,
                                    group_size=self.group_size, 
                                    bits=self.bits)
            mx.eval(w_dequant)  # Force evaluation to avoid graph explosion
            
            expert_weights[eidx] = w_dequant
            CACHE.put(cache_key, w_dequant)
    
    # Compute outputs for each position and topk
    # x shape: [batch, seq, 1, 1, in_dim] from double expand_dims
    # Flatten to [batch*seq*1*1, in_dim] = [batch*seq, in_dim]  
    x_flat = x.reshape(-1, in_dim)
    indices_flat = indices.reshape(-1)  # [batch*seq*topk]
    
    outputs = []
    for i in range(len(indices_flat)):
        eidx = int(indices_flat[i].item())
        token_idx = i // topk  # Which token this expert applies to
        
        x_token = x_flat[token_idx:token_idx+1]  # [1, in_dim]
        w_expert = expert_weights[eidx]  # [out_dim, in_dim]
        
        out = x_token @ w_expert.T  # [1, out_dim]
        
        # Add bias if present
        if "bias" in self:
            bias = self["bias"][eidx:eidx+1]  # [1, out_dim]
            mx.eval(bias)
            out = out + bias
        
        outputs.append(out)
    
    # Stack outputs and reshape to target
    result = mx.concatenate(outputs, axis=0)  # [batch*seq*topk, out_dim]
    
    # Reshape to [batch, seq, topk, 1, out_dim] (extra 1 for SwitchGLU squeeze)
    target_shape = (*index_batch_dims, topk, 1, out_dim)
    result = result.reshape(target_shape)
    
    mx.eval(result)
    return result

def monkey_patch_gather_qmm(model):
    """Patch gather_qmm functionality in QuantizedSwitchLinear."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    
    # Replace __call__ method
    QuantizedSwitchLinear._original_call = QuantizedSwitchLinear.__call__
    QuantizedSwitchLinear.__call__ = patched_gather_qmm
    
    count = sum(1 for _, m in model.named_modules() 
                if isinstance(m, QuantizedSwitchLinear))
    return count

def main():
    print("=" * 60, flush=True)
    print("  ExpertFlow Engine v4 — gather_qmm Compatible", flush=True) 
    print("  DeepSeek V3.1 (378GB) → 128GB M5 Max", flush=True)
    print("=" * 60, flush=True)
    
    # Conservative memory settings
    mx.set_memory_limit(30 * 1024**3)  
    mx.set_cache_limit(10 * 1024**3)
    print(f"  MLX: 30GB memory limit, 10GB cache | Free: {free_gb():.1f} GB", flush=True)
    
    # Load model
    import mlx_lm
    print(f"\n  Loading DeepSeek V3.1...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL_PATH, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s | Free: {free_gb():.1f} GB", flush=True)
    
    # Patch expert computation
    count = monkey_patch_gather_qmm(model)
    print(f"  Patched {count} QuantizedSwitchLinear layers", flush=True)
    
    # Test generation
    test_prompts = [
        "Hi there!",
        "What is the capital of France?", 
        "Explain quantum computing in simple terms:"
    ]
    
    success_count = 0
    
    for prompt in test_prompts:
        print(f"\n  → {prompt!r}", flush=True)
        print(f"    ", end="", flush=True)
        
        try:
            t_start = time.time()
            token_count = 0
            
            for resp in mlx_lm.stream_generate(model, tokenizer, 
                                             prompt=prompt, max_tokens=20):
                text = resp.text if hasattr(resp, "text") else str(resp)
                token_count += 1
                sys.stdout.write(text)
                sys.stdout.flush()
            
            elapsed = time.time() - t_start
            tps = token_count / elapsed if elapsed > 0 else 0
            print(f"\n    ✅ {token_count} tokens, {elapsed:.1f}s, {tps:.1f} tok/s", flush=True)
            success_count += 1
            
        except Exception as e:
            print(f"\n    ❌ {e}", flush=True)
    
    print(f"\n  Cache performance: {CACHE.hit_rate:.1f}% hit rate", flush=True)
    print(f"  Free memory: {free_gb():.1f} GB", flush=True)
    
    if success_count > 0:
        print(f"\n  🚀 SUCCESS: {success_count}/{len(test_prompts)} prompts worked!", flush=True)
        print(f"  🔥 ExpertFlow: DeepSeek V3.1 running on consumer hardware!", flush=True)
        
        # Save success marker
        result = {
            "status": "success",
            "model": "DeepSeek V3.1 4-bit (378GB)",
            "hardware": "Apple M5 Max 128GB",
            "successful_prompts": success_count,
            "cache_hit_rate": round(CACHE.hit_rate, 1),
        }
        with open(os.path.expanduser("~/dev/expertflow/SUCCESS.json"), "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(f"\n  🔧 Still debugging - no successful generations yet", flush=True)

if __name__ == "__main__":
    main()