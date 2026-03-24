#!/usr/bin/env python3
"""
ExpertFlow Engine v3 — Fixed Shape Handling
============================================
Corrected tensor reshaping for DeepSeek V3.1 MoE layers.
"""

import os, sys, time, json, gc, subprocess
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
        self.hits = 0
        self.misses = 0

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

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t * 100 if t > 0 else 0


CACHE = ExpertLRU()


def patched_switch_linear_call(self, x, indices, sorted_indices=False):
    """
    Patched QuantizedSwitchLinear.__call__ with correct shape handling.
    
    Key insight: mx.gather_qmm expects:
    - x: [batch, seq, 1, in_dim] 
    - indices: [batch, seq, topk]
    - output: [batch, seq, topk, out_dim]
    
    We need to mimic this exactly.
    """
    # Debug shapes
    batch_size, seq_len = x.shape[:2] if x.ndim >= 2 else (1, 1)
    in_dim = x.shape[-1]
    out_dim = self.output_dims
    
    # Get unique experts needed across all positions
    indices_flat = indices.reshape(-1)
    mx.eval(indices_flat)
    unique_experts = sorted(set(indices_flat.tolist()))
    
    # Load and cache expert weights
    expert_weights = {}
    for eidx in unique_experts:
        cache_key = (id(self), eidx)
        cached = CACHE.get(cache_key)
        if cached is not None:
            expert_weights[eidx] = cached
        else:
            CACHE.misses += 1
            # Load this expert's weights
            w_slice = self["weight"][eidx]      # [out_dim, packed_in]
            s_slice = self["scales"][eidx]      # [out_dim, num_groups]
            b_slice = self.get("biases")
            b_slice = b_slice[eidx] if b_slice is not None else None
            
            # Dequantize
            w_deq = mx.dequantize(w_slice, s_slice, b_slice,
                                 group_size=self.group_size, bits=self.bits)
            mx.eval(w_deq)
            
            expert_weights[eidx] = w_deq
            CACHE.put(cache_key, w_deq)
    
    # Compute outputs maintaining exact shapes
    # x shape: [batch, seq, 1, in_dim] (from expand_dims in SwitchGLU)
    # indices shape: [batch, seq, topk]
    
    x_reshaped = x.reshape(batch_size * seq_len, 1, in_dim)  # Flatten spatial dims
    indices_reshaped = indices.reshape(batch_size * seq_len, -1)  # [batch*seq, topk]
    
    outputs = []
    for pos in range(batch_size * seq_len):
        x_pos = x_reshaped[pos:pos+1]  # [1, 1, in_dim]
        x_pos_2d = x_pos.reshape(1, in_dim)  # [1, in_dim]
        
        indices_pos = indices_reshaped[pos]  # [topk]
        topk = indices_pos.shape[0]
        
        pos_outputs = []
        for k in range(topk):
            eidx = int(indices_pos[k].item())
            w = expert_weights[eidx]  # [out_dim, in_dim]
            out = x_pos_2d @ w.T  # [1, out_dim]
            
            # Add bias if present
            if "bias" in self:
                bias_slice = self["bias"][eidx:eidx+1]  # [1, out_dim]
                mx.eval(bias_slice)
                out = out + bias_slice
            
            pos_outputs.append(out)
        
        # Stack expert outputs for this position: [topk, out_dim]
        pos_result = mx.concatenate(pos_outputs, axis=0)  # [topk, out_dim]
        outputs.append(pos_result)
    
    # Stack all positions: [batch*seq, topk, out_dim]
    result = mx.concatenate([out[None, :, :] for out in outputs], axis=0)
    
    # Reshape to target: [batch, seq, topk, out_dim]
    topk = indices.shape[-1]
    result = result.reshape(batch_size, seq_len, topk, out_dim)
    
    mx.eval(result)
    return result


def monkey_patch_v3(model):
    """Patch QuantizedSwitchLinear at class level."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    
    QuantizedSwitchLinear._original_call = QuantizedSwitchLinear.__call__
    QuantizedSwitchLinear.__call__ = patched_switch_linear_call
    
    count = sum(1 for _, m in model.named_modules() 
                if isinstance(m, QuantizedSwitchLinear))
    return count


def main():
    print("=" * 60, flush=True)
    print("  ExpertFlow Engine v3 — Shape-Fixed", flush=True)
    print("  DeepSeek V3.1 (378GB) → 128GB M5 Max", flush=True)
    print("=" * 60, flush=True)
    print(f"  Free memory: {free_gb():.1f} GB", flush=True)
    
    # Set conservative memory limits
    mx.set_memory_limit(25 * 1024**3)
    mx.set_cache_limit(8 * 1024**3)
    print(f"  MLX limits: 25GB memory, 8GB cache", flush=True)
    
    # Load model
    import mlx_lm
    print(f"\n  Loading model (lazy)...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(MODEL_PATH, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s | Free: {free_gb():.1f} GB", flush=True)
    
    # Patch expert layers  
    count = monkey_patch_v3(model)
    print(f"  Patched {count} expert layers for dynamic loading", flush=True)
    
    # Generate
    prompts = ["Hi", "What is 2+2?", "The capital of France is"]
    
    for prompt in prompts:
        print(f"\n  → Prompt: {prompt!r}", flush=True)
        print(f"    Output: ", end="", flush=True)
        
        t_start = time.time()
        token_count = 0
        full_text = ""
        
        try:
            for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=15):
                text = resp.text if hasattr(resp, "text") else str(resp)
                full_text += text
                token_count += 1
                sys.stdout.write(text)
                sys.stdout.flush()
            
            elapsed = time.time() - t_start
            tps = token_count / elapsed if elapsed > 0 and token_count > 0 else 0
            
            print(f"  [{token_count} tok, {elapsed:.1f}s, {tps:.1f} tok/s]", flush=True)
            
        except Exception as e:
            print(f"\n    ❌ Error: {e}", flush=True)
            # Don't crash on first prompt failure
            if "reshape" in str(e):
                print("    (Shape error - continuing to debug...)", flush=True)
    
    print(f"\n  Cache: {CACHE.hit_rate:.1f}% hit rate", flush=True)
    print(f"  Free memory: {free_gb():.1f} GB", flush=True)
    
    # Save a test result
    if token_count > 0:
        result = {
            "model": "DeepSeek V3.1 4-bit",
            "status": "success",
            "cache_hit_rate": round(CACHE.hit_rate, 1),
            "tokens_generated": token_count,
        }
        with open(os.path.expanduser("~/dev/expertflow/test-result.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n  🔥 ExpertFlow: FIRST WORKING DEEPSEEK V3.1 TOKENS!", flush=True)
    else:
        print(f"\n  🔧 Still debugging shape issues...", flush=True)


if __name__ == "__main__":
    main()