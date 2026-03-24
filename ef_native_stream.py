#!/usr/bin/env python3
"""
ExpertFlow v4 — Native MLX-LM Generate + Expert Streaming
=========================================================
Best of both worlds:
- MLX-LM's native generate (working KV cache, proper sampling)
- Our expert streaming (handles massive MoE models)

Strategy: Monkey-patch the MoE forward pass with our streaming version,
then use mlx_lm.generate() normally.
"""

import os, sys, time, json, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
from collections import OrderedDict

def free_gb():
    try:
        out = subprocess.check_output(["vm_stat"], timeout=2).decode()
        f = i = 0
        for l in out.split("\n"):
            if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
            elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
        return (f + i) * 16384 / 1e9
    except:
        return 999

# ═══ Expert Cache ═══
class ExpertCache:
    def __init__(self, budget_gb=20):
        self.cache = OrderedDict()
        self.budget = int(budget_gb * 1024**3)
        self.size = 0
        self.hits = self.misses = 0
        self.ENTRY = 84 * 1024**2
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        while self.size + self.ENTRY > self.budget and self.cache:
            self.cache.popitem(last=False)
            self.size -= self.ENTRY
        self.cache[key] = value
        self.size += self.ENTRY
    
    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / max(total, 1) * 100
        return f"{len(self.cache)} experts, {rate:.0f}% hit rate"

cache = ExpertCache()

def dequant_on_cpu(proj, eidx):
    """CPU dequant to avoid Metal timeout."""
    with mx.stream(mx.cpu):
        w = proj["weight"][eidx]
        s = proj["scales"][eidx]
        b = proj.get("biases")
        b = b[eidx] if b is not None else None
        result = mx.dequantize(w, s, b, group_size=proj.group_size, bits=proj.bits)
        mx.eval(result)
    return result

def get_expert(mlp, layer_idx, eidx):
    """Get cached or load expert weights."""
    key = (layer_idx, eidx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    
    gw = dequant_on_cpu(mlp.gate_proj, eidx)
    uw = dequant_on_cpu(mlp.up_proj, eidx)
    dw = dequant_on_cpu(mlp.down_proj, eidx)
    
    cache.put(key, (gw, uw, dw))
    return (gw, uw, dw)

def streaming_moe_forward(self, x):
    """
    Streaming MoE forward pass - replaces the original MoE.__call__.
    Uses our expert streaming but keeps interface compatible with MLX-LM.
    """
    B, S, H = x.shape
    
    # Router
    inds, scores = self.gate(x)
    mx.eval(inds, scores)
    
    topk = inds.shape[-1]
    mlp = self.switch_mlp
    layer_idx = getattr(self, '_layer_idx', 0)
    
    # Shared experts
    shared = None
    if hasattr(self, 'shared_experts') and self.shared_experts is not None:
        shared = self.shared_experts(x)
        mx.eval(shared)
    
    # Streaming expert computation
    x_flat = x.reshape(B * S, H)
    inds_2d = inds.reshape(B * S, topk)
    scores_2d = scores.reshape(B * S, topk)
    
    token_outs = []
    for t in range(B * S):
        x_t = x_flat[t:t+1]
        out = mx.zeros((1, H))
        
        for k in range(topk):
            eidx = int(inds_2d[t, k].item())
            score = scores_2d[t, k]
            
            gw, uw, dw = get_expert(mlp, layer_idx, eidx)
            
            g = x_t @ gw.T
            u = x_t @ uw.T
            out = out + (nn.silu(g) * u) @ dw.T * score
        
        mx.eval(out)
        token_outs.append(out)
    
    routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
    result = (routed + shared) if shared is not None else routed
    mx.eval(result)
    
    return result

def monkey_patch_moe(model):
    """Replace MoE forward passes with our streaming version."""
    layer_idx = 0
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
            # This is a MoE layer
            layer.mlp._layer_idx = layer_idx
            layer.mlp.__call__ = streaming_moe_forward.__get__(layer.mlp, layer.mlp.__class__)
            layer_idx += 1
    
    print(f"Monkey-patched {layer_idx} MoE layers for streaming")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    args = p.parse_args()
    
    print("=" * 60)
    print("  ExpertFlow v4 — Native Generate + Expert Streaming")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free: {free_gb():.1f}GB")
    
    # Memory settings
    mx.set_memory_limit(int(110 * 1024**3))
    mx.set_cache_limit(int(15 * 1024**3))
    try:
        mx.set_wired_limit(int(80 * 1024**3))
        print("  Wired: 80GB")
    except:
        pass
    
    import mlx_lm
    
    print("  Loading model...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    
    # Apply our expert streaming to MoE layers
    monkey_patch_moe(model)
    
    print(f"\n  Generating with native MLX-LM + expert streaming...")
    print(f"  Prompt: {args.prompt!r}")
    
    t_start = time.time()
    
    try:
        # Use MLX-LM's native generate - gets KV cache for free
        response = mlx_lm.generate(
            model, 
            tokenizer, 
            prompt=args.prompt, 
            max_tokens=args.max_tokens,
            verbose=True,
        )
        
        total_time = time.time() - t_start
        
        print(f"\n{'='*60}")
        print(f"  OUTPUT: {response}")
        print(f"  Time: {total_time:.1f}s")
        
        # Estimate token count
        output_text = response[len(args.prompt):]
        output_tokens = len(tokenizer.encode(output_text))
        
        if output_tokens > 0:
            print(f"  Tokens: {output_tokens}")
            print(f"  Speed: {output_tokens/total_time:.3f} tok/s")
        
        print(f"  Cache: {cache.stats()}")
        
        # Save results
        results = {
            "prompt": args.prompt,
            "output": output_text,
            "full_text": response,
            "tokens": output_tokens,
            "total_time_s": round(total_time, 1),
            "tok_s": round(output_tokens/total_time, 4) if output_tokens > 0 else 0,
            "cache_stats": cache.stats(),
            "approach": "native_mlx_lm + expert_streaming",
        }
        
        outfile = os.path.expanduser(f"~/dev/expertflow/native_result_{time.strftime('%H%M%S')}.json")
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {outfile}")
        
    except Exception as e:
        total_time = time.time() - t_start
        print(f"\nERROR after {total_time:.1f}s: {e}")
        print(f"Free: {free_gb():.1f}GB")

if __name__ == "__main__":
    main()