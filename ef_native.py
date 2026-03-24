#!/usr/bin/env python3
"""
ExpertFlow — Native MLX-LM Generate with Memory Tuning
======================================================
Skip custom MoE code entirely. Just use mlx-lm's native generate
with aggressive memory settings. If it works, we get real tok/s.
"""
import os, sys, time, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx

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

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    args = p.parse_args()
    
    print(f"Model: {os.path.basename(args.model)}")
    print(f"Free: {free_gb():.1f}GB")
    
    # Aggressive memory settings
    mx.set_memory_limit(0)  # No limit — let MLX use all available
    mx.set_cache_limit(0)   # No cache limit
    
    # Try wired limit to prevent macOS eviction
    try:
        mx.metal.set_wired_limit(int(100 * 1024**3))  # Wire 100GB
        print("Wired limit: 100GB")
    except Exception as e:
        print(f"Wired limit failed: {e}")
    
    import mlx_lm
    
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"Layers: {len(model.model.layers)}")
    
    print(f"\nGenerating (native mlx-lm)...")
    print(f"Prompt: {args.prompt!r}")
    
    t_start = time.time()
    
    try:
        response = mlx_lm.generate(
            model, 
            tokenizer, 
            prompt=args.prompt, 
            max_tokens=args.max_tokens,
            verbose=True,
        )
        
        total_time = time.time() - t_start
        
        print(f"\n{'='*50}")
        print(f"Output: {response}")
        print(f"Time: {total_time:.1f}s")
        
        # Count tokens in response
        resp_tokens = len(tokenizer.encode(response)) - len(tokenizer.encode(args.prompt))
        if resp_tokens > 0:
            print(f"Tokens: {resp_tokens}")
            print(f"Speed: {resp_tokens/total_time:.2f} tok/s")
        
    except Exception as e:
        total_time = time.time() - t_start
        print(f"\nERROR after {total_time:.1f}s: {e}")
        print(f"Free: {free_gb():.1f}GB")

if __name__ == "__main__":
    main()
