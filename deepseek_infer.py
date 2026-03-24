#!/usr/bin/env python3
"""
ExpertFlow: Real DeepSeek V3.1 Inference on M5 Max
====================================================
Actually generates tokens from a 378GB model on 128GB RAM.
Uses mlx-lm with lazy loading — lets Apple's unified memory + NVMe handle the rest.
"""

import time, os, sys, signal, json, resource

# Timeout handler
def alarm_handler(signum, frame):
    print("\n⏰ Timeout reached — stopping", flush=True)
    sys.exit(0)

signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(300)  # 5 min max

os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx_lm

MODEL_PATH = os.path.expanduser("~/models/deepseek-v3.1-4bit")

def get_free_gb():
    import subprocess
    out = subprocess.check_output(["vm_stat"]).decode()
    free = inactive = 0
    for line in out.split("\n"):
        if "Pages free" in line:
            free = int(line.split()[-1].rstrip("."))
        elif "Pages inactive" in line:
            inactive = int(line.split()[-1].rstrip("."))
    return (free + inactive) * 16384 / 1e9

print("=" * 60, flush=True)
print("  DeepSeek V3.1 (671B/37B active) — Real Inference", flush=True)
print("=" * 60, flush=True)
print(f"  Free memory: {get_free_gb():.1f} GB", flush=True)

# Phase 1: Load model (lazy)
print(f"\n  Loading model (lazy)...", flush=True)
t0 = time.time()
model, tokenizer = mlx_lm.load(MODEL_PATH, lazy=True)
load_time = time.time() - t0
print(f"  Model loaded in {load_time:.1f}s", flush=True)
print(f"  Free memory: {get_free_gb():.1f} GB", flush=True)

# Phase 2: Generate tokens
prompts = [
    "What is the capital of France?",
]

results = []

for prompt in prompts:
    print(f"\n  Prompt: {prompt!r}", flush=True)
    print(f"  Free memory before: {get_free_gb():.1f} GB", flush=True)
    
    t0 = time.time()
    
    # Track token generation
    tokens_generated = 0
    first_token_time = None
    
    response = mlx_lm.generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=20,
        verbose=True,
    )
    
    gen_time = time.time() - t0
    free_after = get_free_gb()
    
    # Count tokens in response
    response_tokens = len(tokenizer.encode(response)) - len(tokenizer.encode(prompt))
    tok_s = response_tokens / gen_time if gen_time > 0 else 0
    
    print(f"\n  Response: {response}", flush=True)
    print(f"  Tokens: {response_tokens}", flush=True)
    print(f"  Time: {gen_time:.1f}s", flush=True)
    print(f"  Speed: {tok_s:.2f} tok/s", flush=True)
    print(f"  Free memory after: {free_after:.1f} GB", flush=True)
    
    results.append({
        "prompt": prompt,
        "response": response,
        "tokens": response_tokens,
        "time_s": round(gen_time, 1),
        "tok_s": round(tok_s, 2),
        "free_gb_before": round(get_free_gb(), 1),
    })

# Save results
outpath = os.path.expanduser("~/dev/expertflow/deepseek-inference.json")
with open(outpath, "w") as f:
    json.dump({
        "model": "DeepSeek V3.1 4-bit",
        "total_params": "671B",
        "active_params": "37B",
        "model_size_gb": 378,
        "ram_gb": 128,
        "ratio": "2.95x RAM",
        "results": results,
    }, f, indent=2)

print(f"\n  Results saved: {outpath}", flush=True)
print(f"\n  🔥 DeepSeek V3.1 inference COMPLETE on 128GB M5 Max!", flush=True)
