#!/usr/bin/env python3
"""
Deep Benchmark — Multiple runs, multiple prompt lengths, warmup, statistics.
No hand-waving. Real numbers with variance.
"""

import os, time, json, gc, resource, statistics
import mlx.core as mx
from mlx_lm import load, generate

PROMPTS = {
    "tiny": "Hi",
    "short": "Explain quantum computing in simple terms.",
    "medium": "Write a Python function that implements a binary search tree with insert, delete, and search operations. Include proper error handling and type hints.",
    "long": "You are a senior software engineer. Write a detailed technical design document for a distributed task queue system that supports priority scheduling, dead letter queues, exactly-once delivery semantics, horizontal scaling, and monitoring. Include architecture descriptions, API design, data models, and failure handling strategies. Be thorough and specific.",
}
MAX_TOKENS = 200
RUNS = 5  # Multiple runs per test for statistics

MODELS = [
    ("Gemma-3-4B", "/Users/jhammant/.lmstudio/models/mlx-community/gemma-3-4b-it-qat-4bit", 3.0),
    ("GPT-OSS-20B", "/Users/jhammant/.lmstudio/models/mlx-community/gpt-oss-20b-MXFP4-Q8", 12.1),
]

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)

def banner(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}", flush=True)

all_results = []

for model_name, model_path, size_gb in MODELS:
    banner(f"{model_name} ({size_gb} GB) — {RUNS} runs per prompt")
    
    if not os.path.exists(model_path):
        print(f"  SKIPPED — not found")
        continue
    
    gc.collect(); mx.clear_cache()
    rss0 = rss_gb()
    
    t0 = time.time()
    model, tokenizer = load(model_path)
    load_t = time.time() - t0
    mem = rss_gb() - rss0
    print(f"  Loaded in {load_t:.1f}s | Memory: {mem:.1f} GB")
    
    # Warmup — 3 generations to get GPU hot
    print(f"  Warming up (3 generations)...")
    for _ in range(3):
        generate(model, tokenizer, prompt="Hello world", max_tokens=50, verbose=False)
    
    model_results = {
        "model": model_name, "size_gb": size_gb,
        "load_time_s": round(load_t, 1), "memory_gb": round(mem, 1),
        "runs_per_test": RUNS, "max_tokens": MAX_TOKENS,
        "prompts": {}
    }
    
    for prompt_name, prompt_text in PROMPTS.items():
        input_toks = len(tokenizer.encode(prompt_text))
        print(f"\n  [{prompt_name}] {input_toks} input tokens, {RUNS} runs:")
        
        run_times = []
        run_toks = []
        run_tok_s = []
        
        for run in range(RUNS):
            t0 = time.time()
            resp = generate(model, tokenizer, prompt=prompt_text, max_tokens=MAX_TOKENS, verbose=False)
            elapsed = time.time() - t0
            
            out_toks = len(tokenizer.encode(resp)) - input_toks
            if out_toks <= 0: out_toks = MAX_TOKENS
            tok_s = out_toks / elapsed
            
            run_times.append(elapsed)
            run_toks.append(out_toks)
            run_tok_s.append(tok_s)
            print(f"    Run {run+1}: {out_toks} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")
        
        avg_tok_s = statistics.mean(run_tok_s)
        std_tok_s = statistics.stdev(run_tok_s) if len(run_tok_s) > 1 else 0
        min_tok_s = min(run_tok_s)
        max_tok_s = max(run_tok_s)
        avg_time = statistics.mean(run_times)
        avg_toks = statistics.mean(run_toks)
        
        print(f"    ────────────────────────────────────")
        print(f"    AVG: {avg_tok_s:.1f} tok/s ± {std_tok_s:.1f} | Min: {min_tok_s:.1f} | Max: {max_tok_s:.1f}")
        print(f"    AVG time: {avg_time:.2f}s | AVG tokens: {avg_toks:.0f}")
        
        model_results["prompts"][prompt_name] = {
            "input_tokens": input_toks,
            "avg_output_tokens": round(avg_toks),
            "avg_tok_s": round(avg_tok_s, 1),
            "std_tok_s": round(std_tok_s, 1),
            "min_tok_s": round(min_tok_s, 1),
            "max_tok_s": round(max_tok_s, 1),
            "avg_time_s": round(avg_time, 2),
            "runs": [{"tok_s": round(t, 1), "tokens": n, "time_s": round(e, 2)} 
                     for t, n, e in zip(run_tok_s, run_toks, run_times)]
        }
    
    all_results.append(model_results)
    del model, tokenizer
    gc.collect(); mx.clear_cache()
    time.sleep(3)

# Summary
banner("DEEP BENCHMARK SUMMARY")
print(f"")
print(f"{'Model':<18} {'Prompt':<10} {'Avg tok/s':>10} {'± StdDev':>9} {'Min':>8} {'Max':>8} {'Avg Time':>9}")
print(f"{'-'*72}")
for r in all_results:
    for pname in PROMPTS:
        if pname in r["prompts"]:
            p = r["prompts"][pname]
            print(f"{r['model']:<18} {pname:<10} {p['avg_tok_s']:>9.1f}  {p['std_tok_s']:>8.1f}  {p['min_tok_s']:>7.1f}  {p['max_tok_s']:>7.1f}  {p['avg_time_s']:>8.2f}s")
    print()

outpath = os.path.expanduser("~/dev/expertflow/deep-benchmark.json")
with open(outpath, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results: {outpath}")
