#!/usr/bin/env python3
"""
ExpertFlow Inference Benchmark v2 — All Models
===============================================
Tests MLX models via mlx-lm and GGUF models via LM Studio API.
"""

import os, sys, time, json, gc, resource, requests
import mlx.core as mx

PROMPT_SHORT = "Explain quantum computing in simple terms."
PROMPT_LONG = "You are a senior software engineer. Write a detailed technical design document for a distributed task queue system that supports priority scheduling, dead letter queues, exactly-once delivery semantics, horizontal scaling, and monitoring. Include architecture diagrams descriptions, API design, data models, and failure handling strategies."
MAX_TOKENS = 200

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)

def banner(text):
    print(f"\n{'#'*70}")
    print(f"  {text}")
    print(f"{'#'*70}", flush=True)

def bench_mlx(name, path, size_gb):
    """Benchmark MLX model via mlx-lm."""
    from mlx_lm import load, generate
    
    if not os.path.exists(path):
        print(f"  SKIPPED — not found: {path}")
        return None
    
    gc.collect(); mx.clear_cache()
    rss0 = rss_gb()
    
    print(f"  Loading...", flush=True)
    t0 = time.time()
    try:
        model, tokenizer = load(path)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None
    load_t = time.time() - t0
    mem = rss_gb() - rss0
    print(f"  Loaded in {load_t:.1f}s | {mem:.1f} GB")
    
    # Warmup
    try:
        generate(model, tokenizer, prompt="Hello", max_tokens=5, verbose=False)
    except: pass
    
    results = {"model": name, "size_gb": size_gb, "load_time_s": round(load_t, 1), "memory_gb": round(mem, 1), "backend": "mlx-lm"}
    
    for label, prompt in [("short", PROMPT_SHORT), ("long", PROMPT_LONG)]:
        input_toks = len(tokenizer.encode(prompt))
        print(f"  [{label}] {input_toks} input tokens → {MAX_TOKENS} max output...", flush=True)
        t0 = time.time()
        try:
            resp = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
            elapsed = time.time() - t0
            out_toks = len(tokenizer.encode(resp)) - input_toks
            if out_toks <= 0: out_toks = MAX_TOKENS
            tok_s = out_toks / elapsed
            print(f"  [{label}] ✅ {out_toks} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")
            results[f"{label}_input_tokens"] = input_toks
            results[f"{label}_output_tokens"] = out_toks
            results[f"{label}_time_s"] = round(elapsed, 2)
            results[f"{label}_tok_s"] = round(tok_s, 1)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            results[f"{label}_tok_s"] = "FAILED"
    
    del model, tokenizer
    gc.collect(); mx.clear_cache()
    time.sleep(2)
    return results

def bench_lmstudio(name, model_id, size_gb):
    """Benchmark GGUF model via LM Studio API (must be loaded first)."""
    API = "http://localhost:1234/v1"
    
    # Load model
    print(f"  Loading via LM Studio...", flush=True)
    t0 = time.time()
    try:
        r = requests.post(f"{API}/models/load", json={"model": model_id}, timeout=120)
        if r.status_code != 200:
            print(f"  Load failed: {r.status_code} {r.text[:200]}")
            return None
    except Exception as e:
        print(f"  LM Studio API error: {e}")
        return None
    load_t = time.time() - t0
    print(f"  Loaded in {load_t:.1f}s")
    
    results = {"model": name, "size_gb": size_gb, "load_time_s": round(load_t, 1), "backend": "lmstudio"}
    
    # Warmup
    try:
        requests.post(f"{API}/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        }, timeout=60)
    except: pass
    
    for label, prompt in [("short", PROMPT_SHORT), ("long", PROMPT_LONG)]:
        print(f"  [{label}] Generating {MAX_TOKENS} tokens...", flush=True)
        t0 = time.time()
        try:
            r = requests.post(f"{API}/chat/completions", json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.7
            }, timeout=300)
            elapsed = time.time() - t0
            data = r.json()
            usage = data.get("usage", {})
            out_toks = usage.get("completion_tokens", MAX_TOKENS)
            prompt_toks = usage.get("prompt_tokens", 0)
            tok_s = out_toks / elapsed if elapsed > 0 else 0
            print(f"  [{label}] ✅ {out_toks} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s (prompt: {prompt_toks} toks)")
            results[f"{label}_input_tokens"] = prompt_toks
            results[f"{label}_output_tokens"] = out_toks
            results[f"{label}_time_s"] = round(elapsed, 2)
            results[f"{label}_tok_s"] = round(tok_s, 1)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            results[f"{label}_tok_s"] = "FAILED"
    
    # Unload
    try:
        requests.post(f"{API}/models/unload", json={"model": model_id}, timeout=30)
    except: pass
    time.sleep(2)
    
    return results


# ═══════════════════════════════════════════════════════════════════
banner("M5 Max Full Model Inference Benchmark")
print(f"Hardware: Apple M5 Max, 128GB Unified Memory")
print(f"Max tokens: {MAX_TOKENS}")

all_results = []

# MLX models
mlx_models = [
    ("Gemma-3-4B (MLX 4bit)", "/Users/jhammant/.lmstudio/models/mlx-community/gemma-3-4b-it-qat-4bit", 3.0),
    ("GPT-OSS-20B (MLX MXFP4)", "/Users/jhammant/.lmstudio/models/mlx-community/gpt-oss-20b-MXFP4-Q8", 12.1),
    ("Qwen3-Coder-Next (MLX 4bit)", "/Users/jhammant/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-4bit", 44.9),
    ("Qwen3-Next-80B-A3B (MLX 4bit)", "/Users/jhammant/.lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit", 44.9),
]

for i, (name, path, size) in enumerate(mlx_models):
    banner(f"MLX Model {i+1}/{len(mlx_models)}: {name}")
    r = bench_mlx(name, path, size)
    if r: all_results.append(r)

# GGUF models via LM Studio
gguf_models = [
    ("Qwen3.5-35B-A3B (GGUF)", "qwen3.5-35b-a3b", 22.1),
    ("GPT-OSS-120B (GGUF)", "gpt-oss-120b", 66.0),
]

# Check if LM Studio API is running
try:
    r = requests.get("http://localhost:1234/v1/models", timeout=5)
    lms_running = r.status_code == 200
except:
    lms_running = False

if lms_running:
    for i, (name, model_id, size) in enumerate(gguf_models):
        banner(f"GGUF Model {i+1}/{len(gguf_models)}: {name}")
        r = bench_lmstudio(name, model_id, size)
        if r: all_results.append(r)
else:
    print("\n  ⚠️ LM Studio API not running — skipping GGUF models")

# Summary
banner("FULL BENCHMARK RESULTS")
print(f"")
print(f"{'Model':<35} {'Size':>6} {'Load':>7} {'Short':>8} {'Long':>8} {'Mem':>6} {'Backend':>10}")
print(f"{'-'*81}")
for r in all_results:
    short = f"{r.get('short_tok_s', 'N/A')}" if isinstance(r.get('short_tok_s'), (int, float)) else "FAIL"
    long_ = f"{r.get('long_tok_s', 'N/A')}" if isinstance(r.get('long_tok_s'), (int, float)) else "FAIL"
    mem = f"{r.get('memory_gb', '?')}G" if 'memory_gb' in r else "N/A"
    print(f"{r['model']:<35} {r['size_gb']:>5.1f}G {r['load_time_s']:>6.1f}s {short:>7} {long_:>7} {mem:>6} {r.get('backend','?'):>10}")

if all_results:
    valid = [r for r in all_results if isinstance(r.get('short_tok_s'), (int, float))]
    if valid:
        best = max(valid, key=lambda x: x['short_tok_s'])
        print(f"\n  🏆 Fastest: {best['model']} — {best['short_tok_s']} tok/s")
        
        # Efficiency ranking (tok/s per GB of memory)
        for r in valid:
            r['_eff'] = r['short_tok_s'] / max(r.get('memory_gb', r['size_gb']), 0.1)
        best_eff = max(valid, key=lambda x: x['_eff'])
        print(f"  🏆 Most efficient: {best_eff['model']} — {best_eff['short_tok_s']} tok/s @ {best_eff.get('memory_gb', best_eff['size_gb'])} GB")

outpath = os.path.expanduser("~/dev/expertflow/inference-benchmark-v2.json")
with open(outpath, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults: {outpath}")
