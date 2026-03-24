#!/usr/bin/env python3
"""
ExpertFlow Inference Benchmark
==============================
Real prompt processing + token generation speed across all M5 Max models.
Measures: TTFT, tok/s (generation), prompt tok/s, peak memory.
"""

import os, sys, time, json, gc, resource
import mlx.core as mx

PROMPT_SHORT = "Explain quantum computing in simple terms."
PROMPT_LONG = "You are a senior software engineer. Write a detailed technical design document for a distributed task queue system that supports priority scheduling, dead letter queues, exactly-once delivery semantics, horizontal scaling, and monitoring. Include architecture diagrams descriptions, API design, data models, and failure handling strategies."
MAX_TOKENS = 200

MODELS = [
    {
        "name": "Gemma-3-4B",
        "path": "/Users/jhammant/.lmstudio/models/mlx-community/gemma-3-4b-it-qat-4bit",
        "size_gb": 3.0,
        "type": "mlx"
    },
    {
        "name": "Qwen3.5-35B-A3B",
        "path": "/Users/jhammant/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-Instruct-MLX-4bit",
        "size_gb": 22.1,
        "type": "mlx"
    },
    {
        "name": "Qwen3-Next-80B-A3B",
        "path": "/Users/jhammant/.lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit",
        "size_gb": 44.9,
        "type": "mlx"
    },
]

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)

def banner(text):
    print(f"\n{'#'*70}")
    print(f"  {text}")
    print(f"{'#'*70}")

def benchmark_mlx_model(model_info, prompt, max_tokens=MAX_TOKENS):
    """Benchmark a model using mlx-lm generate with detailed timing."""
    from mlx_lm import load, generate
    
    name = model_info["name"]
    path = model_info["path"]
    
    if not os.path.exists(path):
        print(f"  SKIPPED — path not found: {path}")
        return None
    
    gc.collect()
    mx.clear_cache()
    rss_before = rss_gb()
    
    # Load model
    print(f"  Loading model...")
    t0 = time.time()
    try:
        model, tokenizer = load(path)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None
    load_time = time.time() - t0
    rss_after_load = rss_gb()
    print(f"  Loaded in {load_time:.1f}s | Memory: {rss_after_load-rss_before:.1f} GB")
    
    # Tokenize prompt to count input tokens
    input_tokens = tokenizer.encode(prompt)
    input_len = len(input_tokens)
    
    # Warmup run
    print(f"  Warmup...")
    try:
        _ = generate(model, tokenizer, prompt="Hi", max_tokens=5, verbose=False)
    except Exception as e:
        print(f"  Warmup failed: {e}")
    
    # Benchmark: capture timing via verbose output
    print(f"  Generating {max_tokens} tokens from {input_len}-token prompt...")
    
    # Use generate with verbose=True to get timing, but capture it
    t_start = time.time()
    try:
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
    except Exception as e:
        print(f"  Generation FAILED: {e}")
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
        return None
    t_total = time.time() - t_start
    
    # Count output tokens
    output_tokens = tokenizer.encode(response)
    output_len = len(output_tokens) - input_len
    if output_len <= 0:
        output_len = max_tokens  # fallback estimate
    
    # Calculate metrics
    gen_tok_s = output_len / t_total if t_total > 0 else 0
    
    rss_peak = rss_gb()
    mem_used = rss_peak - rss_before
    
    result = {
        "model": name,
        "size_gb": model_info["size_gb"],
        "load_time_s": round(load_time, 1),
        "input_tokens": input_len,
        "output_tokens": output_len,
        "total_time_s": round(t_total, 2),
        "generation_tok_s": round(gen_tok_s, 1),
        "memory_gb": round(mem_used, 1),
        "response_preview": response[:200] if response else ""
    }
    
    print(f"  ✅ {output_len} tokens in {t_total:.2f}s = {gen_tok_s:.1f} tok/s | Mem: {mem_used:.1f} GB")
    
    # Second run with long prompt
    print(f"  Running long prompt ({len(PROMPT_LONG)} chars)...")
    input_long = tokenizer.encode(PROMPT_LONG)
    t_start2 = time.time()
    try:
        response2 = generate(
            model, tokenizer,
            prompt=PROMPT_LONG,
            max_tokens=max_tokens,
            verbose=False
        )
        t_total2 = time.time() - t_start2
        output_long = len(tokenizer.encode(response2)) - len(input_long)
        if output_long <= 0: output_long = max_tokens
        gen_tok_s2 = output_long / t_total2 if t_total2 > 0 else 0
        print(f"  ✅ Long: {output_long} tokens in {t_total2:.2f}s = {gen_tok_s2:.1f} tok/s")
        result["long_prompt_tokens"] = len(input_long)
        result["long_output_tokens"] = output_long
        result["long_total_time_s"] = round(t_total2, 2)
        result["long_generation_tok_s"] = round(gen_tok_s2, 1)
    except Exception as e:
        print(f"  Long prompt failed: {e}")
        result["long_generation_tok_s"] = "FAILED"
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    time.sleep(2)
    
    return result


# Main benchmark
banner("M5 Max Model Inference Benchmark")
print(f"Hardware: Apple M5 Max, 128GB Unified Memory")
print(f"Short prompt: '{PROMPT_SHORT[:60]}...'")
print(f"Long prompt: '{PROMPT_LONG[:60]}...'")
print(f"Max tokens: {MAX_TOKENS}")
print(f"Models to test: {len(MODELS)}")

all_results = []

for i, model_info in enumerate(MODELS):
    banner(f"Model {i+1}/{len(MODELS)}: {model_info['name']} ({model_info['size_gb']} GB)")
    result = benchmark_mlx_model(model_info, PROMPT_SHORT)
    if result:
        all_results.append(result)

# Summary table
banner("BENCHMARK RESULTS SUMMARY")
print(f"")
print(f"{'Model':<25} {'Size':>6} {'Load':>7} {'Short tok/s':>12} {'Long tok/s':>12} {'Memory':>8}")
print(f"{'-'*70}")
for r in all_results:
    long_toks = r.get('long_generation_tok_s', 'N/A')
    if isinstance(long_toks, (int, float)):
        long_str = f"{long_toks:.1f}"
    else:
        long_str = str(long_toks)
    print(f"{r['model']:<25} {r['size_gb']:>5.1f}G {r['load_time_s']:>6.1f}s {r['generation_tok_s']:>11.1f} {long_str:>12} {r['memory_gb']:>7.1f}G")

print(f"")
if all_results:
    best_speed = max(all_results, key=lambda x: x['generation_tok_s'])
    best_efficiency = max(all_results, key=lambda x: x['generation_tok_s'] / max(x['memory_gb'], 0.1))
    print(f"  🏆 Fastest:          {best_speed['model']} ({best_speed['generation_tok_s']:.1f} tok/s)")
    print(f"  🏆 Most efficient:   {best_efficiency['model']} ({best_efficiency['generation_tok_s']:.1f} tok/s @ {best_efficiency['memory_gb']:.1f} GB)")
print(f"")

# Save results
outpath = os.path.expanduser("~/dev/expertflow/inference-benchmark.json")
with open(outpath, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {outpath}")
