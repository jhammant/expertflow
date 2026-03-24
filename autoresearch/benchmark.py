#!/usr/bin/env python3
"""
ExpertFlow AutoResearch — Fixed Benchmark
==========================================
DO NOT MODIFY. This measures ExpertFlow performance.

Metrics:
  - layers_completed: How many layers processed successfully
  - tokens_generated: Actual tokens produced
  - time_per_layer_s: Average time per layer
  - peak_memory_gb: Peak memory usage
  - output_text: Generated text (for coherence check)
"""

import os, sys, time, json, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

# Fixed constants
MODEL_PATHS = {
    "glm": os.path.expanduser("~/models/glm-4.5-4bit"),
    "deepseek": os.path.expanduser("~/models/deepseek-v3.1-4bit"),
}
PROMPT = "The capital of France is"
MAX_TOKENS = 3
MEMORY_LIMIT_GB = 20

def get_memory():
    out = subprocess.check_output(["vm_stat"]).decode()
    f = i = 0
    for l in out.split("\n"):
        if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
        elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
    return (f + i) * 16384 / 1e9

def run_benchmark(model_name="glm"):
    """Run ExpertFlow benchmark. Returns results dict."""
    import mlx.core as mx
    # Let expertflow_opt.py handle device selection per-layer
    mx.set_memory_limit(MEMORY_LIMIT_GB * 1024**3)
    mx.set_cache_limit(int(MEMORY_LIMIT_GB * 0.3 * 1024**3))
    
    # Import the engine (the modifiable part)
    sys.path.insert(0, os.path.dirname(__file__))
    import expertflow_opt as engine
    
    import mlx_lm
    model_path = MODEL_PATHS[model_name]
    
    results = {
        "model": model_name,
        "prompt": PROMPT,
        "layers_completed": 0,
        "total_layers": 0,
        "tokens_generated": 0,
        "output_text": "",
        "layer_times": [],
        "peak_memory_gb": 0,
        "errors": [],
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    try:
        # Load model
        t0 = time.time()
        model, tokenizer = mlx_lm.load(model_path, lazy=True)
        results["load_time_s"] = round(time.time() - t0, 2)
        results["total_layers"] = len(model.model.layers)
        
        # Encode prompt
        input_ids = tokenizer.encode(PROMPT)
        
        # Run inference
        start_memory = get_memory()
        
        for token_step in range(MAX_TOKENS):
            t_token = time.time()
            current = mx.array([input_ids if token_step == 0 else input_ids])
            
            # Forward pass through layers
            x = model.model.embed_tokens(current)
            mx.eval(x)
            
            layers_this_token = 0
            for i, layer in enumerate(model.model.layers):
                t_layer = time.time()
                
                try:
                    x = engine.process_layer(layer, x, i, model)
                    layers_this_token = i + 1
                    
                    layer_time = time.time() - t_layer
                    results["layer_times"].append(round(layer_time, 3))
                    
                    # Track memory
                    mem = get_memory()
                    peak = MEMORY_LIMIT_GB + start_memory - mem
                    results["peak_memory_gb"] = max(results["peak_memory_gb"], round(peak, 1))
                    
                except Exception as e:
                    results["errors"].append(f"L{i}: {str(e)[:100]}")
                    break
            
            results["layers_completed"] = max(results["layers_completed"], layers_this_token)
            
            if layers_this_token < results["total_layers"]:
                break  # Didn't complete all layers
            
            # Final norm + logits
            try:
                x = model.model.norm(x)
                mx.eval(x)
                logits = model.lm_head(x[:, -1:, :])
                mx.eval(logits)
                
                next_id = int(mx.argmax(logits[0, 0]).item())
                input_ids.append(next_id)
                
                try:
                    text = tokenizer.decode([next_id])
                    results["output_text"] += text
                    results["tokens_generated"] += 1
                except:
                    pass
                    
            except Exception as e:
                results["errors"].append(f"head: {str(e)[:100]}")
                break
            
            results[f"token_{token_step}_time_s"] = round(time.time() - t_token, 2)
        
    except Exception as e:
        results["errors"].append(f"fatal: {str(e)[:200]}")
    
    # Summary stats
    if results["layer_times"]:
        results["avg_layer_time_s"] = round(sum(results["layer_times"]) / len(results["layer_times"]), 3)
        results["total_time_s"] = round(sum(results["layer_times"]), 1)
    
    results["completion_pct"] = round(results["layers_completed"] / max(results["total_layers"], 1) * 100, 1)
    results["free_memory_gb"] = round(get_memory(), 1)
    
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="glm", choices=["glm", "deepseek"])
    args = p.parse_args()
    
    print(f"Running ExpertFlow benchmark ({args.model})...", flush=True)
    results = run_benchmark(args.model)
    
    # Save results
    outfile = os.path.expanduser(f"~/dev/expertflow/bench_{results['timestamp']}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"  Model: {results['model']}")
    print(f"  Layers: {results['layers_completed']}/{results['total_layers']} ({results['completion_pct']}%)")
    print(f"  Tokens: {results['tokens_generated']}")
    print(f"  Output: {results['output_text']!r}")
    if results.get('avg_layer_time_s'):
        print(f"  Avg layer: {results['avg_layer_time_s']}s")
        print(f"  Total: {results['total_time_s']}s")
    if results['errors']:
        print(f"  Errors: {results['errors'][:3]}")
    print(f"  Saved: {outfile}")
