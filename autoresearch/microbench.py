#!/usr/bin/env python3
"""
ExpertFlow MicroBenchmark — Fast Iteration
==========================================
Tests just 5 MoE layers for rapid optimization cycles.
Target: <30 seconds per experiment.
"""

import os, sys, time, json, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

MODEL_PATH = os.path.expanduser("~/models/glm-4.5-4bit")
TEST_LAYERS = 5  # Only test first 5 MoE layers
PROMPT = "Hello"
MEMORY_LIMIT = 20

def free_gb():
    out = subprocess.check_output(["vm_stat"]).decode()
    f = i = 0
    for l in out.split("\n"):
        if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
        elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
    return (f + i) * 16384 / 1e9

def micro_benchmark():
    """Fast benchmark: just test MoE layer processing."""
    import mlx.core as mx
    mx.set_memory_limit(MEMORY_LIMIT * 1024**3)
    
    sys.path.insert(0, os.path.dirname(__file__))
    import expertflow_opt as engine
    
    import mlx_lm
    
    results = {
        "test": "microbench",
        "layers_tested": TEST_LAYERS,
        "layer_times": [],
        "errors": [],
        "timestamp": time.strftime("%H%M%S"),
    }
    
    try:
        # Load model  
        t0 = time.time()
        model, tokenizer = mlx_lm.load(MODEL_PATH, lazy=True)
        results["load_time"] = round(time.time() - t0, 1)
        
        # Encode prompt
        input_ids = tokenizer.encode(PROMPT)
        
        # Minimal forward pass
        current = mx.array([input_ids])
        x = model.model.embed_tokens(current)
        mx.eval(x)
        
        # Test first TEST_LAYERS MoE layers
        moe_count = 0
        for i, layer in enumerate(model.model.layers):
            if moe_count >= TEST_LAYERS:
                break
                
            is_moe = hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp')
            if not is_moe:
                continue
            
            print(f"Testing MoE layer {i}...", flush=True)
            t_layer = time.time()
            
            try:
                x = engine.process_layer(layer, x, i, model)
                layer_time = time.time() - t_layer
                results["layer_times"].append(round(layer_time, 2))
                moe_count += 1
                print(f"  L{i}: {layer_time:.2f}s")
                
            except Exception as e:
                error_msg = f"L{i}: {str(e)[:100]}"
                results["errors"].append(error_msg)
                print(f"  ERROR: {error_msg}")
                break
        
        results["completed_layers"] = len(results["layer_times"])
        results["avg_layer_time"] = round(sum(results["layer_times"]) / max(len(results["layer_times"]), 1), 2)
        results["total_time"] = round(sum(results["layer_times"]), 1)
        
    except Exception as e:
        results["errors"].append(f"Fatal: {str(e)}")
    
    results["free_gb"] = round(free_gb(), 1)
    return results

if __name__ == "__main__":
    print("🏃 ExpertFlow MicroBench — Fast Iteration")
    t0 = time.time()
    
    results = micro_benchmark()
    elapsed = time.time() - t0
    
    print(f"\n{'='*40}")
    print(f"📊 RESULTS ({elapsed:.1f}s total)")
    print(f"{'='*40}")
    print(f"Completed: {results.get('completed_layers', 0)}/{TEST_LAYERS}")
    if results.get('avg_layer_time'):
        print(f"Avg layer: {results['avg_layer_time']}s")
        print(f"Total MoE: {results['total_time']}s")
    if results['errors']:
        print(f"Errors: {results['errors']}")
    print(f"Memory: {results['free_gb']}GB free")
    
    # Quick save
    outfile = f"micro_{results['timestamp']}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {outfile}")
    
    # Performance score (higher = better)
    if results.get('completed_layers', 0) > 0:
        score = results['completed_layers'] / max(results['avg_layer_time'], 0.1)
        print(f"🎯 SCORE: {score:.1f} (layers/second)")
    else:
        print("💀 FAILED: No layers completed")