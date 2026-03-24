#!/usr/bin/env python3
"""
Single Token Generation Test
============================
Minimal test to get actual tok/s on both models.
Just generate 1 token to measure baseline speed.
"""

import os, sys, time, json
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

def single_token_test(model_path, prompt="Hi"):
    """Generate exactly 1 token and measure time."""
    
    print(f"Testing: {os.path.basename(model_path)}")
    print(f"Prompt: {prompt!r}")
    
    # Use CPU to avoid Metal timeout
    mx.set_default_device(mx.cpu)
    mx.set_memory_limit(int(50 * 1024**3))  # 50GB limit
    
    import mlx_lm
    
    t_load = time.time()
    model, tokenizer = mlx_lm.load(model_path, lazy=True)
    load_time = time.time() - t_load
    
    print(f"Loaded in {load_time:.1f}s")
    print(f"Layers: {len(model.model.layers)}")
    
    # Encode
    input_ids = tokenizer.encode(prompt)
    print(f"Input tokens: {len(input_ids)}")
    
    # Single forward pass
    t_start = time.time()
    
    tokens = mx.array([input_ids])
    logits = model(tokens)
    mx.eval(logits)
    
    # Sample
    next_id = int(mx.argmax(logits[0, -1, :]).item())
    
    generation_time = time.time() - t_start
    
    # Decode
    try:
        output_text = tokenizer.decode([next_id])
    except:
        output_text = f"[ID:{next_id}]"
    
    print(f"Generated: {output_text!r}")
    print(f"Time: {generation_time:.2f}s")
    print(f"Speed: {1/generation_time:.2f} tok/s")
    
    return {
        "model": os.path.basename(model_path),
        "prompt": prompt,
        "output": output_text,
        "load_time_s": round(load_time, 1),
        "generation_time_s": round(generation_time, 2),
        "tok_s": round(1/generation_time, 2),
        "layers": len(model.model.layers),
    }

def main():
    models = [
        "~/models/glm-4.5-4bit",
        "~/models/deepseek-v3.1-4bit"
    ]
    
    results = []
    
    for model_path in models:
        full_path = os.path.expanduser(model_path)
        if os.path.exists(full_path):
            print("="*50)
            try:
                result = single_token_test(full_path)
                results.append(result)
                print(f"✅ {result['model']}: {result['tok_s']} tok/s")
            except Exception as e:
                print(f"❌ {os.path.basename(full_path)}: {e}")
                results.append({
                    "model": os.path.basename(full_path),
                    "error": str(e)
                })
        else:
            print(f"❌ Model not found: {full_path}")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for r in results:
        if 'tok_s' in r:
            print(f"{r['model']}: {r['tok_s']} tok/s ({r['generation_time_s']}s)")
        else:
            print(f"{r['model']}: ERROR")
    
    # Save
    outfile = os.path.expanduser(f"~/dev/expertflow/token_speed_test_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")

if __name__ == "__main__":
    main()