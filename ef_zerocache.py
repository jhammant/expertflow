#!/usr/bin/env python3
"""
ExpertFlow — Zero-Cache GPU Inference
======================================
Strategy: Never accumulate expert weights in memory.
1. CPU prefault each expert (dequant + eval on CPU)
2. GPU matmul immediately
3. Delete weights immediately after use
4. mx.clear_cache() after every layer

The previous runs died from memory pressure because cached experts
accumulated. This version keeps ZERO cache — trades speed for completion.
"""

import os, sys, time, json, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
import gc

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

def dequant_on_cpu(proj, eidx):
    """Dequantize on CPU stream, eval to force NVMe reads."""
    with mx.stream(mx.cpu):
        w = proj["weight"][eidx]
        s = proj["scales"][eidx]
        b = proj.get("biases")
        b = b[eidx] if b is not None else None
        result = mx.dequantize(w, s, b, group_size=proj.group_size, bits=proj.bits)
        mx.eval(result)
    return result

def moe_forward(moe_module, x, layer_idx):
    B, S, H = x.shape
    
    # Router
    inds, scores = moe_module.gate(x)
    mx.eval(inds, scores)
    
    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp
    
    # Shared experts
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)
    
    # Process tokens — zero cache, immediate cleanup
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
            
            # CPU prefault → GPU matmul → delete
            gw = dequant_on_cpu(mlp.gate_proj, eidx)
            uw = dequant_on_cpu(mlp.up_proj, eidx)
            
            g = x_t @ gw.T
            u = x_t @ uw.T
            del gw, uw  # Free immediately
            
            activated = nn.silu(g) * u
            del g, u
            
            dw = dequant_on_cpu(mlp.down_proj, eidx)
            expert_out = activated @ dw.T
            del dw, activated
            
            out = out + expert_out * score
            mx.eval(out)  # Eval per expert
        
        token_outs.append(out)
    
    routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
    result = (routed + shared) if shared is not None else routed
    mx.eval(result)
    
    # Aggressive cleanup after every MoE layer
    del token_outs, routed
    if shared is not None:
        del shared
    mx.clear_cache()
    gc.collect()
    
    return result

def generate(model, tokenizer, prompt, max_tokens):
    input_ids = tokenizer.encode(prompt)
    total_layers = len(model.model.layers)
    generated = []
    token_times = []
    
    print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
    print(f"  Layers: {total_layers}", flush=True)
    
    for step in range(max_tokens):
        t0 = time.time()
        
        all_ids = input_ids + generated
        x = model.model.embed_tokens(mx.array([all_ids]))
        mx.eval(x)
        
        for i, layer in enumerate(model.model.layers):
            # Attention
            h = layer.input_layernorm(x)
            h = layer.self_attn(h)
            x = x + h
            mx.eval(x)
            
            # MLP/MoE
            h = layer.post_attention_layernorm(x)
            mx.eval(h)
            
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
                h = moe_forward(layer.mlp, h, i)
            else:
                h = layer.mlp(h)
                mx.eval(h)
            
            x = x + h
            mx.eval(x)
            
            # Cleanup every layer
            mx.clear_cache()
            
            if (i+1) % 10 == 0:
                f = free_gb()
                print(f"[L{i+1}/{total_layers} {f:.0f}G]", end=" ", flush=True)
        
        # Logits
        x = model.model.norm(x)
        logits = model.lm_head(x[:, -1:, :])
        mx.eval(logits)
        
        next_id = int(mx.argmax(logits[0, 0]).item())
        generated.append(next_id)
        
        dt = time.time() - t0
        token_times.append(dt)
        
        try:
            text = tokenizer.decode([next_id])
        except:
            text = f"[{next_id}]"
        
        print(f"\n  ✅ Token {step+1}: {text!r} ({dt:.1f}s | {1/dt:.3f} tok/s)", flush=True)
        
        if hasattr(tokenizer, 'eos_token_id') and next_id == tokenizer.eos_token_id:
            break
    
    try:
        full = tokenizer.decode(generated)
    except:
        full = str(generated)
    
    avg = len(generated) / sum(token_times) if token_times else 0
    return {
        "prompt": prompt,
        "output": full,
        "full_text": prompt + full,
        "tokens": len(generated),
        "avg_tok_s": round(avg, 4),
        "times": [round(t, 2) for t in token_times],
        "total_s": round(sum(token_times), 1),
    }

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=5)
    args = p.parse_args()
    
    print("=" * 60)
    print("  ExpertFlow — Zero-Cache GPU Inference")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free: {free_gb():.1f}GB")
    
    # Let MLX manage memory, set generous limits
    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(10 * 1024**3))
    
    # Try wired limit
    try:
        mx.set_wired_limit(int(80 * 1024**3))
        print("  Wired: 80GB")
    except:
        pass
    
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    
    results = generate(model, tokenizer, args.prompt, args.max_tokens)
    
    print(f"\n{'=' * 60}")
    print(f"  OUTPUT: {results['full_text']}")
    print(f"  SPEED:  {results['avg_tok_s']} tok/s")
    print(f"  TOTAL:  {results['total_s']}s for {results['tokens']} tokens")
    
    outfile = os.path.expanduser(f"~/dev/expertflow/zerocache_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")

if __name__ == "__main__":
    main()
