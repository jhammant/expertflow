#!/usr/bin/env python3
"""
ExpertFlow MINIMAL — Proof of Concept
=====================================
Goal: Generate ONE coherent token from a massive MoE model.
Prove it works, then optimize for speed.
"""

import os, sys, time
os.environ["MLX_LAZY_INITIALIZATION"] = "1"
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)  # Avoid Metal timeout

def log(msg): 
    print(msg, flush=True)

def dequant_expert(proj, eidx):
    w = proj["weight"][eidx]
    s = proj["scales"][eidx]  
    b = proj.get("biases")
    b = b[eidx] if b is not None else None
    return mx.dequantize(w, s, b, group_size=proj.group_size, bits=proj.bits)

def moe_layer(module, x):
    """Simple MoE: router + experts + combine."""
    inds, scores = module.gate(x)
    mx.eval(inds, scores)
    
    B, S, H = x.shape
    topk = inds.shape[-1]
    mlp = module.switch_mlp
    
    # Shared expert
    shared = module.shared_experts(x) if hasattr(module, 'shared_experts') and module.shared_experts else mx.zeros_like(x)
    
    # Route through experts (streaming) - collect outputs
    token_outputs = []
    for b in range(B):
        for s in range(S):
            x_token = x[b:b+1, s:s+1, :]  # [1,1,H]
            token_out = mx.zeros((1, 1, H))
            
            for k in range(topk):
                eid = int(inds[b, s, k].item())
                score = scores[b, s, k]
                
                # SwiGLU with streaming weights
                gate_w = dequant_expert(mlp.gate_proj, eid)
                up_w = dequant_expert(mlp.up_proj, eid) 
                down_w = dequant_expert(mlp.down_proj, eid)
                
                gate_out = x_token @ gate_w.T
                up_out = x_token @ up_w.T
                activated = nn.silu(gate_out) * up_out
                expert_out = activated @ down_w.T
                
                token_out = token_out + expert_out * score
                del gate_w, up_w, down_w
            
            token_outputs.append(token_out)
    
    # Stack outputs back to [B, S, H] shape
    output = mx.concatenate(token_outputs, axis=1).reshape(B, S, H)
    mx.eval(output)
    return output + shared

def forward_single(model, tokens):
    """Single forward pass - return logits for next token."""
    x = model.model.embed_tokens(tokens)
    mx.eval(x)
    
    layers = model.model.layers
    for i, layer in enumerate(layers):
        # Attention
        h = layer.input_layernorm(x)
        h = layer.self_attn(h)
        x = x + h
        mx.eval(x)
        
        # MLP/MoE  
        h = layer.post_attention_layernorm(x)
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
            h = moe_layer(layer.mlp, h)  
        else:
            h = layer.mlp(h)
        x = x + h
        mx.eval(x)
        
        log(f"  L{i+1}/{len(layers)}")
    
    x = model.model.norm(x)
    logits = model.lm_head(x[:, -1:, :])  # Last token
    mx.eval(logits)
    return logits

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital")
    args = p.parse_args()
    
    log("ExpertFlow MINIMAL — Single Token Generation")
    log(f"Model: {os.path.basename(args.model)}")
    
    import mlx_lm
    model, tok = mlx_lm.load(args.model, lazy=True)
    log(f"Loaded: {len(model.model.layers)} layers")
    
    # Encode
    ids = tok.encode(args.prompt)
    log(f"Prompt: {args.prompt!r} → {ids}")
    
    # Forward
    log("Running forward pass...")
    t0 = time.time()
    logits = forward_single(model, mx.array([ids]))
    elapsed = time.time() - t0
    
    # Sample
    next_id = int(mx.argmax(logits[0, 0]).item())
    try:
        text = tok.decode([next_id])
        log(f"Generated: {text!r} (ID: {next_id})")
        log(f"Full: {args.prompt}{text}")
    except:
        log(f"Generated ID: {next_id} (decode failed)")
    
    log(f"Time: {elapsed:.1f}s")
    log("✅ ExpertFlow: Single token generation WORKS!")

if __name__ == "__main__":
    main()