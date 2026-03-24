#!/usr/bin/env python3
"""
ExpertFlow Optimizable Engine — Experiment 8: No Weight Cleanup
===============================================================
HYPOTHESIS: The original 25.0 score didn't have weight cleanup.
Maybe keeping weights in memory is faster than `del`.
"""

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

def dequant_expert(proj, expert_idx):
    w = proj["weight"][expert_idx]
    s = proj["scales"][expert_idx]  
    b = proj.get("biases")
    b = b[expert_idx] if b is not None else None
    return mx.dequantize(w, s, b, group_size=proj.group_size, bits=proj.bits)

def process_moe(moe_module, x, layer_idx):
    B, S, H = x.shape
    
    inds, scores = moe_module.gate(x)
    mx.eval(inds, scores)
    
    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp
    
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)
    
    x_flat = x.reshape(B * S, H)
    inds_flat = inds.reshape(B * S, topk)
    scores_flat = scores.reshape(B * S, topk)
    
    token_outputs = []
    
    for t in range(B * S):
        x_t = x_flat[t:t+1]
        token_out = mx.zeros((1, H))
        
        for k in range(topk):
            eidx = int(inds_flat[t, k].item())
            score = scores_flat[t, k]
            
            gate_w = dequant_expert(mlp.gate_proj, eidx)
            up_w = dequant_expert(mlp.up_proj, eidx)
            down_w = dequant_expert(mlp.down_proj, eidx)
            
            gate_out = x_t @ gate_w.T
            up_out = x_t @ up_w.T
            activated = nn.silu(gate_out) * up_out
            expert_out = activated @ down_w.T
            
            token_out = token_out + expert_out * score
            # NO `del` statements - let MLX/Python handle cleanup
        
        mx.eval(token_out)
        token_outputs.append(token_out)
    
    routed = mx.concatenate(token_outputs, axis=0).reshape(B, S, H)
    
    if shared is not None:
        result = routed + shared
    else:
        result = routed
    
    mx.eval(result)
    return result

def process_layer(layer, x, layer_idx, model=None):
    h = layer.input_layernorm(x)
    h = layer.self_attn(h)
    x = x + h
    mx.eval(x)
    
    h = layer.post_attention_layernorm(x)
    
    is_moe = hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp')
    
    if is_moe:
        h = process_moe(layer.mlp, h, layer_idx)
    else:
        h = layer.mlp(h)
        mx.eval(h)
    
    x = x + h
    mx.eval(x)
    return x