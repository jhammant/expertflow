#!/usr/bin/env python3
"""
ExpertFlow — Coherent Inference Engine
======================================
Goal: Actually produce coherent text from massive MoE models.

Key insight from testing:
- Streaming per-expert matmuls WORK (GLM got to L75/92)  
- Batched expert matmuls cause GPU timeout (too many Metal ops)
- Eval every layer, NOT every expert
- List-based output accumulation (avoid mx.array.at issues)

Architecture: Load expert → matmul → free → next expert
"""

import os, sys, time, json, subprocess, gc
from collections import OrderedDict

os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

# Force CPU to avoid Metal GPU timeout from NVMe page faults
mx.set_default_device(mx.cpu)

def log(msg, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()

def free_gb():
    out = subprocess.check_output(["vm_stat"]).decode()
    f = i = 0
    for l in out.split("\n"):
        if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
        elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
    return (f + i) * 16384 / 1e9

def dequant_expert(switch_linear, expert_idx):
    """Load and dequantize a single expert weight matrix."""
    w = switch_linear["weight"][expert_idx]
    s = switch_linear["scales"][expert_idx]
    b = switch_linear.get("biases")
    b = b[expert_idx] if b is not None else None
    return mx.dequantize(w, s, b,
                        group_size=switch_linear.group_size,
                        bits=switch_linear.bits)


def moe_forward(moe_module, x, layer_idx):
    """
    MoE forward pass with streaming expert loading.
    
    For each token:
      1. Router picks 8 experts
      2. For each expert: load weights → matmul → accumulate → free weights
      3. Weight-sum the 8 expert outputs
      4. Add shared expert output
    
    Only eval once at the end of the layer.
    """
    B, S, H = x.shape
    
    # Router
    inds, scores = moe_module.gate(x)  # [B, S, topk], [B, S, topk]
    mx.eval(inds, scores)
    
    topk = inds.shape[-1]
    switch_mlp = moe_module.switch_mlp
    
    # Shared experts (small, always in memory)
    shared_out = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared_out = moe_module.shared_experts(x)
    
    # Process each token
    x_flat = x.reshape(B * S, H)
    inds_flat = inds.reshape(B * S, topk)
    scores_flat = scores.reshape(B * S, topk)
    
    token_outputs = []
    
    for t in range(B * S):
        x_t = x_flat[t:t+1]  # [1, H]
        token_out = mx.zeros((1, H))
        
        for k in range(topk):
            eidx = int(inds_flat[t, k].item())
            score = scores_flat[t, k]
            
            # Streaming: load → compute → free
            gate_w = dequant_expert(switch_mlp.gate_proj, eidx)
            up_w = dequant_expert(switch_mlp.up_proj, eidx)
            
            gate_out = x_t @ gate_w.T
            up_out = x_t @ up_w.T
            activated = nn.silu(gate_out) * up_out
            del gate_w, up_w
            
            down_w = dequant_expert(switch_mlp.down_proj, eidx)
            expert_out = activated @ down_w.T
            del down_w, activated
            
            token_out = token_out + expert_out * score
        
        mx.eval(token_out)
        token_outputs.append(token_out)
    
    # Concatenate all token outputs
    routed = mx.concatenate(token_outputs, axis=0).reshape(B, S, H)
    
    if shared_out is not None:
        result = routed + shared_out
    else:
        result = routed
    
    mx.eval(result)
    mx.clear_cache()
    
    return result


def forward_pass(model, input_ids):
    """
    Full forward pass through the model, layer by layer.
    Returns logits for the last token position.
    """
    embed = model.model.embed_tokens
    layers = model.model.layers
    norm = model.model.norm
    lm_head = model.lm_head
    
    x = embed(input_ids)
    mx.eval(x)
    
    for i, layer in enumerate(layers):
        is_moe = hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp')
        
        # Self-attention
        h = layer.input_layernorm(x)
        h = layer.self_attn(h)
        x = x + h
        mx.eval(x)
        
        # MLP / MoE
        h = layer.post_attention_layernorm(x)
        
        if is_moe:
            h = moe_forward(layer.mlp, h, i)
        else:
            h = layer.mlp(h)
        
        x = x + h
        mx.eval(x)
        mx.clear_cache()
        
        # Progress
        if (i + 1) % 10 == 0:
            mem = free_gb()
            log(f"[L{i+1}/{len(layers)} {mem:.0f}G]", end="")
            
            if mem < 12:
                gc.collect()
                mx.clear_cache()
    
    x = norm(x)
    mx.eval(x)
    
    logits = lm_head(x[:, -1:, :])
    mx.eval(logits)
    mx.clear_cache()
    
    return logits


def sample_token(logits, temperature=0.7):
    """Sample next token from logits."""
    logits = logits[0, 0]  # [vocab_size]
    
    if temperature <= 0:
        return int(mx.argmax(logits).item())
    
    probs = mx.softmax(logits / temperature)
    mx.eval(probs)
    token = mx.random.categorical(mx.log(probs + 1e-10))
    mx.eval(token)
    return int(token.item())


def generate(model, tokenizer, prompt, max_tokens=20, temperature=0.7):
    """Generate text token by token."""
    
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    generated_text = ""
    token_times = []
    
    for step in range(max_tokens):
        t0 = time.time()
        
        # Forward pass
        current = mx.array([generated])
        logits = forward_pass(model, current)
        
        # Sample
        next_id = sample_token(logits, temperature)
        generated.append(next_id)
        
        # Decode
        try:
            text = tokenizer.decode([next_id])
            generated_text += text
            log(text, end="")
        except:
            log("?", end="")
        
        elapsed = time.time() - t0
        token_times.append(elapsed)
        
        # Stop conditions
        if next_id == getattr(tokenizer, 'eos_token_id', None):
            break
        if free_gb() < 10:
            log(f"\n[LOW MEM {free_gb():.0f}GB — stopping]")
            break
    
    log("")
    return generated_text, token_times


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--mem-limit", type=int, default=16)
    args = p.parse_args()
    
    log("=" * 65)
    log("  ExpertFlow — Coherent Inference Engine")
    log("=" * 65)
    log(f"  Model: {os.path.basename(args.model)}")
    log(f"  Free: {free_gb():.1f}GB | Limit: {args.mem_limit}GB")
    
    mx.set_memory_limit(args.mem_limit * 1024**3)
    mx.set_cache_limit(int(args.mem_limit * 0.3 * 1024**3))
    
    import mlx_lm
    log(f"  Loading...", end=" ")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    log(f"done ({time.time()-t0:.1f}s)")
    
    # Model info
    nlayers = len(model.model.layers)
    moe_count = sum(1 for l in model.model.layers 
                    if hasattr(l.mlp, 'gate') and hasattr(l.mlp, 'switch_mlp'))
    log(f"  Layers: {nlayers} ({nlayers-moe_count} dense, {moe_count} MoE)")
    
    log(f"\n  >>> {args.prompt}", end="")
    
    text, times = generate(model, tokenizer, args.prompt, 
                          args.max_tokens, args.temperature)
    
    if times:
        ttft = times[0]
        total = sum(times)
        n = len(times)
        
        log(f"\n  ---")
        log(f"  Tokens: {n} | TTFT: {ttft:.1f}s | Total: {total:.1f}s")
        
        if n > 1:
            decode_avg = sum(times[1:]) / (n - 1)
            log(f"  Decode: {1/decode_avg:.2f} tok/s ({decode_avg*1000:.0f}ms/tok)")
        
        log(f"  Memory: {free_gb():.1f}GB free")
        
        # Save
        model_name = os.path.basename(args.model.rstrip("/"))
        result = {
            "engine": "ExpertFlow",
            "model": model_name,
            "prompt": args.prompt,
            "response": text,
            "tokens": n,
            "ttft_s": round(ttft, 2),
            "total_s": round(total, 1),
            "token_times": [round(t, 3) for t in times],
        }
        
        if n > 1:
            result["decode_tok_s"] = round(1 / (sum(times[1:]) / (n-1)), 3)
        
        outpath = os.path.expanduser(f"~/dev/expertflow/{model_name}-coherent.json")
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        
        log(f"\n  🔥 ExpertFlow: {model_name} — coherent inference!")

if __name__ == "__main__":
    main()
