#!/usr/bin/env python3
"""
ExpertFlow — CPU + Large Cache (Guaranteed to work)
==================================================
Use CPU to avoid Metal timeout, but keep the large expert cache
for speed. Get ACTUAL token generation working first.
"""

import os, sys, time, json, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
import gc
from collections import OrderedDict

# Force CPU to avoid Metal timeout
mx.set_default_device(mx.cpu)

# Large cache for speed
CACHE_BUDGET_GB = 50

class ExpertCache:
    def __init__(self, budget_gb=CACHE_BUDGET_GB):
        self.cache = OrderedDict()
        self.budget_bytes = int(budget_gb * 1024**3)
        self.current_bytes = 0
        self.hits = self.misses = self.evictions = 0
    
    def get(self, layer_idx, expert_idx):
        k = (layer_idx, expert_idx)
        if k in self.cache:
            self.hits += 1
            self.cache.move_to_end(k)
            return self.cache[k]
        self.misses += 1
        return None
    
    def put(self, layer_idx, expert_idx, weights):
        k = (layer_idx, expert_idx)
        est_size = 28 * 1024**2 * 3  # 3 weight matrices
        
        while self.current_bytes + est_size > self.budget_bytes and self.cache:
            old_k, old_v = self.cache.popitem(last=False)
            del old_v
            self.current_bytes -= est_size
            self.evictions += 1
        
        self.cache[k] = weights
        self.current_bytes += est_size
    
    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / max(total, 1) * 100
        return f"{len(self.cache)} entries, {self.current_bytes/1e9:.1f}GB, {rate:.0f}% hit rate"

cache = ExpertCache()

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

def dequant_expert(switch_linear, expert_idx):
    w = switch_linear["weight"][expert_idx]
    s = switch_linear["scales"][expert_idx]
    b = switch_linear.get("biases")
    b = b[expert_idx] if b is not None else None
    return mx.dequantize(w, s, b, group_size=switch_linear.group_size, bits=switch_linear.bits)

def moe_forward(moe_module, x, layer_idx):
    B, S, H = x.shape
    
    # Router
    inds, scores = moe_module.gate(x)
    mx.eval(inds, scores)
    
    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp
    
    # Shared experts
    shared_out = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared_out = moe_module.shared_experts(x)
        mx.eval(shared_out)
    
    # Process tokens
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
            
            # Try cache first
            weights = cache.get(layer_idx, eidx)
            if weights is None:
                gate_w = dequant_expert(mlp.gate_proj, eidx)
                up_w = dequant_expert(mlp.up_proj, eidx)
                down_w = dequant_expert(mlp.down_proj, eidx)
                mx.eval(gate_w, up_w, down_w)
                cache.put(layer_idx, eidx, (gate_w, up_w, down_w))
            else:
                gate_w, up_w, down_w = weights
            
            # SwiGLU
            gate_out = x_t @ gate_w.T
            up_out = x_t @ up_w.T
            activated = nn.silu(gate_out) * up_out
            expert_out = activated @ down_w.T
            
            token_out = token_out + expert_out * score
        
        mx.eval(token_out)
        token_outputs.append(token_out)
    
    routed = mx.concatenate(token_outputs, axis=0).reshape(B, S, H)
    
    if shared_out is not None:
        result = routed + shared_out
    else:
        result = routed
    
    mx.eval(result)
    return result

def generate(model, tokenizer, prompt, max_tokens=20):
    input_ids = tokenizer.encode(prompt)
    total_layers = len(model.model.layers)
    generated_tokens = []
    token_times = []
    
    print(f"Prompt: {prompt!r} → {len(input_ids)} tokens")
    print(f"Layers: {total_layers}")
    print(f"Cache: {CACHE_BUDGET_GB}GB budget")
    print(flush=True)
    
    for step in range(max_tokens):
        t_start = time.time()
        
        # Build input
        all_ids = input_ids + generated_tokens
        x = model.model.embed_tokens(mx.array([all_ids]))
        mx.eval(x)
        
        # Forward through all layers
        for i, layer in enumerate(model.model.layers):
            # Attention
            h = layer.input_layernorm(x)
            h = layer.self_attn(h)
            x = x + h
            mx.eval(x)
            
            # MLP/MoE
            h = layer.post_attention_layernorm(x)
            
            is_moe = hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp')
            
            if is_moe:
                h = moe_forward(layer.mlp, h, i)
            else:
                h = layer.mlp(h)
                mx.eval(h)
            
            x = x + h
            mx.eval(x)
            
            # Progress every 20 layers
            if (i + 1) % 20 == 0 or i == total_layers - 1:
                print(f"[L{i+1}/{total_layers}]", end=" ", flush=True)
        
        # Generate next token
        x = model.model.norm(x)
        logits = model.lm_head(x[:, -1:, :])
        mx.eval(logits)
        
        next_id = int(mx.argmax(logits[0, 0]).item())
        generated_tokens.append(next_id)
        
        token_time = time.time() - t_start
        token_times.append(token_time)
        
        # Decode
        try:
            text = tokenizer.decode([next_id])
            print(f"\nToken {step+1}: {text!r} ({token_time:.2f}s)")
        except:
            print(f"\nToken {step+1}: ID={next_id} ({token_time:.2f}s)")
        
        # Stop on EOS
        if hasattr(tokenizer, 'eos_token_id') and next_id == tokenizer.eos_token_id:
            break
    
    # Final output
    try:
        full_output = tokenizer.decode(generated_tokens)
    except:
        full_output = str(generated_tokens)
    
    return {
        "prompt": prompt,
        "output": full_output,
        "tokens_generated": len(generated_tokens),
        "avg_tok_s": round(len(generated_tokens) / sum(token_times), 2) if token_times else 0,
        "total_time_s": round(sum(token_times), 1),
        "token_times": [round(t, 2) for t in token_times],
        "cache_stats": cache.stats(),
    }

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    args = p.parse_args()
    
    print("=" * 50)
    print("ExpertFlow — CPU + Large Cache")
    print("=" * 50)
    print(f"Model: {os.path.basename(args.model)}")
    print(f"Free: {free_gb():.1f}GB")
    
    mx.set_memory_limit(int(100 * 1024**3))
    
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    
    results = generate(model, tokenizer, args.prompt, args.max_tokens)
    
    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"Full: {results['prompt']}{results['output']}")
    print(f"Speed: {results['avg_tok_s']} tok/s")
    print(f"Tokens: {results['tokens_generated']}")
    print(f"Time: {results['total_time_s']}s")
    print(f"Cache: {results['cache_stats']}")
    
    # Save
    outfile = os.path.expanduser(f"~/dev/expertflow/cpu_result_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()