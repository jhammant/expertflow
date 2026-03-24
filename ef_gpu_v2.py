#!/usr/bin/env python3
"""
ExpertFlow — GPU Inference with CPU Pre-paging
===============================================
KEY INSIGHT: Metal timeout happens because NVMe page faults stall GPU.
FIX: Dequantize expert weights on CPU stream (forces NVMe reads),
then GPU matmuls access data already in unified memory (no faults).

Apple Silicon unified memory means CPU-materialized data is instantly
available to GPU without any copy.
"""

import os, sys, time, json, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
from collections import OrderedDict

# ═══ Config ═══
CACHE_BUDGET_GB = 70   # Large cache in unified memory
MIN_FREE_GB = 20

# ═══ Expert Cache ═══
class ExpertCache:
    def __init__(self, budget_gb):
        self.cache = OrderedDict()
        self.budget = int(budget_gb * 1024**3)
        self.size = 0
        self.hits = self.misses = self.evictions = 0
        self.ENTRY_SIZE = 28 * 1024**2 * 3  # ~84MB per expert (3 projections)
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        while self.size + self.ENTRY_SIZE > self.budget and self.cache:
            _, v = self.cache.popitem(last=False)
            del v
            self.size -= self.ENTRY_SIZE
            self.evictions += 1
        self.cache[key] = value
        self.size += self.ENTRY_SIZE
    
    def evict_to(self, n):
        while len(self.cache) > n:
            _, v = self.cache.popitem(last=False)
            del v
            self.size -= self.ENTRY_SIZE
            self.evictions += 1
        self.size = max(0, self.size)
    
    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / max(total, 1) * 100
        return f"{len(self.cache)} cached, {self.size/1e9:.1f}GB, {rate:.0f}% hit ({self.hits}h/{self.misses}m), {self.evictions} evicted"

cache = ExpertCache(CACHE_BUDGET_GB)

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

# ═══ Core: CPU-prefault + GPU compute ═══

def prefault_experts(mlp, layer_idx, expert_indices):
    """
    Dequantize on CPU stream → eval → data lands in unified memory.
    GPU can then access it without triggering NVMe page faults.
    """
    for eidx in expert_indices:
        key = (layer_idx, eidx)
        if cache.get(key) is not None:
            continue  # Already cached
        
        # Dequantize on CPU stream (triggers NVMe reads)
        with mx.stream(mx.cpu):
            def _dq(proj, idx):
                w = proj["weight"][idx]
                s = proj["scales"][idx]
                b = proj.get("biases")
                b = b[idx] if b is not None else None
                return mx.dequantize(w, s, b, group_size=proj.group_size, bits=proj.bits)
            
            gw = _dq(mlp.gate_proj, eidx)
            uw = _dq(mlp.up_proj, eidx)
            dw = _dq(mlp.down_proj, eidx)
            
            # Force materialization on CPU (NVMe reads complete here)
            mx.eval(gw, uw, dw)
        
        cache.put(key, (gw, uw, dw))


def moe_forward(moe_module, x, layer_idx):
    """MoE with CPU prefault + GPU compute."""
    B, S, H = x.shape
    
    # Router on GPU (fast)
    inds, scores = moe_module.gate(x)
    mx.eval(inds, scores)
    
    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp
    
    # Get unique experts
    unique = sorted(set(int(e) for e in inds.reshape(-1).tolist()))
    
    # CPU prefault: dequantize all needed experts (NVMe reads here)
    prefault_experts(mlp, layer_idx, unique)
    
    # Shared experts
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)
    
    # GPU compute: all weights already in unified memory
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
            
            gw, uw, dw = cache.get((layer_idx, eidx))
            
            # GPU matmul on pre-faulted data (no NVMe stalls)
            g = x_t @ gw.T
            u = x_t @ uw.T
            out = out + (nn.silu(g) * u) @ dw.T * score
        
        mx.eval(out)
        token_outs.append(out)
    
    routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
    result = (routed + shared) if shared is not None else routed
    mx.eval(result)
    
    # Memory management
    free = free_gb()
    if free < MIN_FREE_GB:
        cache.evict_to(30)
        mx.clear_cache()
    
    return result


# ═══ Generation ═══

def generate(model, tokenizer, prompt, max_tokens):
    input_ids = tokenizer.encode(prompt)
    total_layers = len(model.model.layers)
    generated = []
    token_times = []
    
    print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
    print(f"  Model: {total_layers} layers")
    print(f"  Cache: {CACHE_BUDGET_GB}GB", flush=True)
    
    for step in range(max_tokens):
        t0 = time.time()
        
        all_ids = input_ids + generated
        x = model.model.embed_tokens(mx.array([all_ids]))
        mx.eval(x)
        
        for i, layer in enumerate(model.model.layers):
            # Attention (GPU)
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
        
        print(f"\n  ✅ Token {step+1}: {text!r} ({dt:.1f}s | {1/dt:.2f} tok/s)")
        
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
        "avg_tok_s": round(avg, 3),
        "times": [round(t, 2) for t in token_times],
        "total_s": round(sum(token_times), 1),
        "cache": cache.stats(),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--cache-gb", type=float, default=CACHE_BUDGET_GB)
    args = p.parse_args()
    
    cache.budget = int(args.cache_gb * 1024**3)
    
    print("=" * 60)
    print("  ExpertFlow v2 — GPU + CPU Prefault + Dynamic Cache")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free: {free_gb():.1f}GB")
    print(f"  Cache: {args.cache_gb}GB")
    
    # GPU mode with generous limits
    mx.set_memory_limit(int(110 * 1024**3))
    mx.set_cache_limit(int(20 * 1024**3))
    
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    
    results = generate(model, tokenizer, args.prompt, args.max_tokens)
    
    print(f"\n{'=' * 60}")
    print(f"  FINAL: {results['full_text']}")
    print(f"  Speed: {results['avg_tok_s']} tok/s")
    print(f"  Total: {results['total_s']}s for {results['tokens']} tokens")
    print(f"  Cache: {results['cache']}")
    
    outfile = os.path.expanduser(f"~/dev/expertflow/gpu_result_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")

if __name__ == "__main__":
    main()
