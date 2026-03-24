#!/usr/bin/env python3
"""
ExpertFlow — GPU + Dynamic Expert Caching
==========================================
Uses GPU (Metal) for ALL computation.
Keeps a large LRU cache of dequantized expert weights in unified memory.
Pre-loads experts into cache BEFORE computation to avoid NVMe stalls
during Metal command buffer execution.

Key insight: Metal timeout happens because NVMe page faults during
GPU execution. Solution: pre-dequantize into cache (with eval), then
run the matmul chain on cached weights (no page faults).
"""

import os, sys, time, json, subprocess
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
import gc
from collections import OrderedDict

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

CACHE_BUDGET_GB = 60  # Use up to 60GB for expert cache (out of 128GB)
MIN_FREE_GB = 20       # Emergency eviction if free drops below this
EXPERT_WEIGHT_MB = 28  # Approximate size of one dequantized expert weight set

# ═══════════════════════════════════════════════════════════════
# Expert Weight Cache
# ═══════════════════════════════════════════════════════════════

class ExpertCache:
    """LRU cache for dequantized expert weights."""
    
    def __init__(self, budget_gb=CACHE_BUDGET_GB):
        self.cache = OrderedDict()  # key → (gate_w, up_w, down_w)
        self.budget_bytes = int(budget_gb * 1024**3)
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _key(self, layer_idx, expert_idx):
        return (layer_idx, expert_idx)
    
    def get(self, layer_idx, expert_idx):
        """Get cached expert weights. Returns (gate_w, up_w, down_w) or None."""
        k = self._key(layer_idx, expert_idx)
        if k in self.cache:
            self.hits += 1
            self.cache.move_to_end(k)
            return self.cache[k]
        self.misses += 1
        return None
    
    def put(self, layer_idx, expert_idx, weights):
        """Cache expert weights (gate_w, up_w, down_w)."""
        k = self._key(layer_idx, expert_idx)
        est_size = EXPERT_WEIGHT_MB * 1024**2 * 3  # 3 weight matrices
        
        # Evict until we have space
        while self.current_bytes + est_size > self.budget_bytes and self.cache:
            self._evict_oldest()
        
        self.cache[k] = weights
        self.cache.move_to_end(k)
        self.current_bytes += est_size
    
    def _evict_oldest(self):
        """Remove least recently used entry."""
        if self.cache:
            k, v = self.cache.popitem(last=False)
            del v
            self.current_bytes -= EXPERT_WEIGHT_MB * 1024**2 * 3
            self.current_bytes = max(0, self.current_bytes)
            self.evictions += 1
    
    def emergency_evict(self, keep_n=50):
        """Aggressive eviction when memory pressure detected."""
        while len(self.cache) > keep_n:
            self._evict_oldest()
        gc.collect()
        mx.clear_cache()
    
    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / max(total, 1) * 100
        return f"Cache: {len(self.cache)} entries, {self.current_bytes/1e9:.1f}GB, hit rate: {rate:.0f}% ({self.hits}/{total}), evictions: {self.evictions}"


# Global cache
cache = ExpertCache()

# ═══════════════════════════════════════════════════════════════
# Memory monitoring
# ═══════════════════════════════════════════════════════════════

def free_gb():
    """Get free memory in GB."""
    try:
        out = subprocess.check_output(["vm_stat"], timeout=2).decode()
        f = i = 0
        for l in out.split("\n"):
            if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
            elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
        return (f + i) * 16384 / 1e9
    except:
        return 999  # Assume OK if we can't check


# ═══════════════════════════════════════════════════════════════
# Expert Loading
# ═══════════════════════════════════════════════════════════════

def dequant_expert(switch_linear, expert_idx):
    """Dequantize one expert weight matrix."""
    w = switch_linear["weight"][expert_idx]
    s = switch_linear["scales"][expert_idx]
    b = switch_linear.get("biases")
    b = b[expert_idx] if b is not None else None
    return mx.dequantize(w, s, b, group_size=switch_linear.group_size, bits=switch_linear.bits)


def preload_experts(moe_module, layer_idx, expert_indices):
    """
    Pre-load expert weights into cache BEFORE computation.
    This is the key to avoiding Metal timeout:
    - Dequantize triggers NVMe reads
    - mx.eval() forces completion BEFORE we start the GPU matmul chain
    - Result: matmuls run on cached data, no page faults
    """
    mlp = moe_module.switch_mlp
    loaded = []
    
    for eidx in expert_indices:
        cached = cache.get(layer_idx, eidx)
        if cached is not None:
            continue  # Already cached
        
        # Dequantize all 3 projections
        gate_w = dequant_expert(mlp.gate_proj, eidx)
        up_w = dequant_expert(mlp.up_proj, eidx)
        down_w = dequant_expert(mlp.down_proj, eidx)
        
        # Force materialization (triggers NVMe reads)
        mx.eval(gate_w, up_w, down_w)
        
        cache.put(layer_idx, eidx, (gate_w, up_w, down_w))
        loaded.append(eidx)
    
    return loaded


# ═══════════════════════════════════════════════════════════════
# MoE Forward Pass
# ═══════════════════════════════════════════════════════════════

def moe_forward(moe_module, x, layer_idx):
    """
    GPU-accelerated MoE with dynamic expert caching.
    
    1. Router determines expert assignments
    2. Pre-load ALL needed experts into cache (eval → NVMe complete)
    3. Run matmul chain on GPU using cached weights (no page faults)
    """
    B, S, H = x.shape
    
    # Step 1: Router
    inds, scores = moe_module.gate(x)
    mx.eval(inds, scores)
    
    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp
    
    # Step 2: Get unique experts needed for this layer
    inds_flat = inds.reshape(-1)
    mx.eval(inds_flat)
    unique_experts = sorted(set(int(e) for e in inds_flat.tolist()))
    
    # Step 3: Pre-load ALL needed experts (NVMe I/O happens here, not during matmul)
    preload_experts(moe_module, layer_idx, unique_experts)
    
    # Step 4: Shared experts
    shared_out = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared_out = moe_module.shared_experts(x)
        mx.eval(shared_out)
    
    # Step 5: Expert computation (ALL weights already in cache → no page faults)
    x_flat = x.reshape(B * S, H)
    inds_2d = inds.reshape(B * S, topk)
    scores_2d = scores.reshape(B * S, topk)
    
    token_outputs = []
    
    for t in range(B * S):
        x_t = x_flat[t:t+1]  # [1, H]
        token_out = mx.zeros((1, H))
        
        for k in range(topk):
            eidx = int(inds_2d[t, k].item())
            score = scores_2d[t, k]
            
            # Get from cache (guaranteed hit after preload)
            weights = cache.get(layer_idx, eidx)
            if weights is None:
                # Fallback (shouldn't happen)
                gate_w = dequant_expert(mlp.gate_proj, eidx)
                up_w = dequant_expert(mlp.up_proj, eidx)
                down_w = dequant_expert(mlp.down_proj, eidx)
                mx.eval(gate_w, up_w, down_w)
            else:
                gate_w, up_w, down_w = weights
            
            # SwiGLU — GPU matmul with per-expert eval to avoid Metal timeout
            gate_out = x_t @ gate_w.T
            up_out = x_t @ up_w.T
            activated = nn.silu(gate_out) * up_out
            expert_out = activated @ down_w.T
            
            token_out = token_out + expert_out * score
            mx.eval(token_out)  # Eval per expert — keeps Metal command buffer small
        
        # Already eval'd per expert above
        token_outputs.append(token_out)
    
    # Combine
    routed = mx.concatenate(token_outputs, axis=0).reshape(B, S, H)
    
    if shared_out is not None:
        result = routed + shared_out
    else:
        result = routed
    
    mx.eval(result)
    
    # Memory check after each MoE layer
    free = free_gb()
    if free < MIN_FREE_GB + 5:
        cache.emergency_evict(keep_n=30)
    
    return result


# ═══════════════════════════════════════════════════════════════
# Full Inference
# ═══════════════════════════════════════════════════════════════

def generate(model, tokenizer, prompt, max_tokens=20):
    """Generate tokens with real tok/s measurement."""
    
    input_ids = tokenizer.encode(prompt)
    total_layers = len(model.model.layers)
    generated_tokens = []
    layer_times = []
    token_times = []
    
    print(f"  Prompt: {prompt!r} ({len(input_ids)} tokens)")
    print(f"  Layers: {total_layers}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Cache budget: {CACHE_BUDGET_GB}GB")
    print(flush=True)
    
    for step in range(max_tokens):
        t_token_start = time.time()
        
        # Build input
        all_ids = input_ids + [t for t in generated_tokens]
        x = model.model.embed_tokens(mx.array([all_ids]))
        mx.eval(x)
        
        # Forward pass through all layers
        for i, layer in enumerate(model.model.layers):
            t_layer = time.time()
            
            # Memory pressure check every 10 layers
            if i % 10 == 0:
                free = free_gb()
                if free < MIN_FREE_GB:
                    print(f"  ⚠️ Memory pressure: {free:.1f}GB free — emergency eviction", flush=True)
                    cache.emergency_evict(keep_n=20)
            
            # Attention
            h = layer.input_layernorm(x)
            h = layer.self_attn(h)
            x = x + h
            mx.eval(x)
            
            # MLP/MoE
            h = layer.post_attention_layernorm(x)
            mx.eval(h)
            
            is_moe = hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp')
            
            if is_moe:
                h = moe_forward(layer.mlp, h, i)
            else:
                h = layer.mlp(h)
                mx.eval(h)
            
            x = x + h
            mx.eval(x)
            
            layer_time = time.time() - t_layer
            layer_times.append(layer_time)
            
            # Progress
            if i % 10 == 0 or i == total_layers - 1:
                free = free_gb()
                print(f"  [L{i+1}/{total_layers} {free:.0f}G] ", end="", flush=True)
        
        # Norm + logits
        x = model.model.norm(x)
        mx.eval(x)
        logits = model.lm_head(x[:, -1:, :])
        mx.eval(logits)
        
        # Sample (greedy)
        next_id = int(mx.argmax(logits[0, 0]).item())
        generated_tokens.append(next_id)
        
        token_time = time.time() - t_token_start
        token_times.append(token_time)
        
        # Decode and print
        try:
            text = tokenizer.decode([next_id])
            print(f"\n  Token {step+1}: {text!r} ({token_time:.2f}s)", flush=True)
        except:
            print(f"\n  Token {step+1}: ID={next_id} ({token_time:.2f}s)", flush=True)
        
        # Stop on EOS
        eos_ids = [tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else []
        if next_id in eos_ids:
            print("  [EOS]", flush=True)
            break
    
    # Final decode
    try:
        full_output = tokenizer.decode(generated_tokens)
    except:
        full_output = str(generated_tokens)
    
    return {
        "prompt": prompt,
        "output": full_output,
        "tokens_generated": len(generated_tokens),
        "token_times": [round(t, 3) for t in token_times],
        "avg_tok_s": round(len(generated_tokens) / sum(token_times), 2) if token_times else 0,
        "total_time_s": round(sum(token_times), 1),
        "avg_layer_time_s": round(sum(layer_times) / max(len(layer_times), 1), 3),
        "cache_stats": cache.stats(),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--cache-gb", type=float, default=CACHE_BUDGET_GB)
    args = p.parse_args()
    
    cache.budget_bytes = int(args.cache_gb * 1024**3)
    
    print("=" * 60)
    print("  ExpertFlow — GPU + Dynamic Expert Caching")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free memory: {free_gb():.1f}GB")
    print(f"  Cache budget: {args.cache_gb}GB")
    
    # GPU mode — don't force CPU
    mx.set_memory_limit(int(110 * 1024**3))  # Allow up to 110GB
    mx.set_cache_limit(int(args.cache_gb * 1024**3))
    
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"  Loaded in {0:.1f}s")
    print(f"  Layers: {len(model.model.layers)}")
    print(flush=True)
    
    results = generate(model, tokenizer, args.prompt, args.max_tokens)
    
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Output: {results['prompt']}{results['output']}")
    print(f"  Tokens: {results['tokens_generated']}")
    print(f"  Speed: {results['avg_tok_s']} tok/s")
    print(f"  Total: {results['total_time_s']}s")
    print(f"  {results['cache_stats']}")
    
    # Save results
    outfile = os.path.expanduser(f"~/dev/expertflow/result_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")


if __name__ == "__main__":
    main()
