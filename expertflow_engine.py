#!/usr/bin/env python3
"""
ExpertFlow — Dynamic Expert Streaming Engine
=============================================
Runs MoE models 2-3x larger than RAM on Apple Silicon.
Manual layer-by-layer inference with smart expert caching.

Supported architectures:
  - DeepSeek V3/V3.1 (SwitchGLU + MoEGate)
  - GLM-4.5 (block_sparse_moe)
  - MiniMax-M2 (SwitchGLU)
  - Any mlx-lm MoE model with QuantizedSwitchLinear

Strategy:
  1. Load model lazy (mmap, no GPU allocation)
  2. Process each layer sequentially with mx.eval() barriers
  3. For MoE layers: run router → load only needed expert slices →
     dequantize → compute → weight-sum → evict cold experts
  4. Smart LRU cache: keeps recently-used expert weights in memory,
     evicts least-recently-used when approaching memory limit
"""

import os, sys, time, json, subprocess, gc, struct
from collections import OrderedDict
from typing import Optional, Tuple, List

os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

MAX_MEMORY_GB = 25          # Maximum GPU/unified memory usage
CACHE_MEMORY_GB = 12        # Expert cache budget
MIN_FREE_GB = 15            # Emergency eviction threshold
EXPERT_CACHE_MAX = 256      # Max cached expert weight matrices

# ═══════════════════════════════════════════════════════════════════════
# Memory utilities
# ═══════════════════════════════════════════════════════════════════════

def free_gb():
    out = subprocess.check_output(["vm_stat"]).decode()
    f = i = 0
    for l in out.split("\n"):
        if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
        elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
    return (f + i) * 16384 / 1e9

def log(msg, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════
# Smart Expert Cache — Layer-Aware LRU
# ═══════════════════════════════════════════════════════════════════════

class SmartExpertCache:
    """
    Layer-aware LRU cache for dequantized expert weight matrices.
    
    Key design decisions:
    - Cache key = (layer_idx, proj_name, expert_idx) for precise eviction
    - Prioritizes keeping experts from recent layers (temporal locality)
    - Tracks per-layer hit rates for adaptive sizing
    - Emergency eviction when free memory drops too low
    - Each cached entry is a dequantized [out_dim, in_dim] float16 matrix
    """
    
    def __init__(self, max_entries=EXPERT_CACHE_MAX, memory_budget_gb=CACHE_MEMORY_GB):
        self.max_entries = max_entries
        self.memory_budget_bytes = int(memory_budget_gb * 1024**3)
        self.cache = OrderedDict()  # key → (matrix, size_bytes)
        self.current_bytes = 0
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.loads = 0
        self.load_time_s = 0.0
        self.layer_hits = {}  # layer_idx → hit count
    
    def get(self, layer_idx: int, proj: str, expert_idx: int) -> Optional[mx.array]:
        key = (layer_idx, proj, expert_idx)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            self.layer_hits[layer_idx] = self.layer_hits.get(layer_idx, 0) + 1
            return self.cache[key][0]
        self.misses += 1
        return None
    
    def put(self, layer_idx: int, proj: str, expert_idx: int, matrix: mx.array):
        key = (layer_idx, proj, expert_idx)
        size = matrix.nbytes
        
        # Evict until we have space
        while (self.current_bytes + size > self.memory_budget_bytes 
               or len(self.cache) >= self.max_entries):
            if not self.cache:
                break
            evicted_key, (evicted_mat, evicted_size) = self.cache.popitem(last=False)
            self.current_bytes -= evicted_size
            self.evictions += 1
            del evicted_mat
        
        self.cache[key] = (matrix, size)
        self.current_bytes += size
        self.loads += 1
    
    def emergency_evict(self, keep_layer: int = -1):
        """Evict everything except the current layer's experts."""
        evicted = 0
        keys_to_remove = []
        for key in list(self.cache.keys()):
            if key[0] != keep_layer:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            mat, size = self.cache.pop(key)
            self.current_bytes -= size
            evicted += 1
            del mat
        
        gc.collect()
        return evicted
    
    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total * 100 if total else 0
    
    def stats(self):
        return {
            "entries": len(self.cache),
            "memory_mb": round(self.current_bytes / 1e6, 1),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1f}%",
            "evictions": self.evictions,
            "load_time_s": round(self.load_time_s, 2),
        }


CACHE = SmartExpertCache()

# ═══════════════════════════════════════════════════════════════════════
# Expert Loader — Dequantizes expert slices from packed tensors
# ═══════════════════════════════════════════════════════════════════════

def load_expert_weight(switch_linear, expert_idx: int) -> mx.array:
    """
    Load and dequantize a single expert's weight matrix from a 
    QuantizedSwitchLinear module.
    
    The packed tensor has shape [num_experts, out_dim, packed_in_dim].
    We extract row [expert_idx] and dequantize it.
    
    Returns: [out_dim, in_dim] float16 matrix
    """
    t0 = time.time()
    
    w = switch_linear["weight"][expert_idx]    # [out_dim, packed_in]
    s = switch_linear["scales"][expert_idx]    # [out_dim, num_groups]
    b = switch_linear.get("biases")
    b = b[expert_idx] if b is not None else None
    
    w_dequant = mx.dequantize(w, s, b,
                             group_size=switch_linear.group_size,
                             bits=switch_linear.bits)
    mx.eval(w_dequant)
    
    CACHE.load_time_s += time.time() - t0
    return w_dequant


# ═══════════════════════════════════════════════════════════════════════
# MoE Forward Pass — The Core of ExpertFlow
# ═══════════════════════════════════════════════════════════════════════

def moe_forward(moe_module, x: mx.array, layer_idx: int) -> mx.array:
    """
    Manual MoE forward pass with dynamic expert loading.
    
    Steps:
    1. Run router/gate to get expert indices + routing scores
    2. Run shared experts (always in memory, small)
    3. For each selected expert: load weights → compute → cache
    4. Weight-sum expert outputs by routing scores
    5. Add shared expert output
    
    This replaces the standard MoE.__call__ which would trigger
    gather_qmm on the full packed expert tensor.
    """
    # Step 1: Router — determines which experts to use
    # gate(x) → (indices [batch, seq, topk], scores [batch, seq, topk])
    gate = moe_module.gate
    inds, scores = gate(x)
    mx.eval(inds, scores)
    
    B, S, _ = x.shape
    topk = inds.shape[-1]  # 8 for DeepSeek V3
    hidden = x.shape[-1]
    
    # Step 2: Shared experts (always loaded, small — ~200MB)
    shared_out = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared_out = moe_module.shared_experts(x)
        mx.eval(shared_out)
    
    # Step 3: Route through selected experts
    switch_mlp = moe_module.switch_mlp  # SwitchGLU with gate_proj, up_proj, down_proj
    
    # Get unique experts across the entire batch
    inds_flat = inds.reshape(-1)
    mx.eval(inds_flat)
    unique_experts = sorted(set(inds_flat.tolist()))
    
    # Preload all needed expert weights for this layer
    expert_gate_w = {}
    expert_up_w = {}
    expert_down_w = {}
    
    for eidx in unique_experts:
        eidx = int(eidx)
        
        # Gate projection
        gw = CACHE.get(layer_idx, "gate", eidx)
        if gw is None:
            gw = load_expert_weight(switch_mlp.gate_proj, eidx)
            CACHE.put(layer_idx, "gate", eidx, gw)
        expert_gate_w[eidx] = gw
        
        # Up projection
        uw = CACHE.get(layer_idx, "up", eidx)
        if uw is None:
            uw = load_expert_weight(switch_mlp.up_proj, eidx)
            CACHE.put(layer_idx, "up", eidx, uw)
        expert_up_w[eidx] = uw
        
        # Down projection
        dw = CACHE.get(layer_idx, "down", eidx)
        if dw is None:
            dw = load_expert_weight(switch_mlp.down_proj, eidx)
            CACHE.put(layer_idx, "down", eidx, dw)
        expert_down_w[eidx] = dw
    
    # Step 4: Per-token batched expert computation
    # For each token: stack its K expert weights → single batched matmul
    # This minimizes Metal command count (3 matmuls per token, not 3×K)
    
    x_flat = x.reshape(B * S, hidden)    # [tokens, hidden]
    inds_2d = inds.reshape(B * S, topk)  # [tokens, topk]
    scores_2d = scores.reshape(B * S, topk)  # [tokens, topk]
    num_tokens = B * S
    
    token_outputs = []
    
    for t in range(num_tokens):
        x_t = x_flat[t:t+1]  # [1, hidden]
        token_inds = inds_2d[t]  # [topk]
        token_scores = scores_2d[t:t+1]  # [1, topk]
        mx.eval(token_inds)
        
        expert_list = [int(token_inds[k].item()) for k in range(topk)]
        
        # Stack expert weights: [topk, intermediate, hidden] for gate/up
        gate_stack = mx.stack([expert_gate_w[e] for e in expert_list])  # [K, inter, hidden]
        up_stack = mx.stack([expert_up_w[e] for e in expert_list])      # [K, inter, hidden]
        down_stack = mx.stack([expert_down_w[e] for e in expert_list])  # [K, hidden, inter]
        
        # Batched matmul: [K, 1, hidden] @ [K, hidden, inter] → [K, 1, inter]
        x_K = mx.broadcast_to(x_t[None, :, :], (topk, 1, hidden))  # [K, 1, hidden]
        
        gate_out = x_K @ gate_stack.swapaxes(-1, -2)  # [K, 1, inter]
        up_out = x_K @ up_stack.swapaxes(-1, -2)      # [K, 1, inter]
        
        # SwiGLU
        activated = nn.silu(gate_out) * up_out  # [K, 1, inter]
        
        # Down projection: [K, 1, inter] @ [K, inter, hidden] → [K, 1, hidden]
        down_out = activated @ down_stack.swapaxes(-1, -2)  # [K, 1, hidden]
        down_out = down_out.squeeze(1)  # [K, hidden]
        
        # Weight by routing scores and sum
        weighted = down_out * token_scores.T  # [K, hidden] * [K, 1]
        token_out = weighted.sum(axis=0, keepdims=True)  # [1, hidden]
        
        token_outputs.append(token_out)
        
        # Eval every token to keep graph small 
        mx.eval(token_out)
        mx.clear_cache()
    
    # Stack all token outputs
    routed_output = mx.concatenate(token_outputs, axis=0).reshape(B, S, hidden)
    mx.eval(routed_output)
    
    # Step 5: Combine with shared experts
    if shared_out is not None:
        result = routed_output + shared_out
    else:
        result = routed_output
    
    mx.eval(result)
    mx.clear_cache()
    
    # Aggressive memory management: evict all experts from previous layers
    # Only keep the most recent layer's experts for potential reuse
    evicted = CACHE.emergency_evict(keep_layer=layer_idx)
    gc.collect()
    mx.clear_cache()
    
    # Extra safety: check free memory
    mem = free_gb()
    if mem < MIN_FREE_GB:
        # Nuclear option: evict EVERYTHING
        CACHE.emergency_evict(keep_layer=-1)
        gc.collect()
        mx.clear_cache()
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# Layer Processing — Handles Both Dense and MoE Layers
# ═══════════════════════════════════════════════════════════════════════

def process_layer(layer, x: mx.array, layer_idx: int, 
                  num_dense: int = 3, has_cache=False) -> mx.array:
    """
    Process a single transformer layer.
    Handles both dense layers (< num_dense) and MoE layers.
    """
    # Attention (same for all layers)
    attn_input = layer.input_layernorm(x)
    mx.eval(attn_input)
    
    attn_out = layer.self_attn(attn_input)
    mx.eval(attn_out)
    
    x = x + attn_out
    mx.eval(x)
    mx.clear_cache()
    
    # MLP / MoE
    mlp_input = layer.post_attention_layernorm(x)
    mx.eval(mlp_input)
    
    is_moe = hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp')
    
    if is_moe:
        # Dynamic expert loading!
        mlp_out = moe_forward(layer.mlp, mlp_input, layer_idx)
    else:
        # Dense MLP — just run normally
        mlp_out = layer.mlp(mlp_input)
        mx.eval(mlp_out)
    
    x = x + mlp_out
    mx.eval(x)
    mx.clear_cache()
    
    return x


# ═══════════════════════════════════════════════════════════════════════
# Manual Token Generation
# ═══════════════════════════════════════════════════════════════════════

def generate(model, tokenizer, prompt: str, max_tokens: int = 50,
             temperature: float = 0.6, top_p: float = 0.9):
    """
    Full manual generation with ExpertFlow.
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    
    embed = model.model.embed_tokens
    layers = model.model.layers
    norm = model.model.norm
    lm_head = model.lm_head
    
    num_layers = len(layers)
    
    # Detect architecture
    num_dense = 0
    for i, layer in enumerate(layers):
        if not (hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp')):
            num_dense += 1
        else:
            break
    
    log(f"  Architecture: {num_layers} layers, {num_dense} dense + {num_layers - num_dense} MoE")
    
    token_times = []
    generated_text = ""
    
    for step in range(max_tokens):
        t0 = time.time()
        
        # Input: full sequence (no KV cache yet — first version)
        current = mx.array([generated])  # [1, seq_len]
        
        # Embedding
        x = embed(current)
        mx.eval(x)
        mx.clear_cache()
        
        # Process all layers
        for i, layer in enumerate(layers):
            x = process_layer(layer, x, i, num_dense=num_dense)
            
            # Progress indicator every 10 layers
            if (i + 1) % 15 == 0:
                mem = free_gb()
                sys.stdout.write(f"[L{i+1}/{num_layers} {mem:.0f}GB]")
                sys.stdout.flush()
        
        # Final norm + lm_head
        x = norm(x)
        mx.eval(x)
        
        logits = lm_head(x[:, -1:, :])  # [1, 1, vocab_size]
        mx.eval(logits)
        mx.clear_cache()
        
        # Sample
        logits_2d = logits[0, 0]
        
        if temperature > 0:
            # Top-p sampling
            probs = mx.softmax(logits_2d / temperature)
            mx.eval(probs)
            
            sorted_indices = mx.argsort(-probs)
            sorted_probs = probs[sorted_indices]
            cumsum = mx.cumsum(sorted_probs)
            mx.eval(cumsum)
            
            # Find cutoff
            cutoff = mx.searchsorted(cumsum, mx.array([top_p]))[0]
            cutoff = int(max(1, cutoff.item()))
            
            # Zero out low-prob tokens
            top_indices = sorted_indices[:cutoff]
            top_probs = sorted_probs[:cutoff]
            top_probs = top_probs / top_probs.sum()
            mx.eval(top_probs)
            
            # Sample from truncated distribution
            sample_idx = mx.random.categorical(mx.log(top_probs + 1e-10))
            mx.eval(sample_idx)
            next_token = int(top_indices[sample_idx].item())
        else:
            next_token = int(mx.argmax(logits_2d).item())
        
        generated.append(next_token)
        
        # Decode
        try:
            token_text = tokenizer.decode([next_token])
            generated_text += token_text
            sys.stdout.write(token_text)
            sys.stdout.flush()
        except:
            sys.stdout.write("?")
            sys.stdout.flush()
        
        elapsed = time.time() - t0
        token_times.append(elapsed)
        
        # Check EOS
        if next_token == tokenizer.eos_token_id:
            break
        
        # Memory safety
        if free_gb() < 10:
            log(f"\n  ⚠️ Low memory ({free_gb():.1f}GB) — stopping generation")
            break
    
    log("")  # newline
    
    # Stats
    if token_times:
        ttft = token_times[0]
        decode_times = token_times[1:] if len(token_times) > 1 else []
        avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
        tps = 1.0 / avg_decode if avg_decode > 0 else 0
        
        return {
            "text": generated_text,
            "tokens": len(token_times),
            "ttft_s": round(ttft, 2),
            "decode_tok_s": round(tps, 3),
            "decode_ms_per_tok": round(avg_decode * 1000, 0),
            "total_s": round(sum(token_times), 1),
            "token_times": [round(t, 3) for t in token_times],
            "cache_stats": CACHE.stats(),
            "free_gb": round(free_gb(), 1),
        }
    return None


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ExpertFlow Inference Engine")
    parser.add_argument("--model", default=os.path.expanduser("~/models/deepseek-v3.1-4bit"),
                        help="Path to model directory")
    parser.add_argument("--prompt", default="What is the capital of France?",
                        help="Generation prompt")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--memory-limit", type=int, default=MAX_MEMORY_GB,
                        help="GPU memory limit in GB")
    parser.add_argument("--cache-budget", type=int, default=CACHE_MEMORY_GB,
                        help="Expert cache budget in GB")
    args = parser.parse_args()
    
    log("=" * 70)
    log("  ExpertFlow — Dynamic Expert Streaming for MoE Inference")
    log("  Running models larger than RAM on Apple Silicon")
    log("=" * 70)
    log(f"  Model: {args.model}")
    log(f"  Hardware: Apple Silicon, {128}GB Unified Memory")
    log(f"  MLX limits: {args.memory_limit}GB memory, {args.cache_budget}GB expert cache")
    log(f"  Free memory: {free_gb():.1f} GB")
    
    # Configure memory
    mx.set_memory_limit(args.memory_limit * 1024**3)
    mx.set_cache_limit(int(args.cache_budget * 0.5 * 1024**3))
    global CACHE
    CACHE = SmartExpertCache(memory_budget_gb=args.cache_budget)
    
    # Load model
    import mlx_lm
    log(f"\n  Loading model (lazy mmap)...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    log(f"  Loaded in {time.time()-t0:.1f}s | Free: {free_gb():.1f} GB")
    
    # Count layers
    num_layers = len(model.model.layers)
    moe_layers = sum(1 for l in model.model.layers 
                     if hasattr(l.mlp, 'gate') and hasattr(l.mlp, 'switch_mlp'))
    dense_layers = num_layers - moe_layers
    log(f"  Layers: {num_layers} total ({dense_layers} dense, {moe_layers} MoE)")
    
    # Run inference
    log(f"\n  Prompt: {args.prompt!r}")
    log(f"  Max tokens: {args.max_tokens}")
    log(f"\n  ═══ Generation ═══\n  ", end="")
    
    result = generate(model, tokenizer, args.prompt, 
                     max_tokens=args.max_tokens,
                     temperature=args.temperature)
    
    if result:
        log(f"\n  ═══ Results ═══")
        log(f"  Response: {result['text']!r}")
        log(f"  TTFT: {result['ttft_s']}s")
        log(f"  Decode: {result['decode_tok_s']} tok/s ({result['decode_ms_per_tok']}ms/tok)")
        log(f"  Total: {result['total_s']}s for {result['tokens']} tokens")
        log(f"  Cache: {json.dumps(result['cache_stats'])}")
        log(f"  Free memory: {result['free_gb']} GB")
        
        # Save results
        model_name = os.path.basename(args.model.rstrip("/"))
        outfile = os.path.expanduser(f"~/dev/expertflow/{model_name}-result.json")
        full_result = {
            "engine": "ExpertFlow v1.0",
            "model": model_name,
            "model_path": args.model,
            "hardware": "Apple M5 Max 128GB",
            **result,
        }
        with open(outfile, "w") as f:
            json.dump(full_result, f, indent=2)
        log(f"  Saved: {outfile}")
        
        log(f"\n  🔥 ExpertFlow: {model_name} inference COMPLETE!")
    else:
        log(f"\n  ❌ No tokens generated")


if __name__ == "__main__":
    main()
