#!/usr/bin/env python3
"""
ExpertFlow v2 — Smart Caching + Multi-Model Support
===================================================
Optimized dynamic expert streaming with:
  • Smart LRU cache with usage-pattern eviction
  • Multi-model support (DeepSeek, GLM-4.5, MiniMax-M2)
  • Streaming expert loading (load → compute → evict immediately)
  • Memory pressure monitoring with adaptive throttling
  • Batch processing optimization for GPU efficiency
"""

import os, sys, time, json, subprocess, gc, threading
from collections import OrderedDict, defaultdict
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExpertFlowConfig:
    max_memory_gb: int = 20           # GPU memory limit
    cache_budget_gb: int = 4          # Expert cache budget  
    min_free_gb: int = 20             # Emergency eviction threshold
    max_cache_entries: int = 128      # Max cached experts
    streaming_mode: bool = True       # Load → compute → evict immediately
    batch_size: int = 1               # Token batch size for MoE
    memory_check_interval: int = 5    # Check memory every N layers

# ═══════════════════════════════════════════════════════════════════════
# Memory Management
# ═══════════════════════════════════════════════════════════════════════

def get_memory_info():
    """Get system memory info."""
    out = subprocess.check_output(["vm_stat"]).decode()
    f = i = 0
    for l in out.split("\n"):
        if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
        elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
    free_gb = (f + i) * 16384 / 1e9
    
    # Get swap usage
    swap_out = subprocess.check_output(["sysctl", "vm.swapusage"]).decode()
    swap_used = 0
    if "used = " in swap_out:
        swap_str = swap_out.split("used = ")[1].split()[0]
        if "M" in swap_str:
            swap_used = float(swap_str.replace("M", "")) / 1024
        elif "G" in swap_str:
            swap_used = float(swap_str.replace("G", ""))
    
    return {"free_gb": free_gb, "swap_gb": swap_used}

def log(msg, end="\n"):
    sys.stdout.write(msg + end)
    sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════
# Smart Expert Cache
# ═══════════════════════════════════════════════════════════════════════

class SmartExpertCache:
    """
    Advanced LRU cache with usage pattern analysis and smart eviction.
    
    Features:
    • Layer-aware priority (recent layers preferred)
    • Expert popularity tracking (frequently used experts cached longer)
    • Size-based eviction (evict largest unused experts first)
    • Memory pressure adaptation (aggressive eviction when memory is low)
    • Usage statistics for cache optimization
    """
    
    def __init__(self, config: ExpertFlowConfig):
        self.config = config
        self.cache = OrderedDict()  # (layer, proj, expert) → (matrix, size, access_count)
        self.current_bytes = 0
        self.stats = {
            "hits": 0, "misses": 0, "evictions": 0, "loads": 0,
            "total_load_time": 0.0, "layer_stats": defaultdict(int)
        }
        self.expert_popularity = defaultdict(int)  # expert_id → access_count
        self.layer_recency = {}  # layer → last_access_time
    
    def _generate_key(self, layer: int, proj: str, expert: int):
        return (layer, proj, expert)
    
    def get(self, layer: int, proj: str, expert: int) -> Optional[mx.array]:
        """Get expert weights with smart priority boosting."""
        key = self._generate_key(layer, proj, expert)
        
        if key in self.cache:
            # Move to end and update stats
            matrix, size, access_count = self.cache[key]
            self.cache[key] = (matrix, size, access_count + 1)
            self.cache.move_to_end(key)
            
            self.stats["hits"] += 1
            self.expert_popularity[expert] += 1
            self.layer_recency[layer] = time.time()
            self.stats["layer_stats"][layer] += 1
            
            return matrix
        
        self.stats["misses"] += 1
        return None
    
    def put(self, layer: int, proj: str, expert: int, matrix: mx.array):
        """Cache expert with smart eviction policy."""
        if self.config.streaming_mode:
            return  # Don't cache in streaming mode
        
        key = self._generate_key(layer, proj, expert)
        size = matrix.nbytes
        
        # Smart eviction based on memory pressure and usage patterns
        while (self.current_bytes + size > self.config.cache_budget_gb * 1024**3 
               or len(self.cache) >= self.config.max_cache_entries):
            if not self.cache:
                break
            
            evicted_key = self._select_eviction_candidate()
            if evicted_key:
                evicted_matrix, evicted_size, _ = self.cache.pop(evicted_key)
                self.current_bytes -= evicted_size
                self.stats["evictions"] += 1
                del evicted_matrix
            else:
                break
        
        self.cache[key] = (matrix, size, 1)
        self.current_bytes += size
        self.stats["loads"] += 1
        self.layer_recency[layer] = time.time()
    
    def _select_eviction_candidate(self):
        """Smart eviction candidate selection."""
        if not self.cache:
            return None
        
        current_time = time.time()
        candidates = []
        
        for key, (matrix, size, access_count) in self.cache.items():
            layer, proj, expert = key
            
            # Scoring: lower score = better eviction candidate
            layer_recency_score = current_time - self.layer_recency.get(layer, 0)
            popularity_score = 1.0 / (self.expert_popularity[expert] + 1)
            size_score = size / 1e6  # MB
            access_score = 1.0 / (access_count + 1)
            
            total_score = layer_recency_score * 0.4 + popularity_score * 0.3 + size_score * 0.2 + access_score * 0.1
            candidates.append((total_score, key))
        
        # Return the candidate with the highest eviction score
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    def emergency_evict(self, keep_layers: List[int] = None):
        """Emergency eviction with layer preservation."""
        if keep_layers is None:
            keep_layers = []
        
        keys_to_remove = []
        for key in list(self.cache.keys()):
            layer, _, _ = key
            if layer not in keep_layers:
                keys_to_remove.append(key)
        
        evicted_count = 0
        for key in keys_to_remove:
            matrix, size, _ = self.cache.pop(key)
            self.current_bytes -= size
            evicted_count += 1
            del matrix
        
        gc.collect()
        return evicted_count
    
    def get_stats(self):
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests * 100 if total_requests > 0 else 0
        
        return {
            "entries": len(self.cache),
            "memory_mb": round(self.current_bytes / 1e6, 1),
            "hit_rate": f"{hit_rate:.1f}%",
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "avg_load_time_ms": round(self.stats["total_load_time"] / max(self.stats["loads"], 1) * 1000, 1),
        }

# ═══════════════════════════════════════════════════════════════════════
# Model Architecture Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_moe_architecture(model):
    """Detect MoE architecture type and parameters."""
    config = model.config if hasattr(model, 'config') else None
    
    # Check first MoE layer to determine architecture
    moe_layer = None
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
            moe_layer = layer.mlp
            break
    
    if not moe_layer:
        return {"type": "dense", "experts": 0, "topk": 0}
    
    # Analyze the MoE structure
    gate = moe_layer.gate
    switch_mlp = moe_layer.switch_mlp
    
    arch_info = {
        "type": "moe",
        "topk": getattr(gate, 'top_k', 8),
        "experts": getattr(gate, 'n_routed_experts', 256),
        "has_shared": hasattr(moe_layer, 'shared_experts') and moe_layer.shared_experts is not None,
    }
    
    # Detect specific architectures
    if hasattr(config, 'model_type'):
        if 'deepseek' in config.model_type.lower():
            arch_info["variant"] = "deepseek_v3"
            arch_info["scaling_factor"] = getattr(config, 'routed_scaling_factor', 1.0)
        elif 'glm' in config.model_type.lower():
            arch_info["variant"] = "glm"
        elif 'minimax' in config.model_type.lower():
            arch_info["variant"] = "minimax"
        else:
            arch_info["variant"] = "generic"
    else:
        arch_info["variant"] = "generic"
    
    return arch_info

# ═══════════════════════════════════════════════════════════════════════
# Streaming Expert Loader
# ═══════════════════════════════════════════════════════════════════════

def load_expert_streaming(switch_linear, expert_idx: int, cache: SmartExpertCache, 
                         layer_idx: int, proj: str) -> mx.array:
    """
    Load expert weights in streaming mode: load → use → evict.
    """
    # Check cache first
    cached = cache.get(layer_idx, proj, expert_idx)
    if cached is not None:
        return cached
    
    # Load and dequantize
    t0 = time.time()
    w = switch_linear["weight"][expert_idx]
    s = switch_linear["scales"][expert_idx]
    b = switch_linear.get("biases")
    b = b[expert_idx] if b is not None else None
    
    w_dequant = mx.dequantize(w, s, b,
                             group_size=switch_linear.group_size,
                             bits=switch_linear.bits)
    mx.eval(w_dequant)
    
    load_time = time.time() - t0
    cache.stats["total_load_time"] += load_time
    
    # Cache if not in streaming mode
    if not cache.config.streaming_mode:
        cache.put(layer_idx, proj, expert_idx, w_dequant)
    
    return w_dequant

# ═══════════════════════════════════════════════════════════════════════
# Optimized MoE Forward Pass
# ═══════════════════════════════════════════════════════════════════════

def moe_forward_optimized(moe_module, x: mx.array, layer_idx: int, 
                         cache: SmartExpertCache, arch_info: Dict[str, Any]) -> mx.array:
    """
    Optimized MoE forward pass with streaming expert loading.
    """
    # Router
    gate = moe_module.gate
    inds, scores = gate(x)
    mx.eval(inds, scores)
    
    B, S, hidden = x.shape
    topk = inds.shape[-1]
    
    # Apply scaling factor for DeepSeek
    if arch_info.get("variant") == "deepseek_v3":
        scaling_factor = arch_info.get("scaling_factor", 1.0)
        scores = scores * scaling_factor
    
    # Shared experts
    shared_out = None
    if arch_info["has_shared"]:
        shared_out = moe_module.shared_experts(x)
        mx.eval(shared_out)
    
    # Expert routing with streaming
    switch_mlp = moe_module.switch_mlp
    x_flat = x.reshape(B * S, hidden)
    inds_flat = inds.reshape(B * S, topk)
    scores_flat = scores.reshape(B * S, topk)
    
    # Get unique experts
    unique_experts = sorted(set(inds_flat.reshape(-1).tolist()))
    
    # Process in streaming mode: minimal batching for memory efficiency
    output = mx.zeros_like(x_flat)
    
    # Process tokens and collect outputs
    token_outputs = []
    
    for t in range(B * S):
        x_t = x_flat[t:t+1]  # [1, hidden]
        token_inds = inds_flat[t]  # [topk]
        token_scores = scores_flat[t:t+1]  # [1, topk]
        
        token_output = mx.zeros((1, hidden))
        
        for k in range(topk):
            expert_idx = int(token_inds[k].item())
            score = token_scores[0, k:k+1]  # [1]
            
            # Load expert weights (streaming)
            gate_w = load_expert_streaming(switch_mlp.gate_proj, expert_idx, 
                                         cache, layer_idx, "gate")
            up_w = load_expert_streaming(switch_mlp.up_proj, expert_idx, 
                                       cache, layer_idx, "up")
            down_w = load_expert_streaming(switch_mlp.down_proj, expert_idx, 
                                         cache, layer_idx, "down")
            
            # SwiGLU
            gate_out = x_t @ gate_w.T  # [1, intermediate]
            up_out = x_t @ up_w.T      # [1, intermediate]
            activated = nn.silu(gate_out) * up_out
            expert_out = activated @ down_w.T  # [1, hidden]
            
            # Accumulate with score
            token_output = token_output + expert_out * score
            
            # In streaming mode, immediately clear expert weights
            if cache.config.streaming_mode:
                del gate_w, up_w, down_w
                mx.clear_cache()
        
        token_outputs.append(token_output)
        
        # Eval periodically
        if (t + 1) % cache.config.batch_size == 0 or t == B * S - 1:
            mx.eval(*token_outputs[-cache.config.batch_size:])
    
    # Concatenate all outputs
    output = mx.concatenate(token_outputs, axis=0)
    
    routed_output = output.reshape(B, S, hidden)
    
    # Combine with shared experts
    if shared_out is not None:
        routed_output = routed_output + shared_out
    
    mx.eval(routed_output)
    return routed_output

# ═══════════════════════════════════════════════════════════════════════
# Adaptive Generation Loop
# ═══════════════════════════════════════════════════════════════════════

def generate_adaptive(model, tokenizer, prompt: str, config: ExpertFlowConfig,
                     max_tokens: int = 20) -> Dict[str, Any]:
    """
    Adaptive generation with memory pressure monitoring.
    """
    cache = SmartExpertCache(config)
    arch_info = detect_moe_architecture(model)
    
    log(f"  Architecture: {arch_info}")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    
    # Model components
    embed = model.model.embed_tokens
    layers = model.model.layers
    norm = model.model.norm
    lm_head = model.lm_head
    
    token_times = []
    generated_text = ""
    
    for step in range(max_tokens):
        t0 = time.time()
        
        # Memory check
        mem_info = get_memory_info()
        if mem_info["free_gb"] < config.min_free_gb or mem_info["swap_gb"] > 2:
            log(f"  ⚠️ Memory pressure: {mem_info['free_gb']:.1f}GB free, {mem_info['swap_gb']:.1f}GB swap")
            cache.emergency_evict()
            mx.clear_cache()
            gc.collect()
        
        # Forward pass
        current = mx.array([generated])
        x = embed(current)
        mx.eval(x)
        
        # Process layers
        for i, layer in enumerate(layers):
            # Attention
            attn_input = layer.input_layernorm(x)
            mx.eval(attn_input)
            
            attn_out = layer.self_attn(attn_input)
            mx.eval(attn_out)
            x = x + attn_out
            mx.eval(x)
            
            # MLP/MoE
            mlp_input = layer.post_attention_layernorm(x)
            mx.eval(mlp_input)
            
            if arch_info["type"] == "moe" and hasattr(layer.mlp, 'gate'):
                mlp_out = moe_forward_optimized(layer.mlp, mlp_input, i, cache, arch_info)
            else:
                mlp_out = layer.mlp(mlp_input)
                mx.eval(mlp_out)
            
            x = x + mlp_out
            mx.eval(x)
            
            # Periodic memory management
            if (i + 1) % config.memory_check_interval == 0:
                mx.clear_cache()
                if (i + 1) % 15 == 0:
                    mem = get_memory_info()
                    log(f"[L{i+1}/{len(layers)} {mem['free_gb']:.0f}GB]", end="")
        
        # Final norm + lm_head
        x = norm(x)
        mx.eval(x)
        logits = lm_head(x[:, -1:, :])
        mx.eval(logits)
        
        # Sample (simple argmax for now)
        next_token = int(mx.argmax(logits[0, 0]).item())
        generated.append(next_token)
        
        # Decode
        try:
            token_text = tokenizer.decode([next_token])
            generated_text += token_text
            log(token_text, end="")
        except:
            log("?", end="")
        
        elapsed = time.time() - t0
        token_times.append(elapsed)
        
        # Check EOS or memory emergency
        if next_token == tokenizer.eos_token_id:
            break
        
        mem_info = get_memory_info()
        if mem_info["free_gb"] < 10:
            log(f"\n  🚨 Emergency stop: {mem_info['free_gb']:.1f}GB free")
            break
    
    log("")  # newline
    
    return {
        "text": generated_text,
        "tokens": len(token_times),
        "token_times": token_times,
        "cache_stats": cache.get_stats(),
        "arch_info": arch_info,
    }

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ExpertFlow v2")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=15)
    parser.add_argument("--memory-limit", type=int, default=20)
    parser.add_argument("--cache-budget", type=int, default=3)
    parser.add_argument("--streaming", action="store_true", default=True,
                       help="Enable streaming mode (load→use→evict)")
    args = parser.parse_args()
    
    config = ExpertFlowConfig(
        max_memory_gb=args.memory_limit,
        cache_budget_gb=args.cache_budget,
        streaming_mode=args.streaming,
        batch_size=1,
    )
    
    log("=" * 70)
    log("  ExpertFlow v2 — Smart Caching + Multi-Model Support")
    log("=" * 70)
    log(f"  Model: {args.model}")
    log(f"  Config: {config}")
    
    mem_info = get_memory_info()
    log(f"  Memory: {mem_info['free_gb']:.1f}GB free, {mem_info['swap_gb']:.1f}GB swap")
    
    # Configure MLX
    mx.set_memory_limit(config.max_memory_gb * 1024**3)
    mx.set_cache_limit(int(config.cache_budget_gb * 0.5 * 1024**3))
    
    # Load model
    import mlx_lm
    log(f"\n  Loading model...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    log(f"  Loaded in {time.time()-t0:.1f}s")
    
    # Generate
    log(f"\n  Prompt: {args.prompt!r}")
    log(f"  Output: ", end="")
    
    result = generate_adaptive(model, tokenizer, args.prompt, config, args.max_tokens)
    
    if result:
        log(f"\n\n  Results:")
        log(f"  Text: {result['text']!r}")
        log(f"  Tokens: {result['tokens']}")
        if result['tokens'] > 0:
            total_time = sum(result['token_times'])
            avg_time = total_time / result['tokens']
            log(f"  Speed: {1.0/avg_time:.2f} tok/s ({avg_time*1000:.0f}ms/tok)")
        log(f"  Cache: {json.dumps(result['cache_stats'])}")
        log(f"  Architecture: {result['arch_info']['variant']} ({result['arch_info']['experts']} experts, top-{result['arch_info']['topk']})")
        
        # Save results
        model_name = os.path.basename(args.model.rstrip("/"))
        outfile = os.path.expanduser(f"~/dev/expertflow/{model_name}-v2-result.json")
        with open(outfile, "w") as f:
            json.dump(result, f, indent=2)
        log(f"  Saved: {outfile}")
        
        log(f"\n  🚀 ExpertFlow v2: {model_name} inference complete!")

if __name__ == "__main__":
    main()