#!/usr/bin/env python3
"""
ExpertFlow: Dynamic Expert Loading Demo for Qwen3-Next-80B-A3B
============================================================
Proof-of-concept demonstration of dynamic expert loading without heavy I/O.
Shows the core concept: load non-experts + cache active experts on-demand.
"""

import os
import time
import json
import glob
import struct
import numpy as np
from collections import OrderedDict

# ── Setup ──────────────────────────────────────────────────────────────────
MODEL_DIR = "/Users/jhammant/.lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit"
OUTPUT_DIR = os.path.expanduser("~/dev/expertflow")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS = 48
NUM_EXPERTS = 512
NUM_ACTIVE = 10
HIDDEN_SIZE = 2048
LRU_CACHE_SIZE = 50

def banner(phase, title):
    print(f"\n{'='*70}")
    print(f"  Phase {phase}: {title}")
    print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Analyze tensor structure (MUST COMPLETE)
# ══════════════════════════════════════════════════════════════════════════
def phase1_analyze():
    banner(1, "Memory-Map & Analyze Tensor Structure")

    shard_files = sorted(glob.glob(f"{MODEL_DIR}/model*.safetensors"))
    print(f"Found {len(shard_files)} safetensors shards")

    # Parse headers to get tensor sizes
    tensor_info = {}  # key -> {size_bytes, is_expert, shape, dtype}
    total_expert_bytes = 0
    total_non_expert_bytes = 0
    expert_count = 0
    non_expert_count = 0

    for shard_idx, fpath in enumerate(shard_files):
        with open(fpath, "rb") as fp:
            header_size = struct.unpack("<Q", fp.read(8))[0]
            header_json = fp.read(header_size)
            metadata = json.loads(header_json)

        print(f"  Analyzing shard {shard_idx + 1}/{len(shard_files)}: {os.path.basename(fpath)}")

        for key, info in metadata.items():
            if key == "__metadata__":
                continue
            
            offsets = info["data_offsets"]
            size_bytes = offsets[1] - offsets[0]
            is_expert = "switch_mlp" in key
            
            tensor_info[key] = {
                "size_bytes": size_bytes,
                "is_expert": is_expert,
                "shape": info["shape"],
                "dtype": info["dtype"],
                "shard": shard_idx
            }

            if is_expert:
                total_expert_bytes += size_bytes
                expert_count += 1
            else:
                total_non_expert_bytes += size_bytes
                non_expert_count += 1

    total_bytes = total_expert_bytes + total_non_expert_bytes

    print(f"\n📊 Tensor Analysis Results:")
    print(f"  Expert tensors:     {expert_count:>5}  ({total_expert_bytes / 1e9:.2f} GB)")
    print(f"  Non-expert tensors: {non_expert_count:>5}  ({total_non_expert_bytes / 1e9:.2f} GB)")
    print(f"  Total tensors:      {expert_count + non_expert_count:>5}  ({total_bytes / 1e9:.2f} GB)")
    print(f"  Expert data fraction: {total_expert_bytes / total_bytes * 100:.1f}%")

    # Show sample tensor shapes
    print(f"\n🔍 Sample Expert Tensor Shapes:")
    expert_samples = [k for k in tensor_info.keys() if "switch_mlp" in k and "layers.0." in k][:6]
    for key in expert_samples:
        info = tensor_info[key]
        print(f"  {key}")
        print(f"    Shape: {info['shape']} | Size: {info['size_bytes'] / 1e6:.1f} MB | Type: {info['dtype']}")

    print(f"\n{'*'*60}")
    print(f"  ✅ Analysis Complete: Would load {total_non_expert_bytes / 1e9:.2f} GB non-expert")
    print(f"     (vs {total_bytes / 1e9:.2f} GB full model)")
    print(f"     Memory saving: {(1 - total_non_expert_bytes / total_bytes) * 100:.1f}%")
    print(f"{'*'*60}")

    return tensor_info, total_expert_bytes, total_non_expert_bytes


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Router simulation (MUST COMPLETE)
# ══════════════════════════════════════════════════════════════════════════
def phase2_router(tensor_info):
    banner(2, "Router Simulation")

    # Find gate weights
    gate_keys = [k for k in tensor_info.keys() if ".mlp.gate." in k and "shared" not in k]
    print(f"Found {len(gate_keys)} gate tensors across {NUM_LAYERS} layers")

    # Show gate tensor structure
    sample_gate = next((k for k in gate_keys if "layers.0." in k and "weight" in k), None)
    if sample_gate:
        gate_info = tensor_info[sample_gate]
        print(f"Sample gate weight: {sample_gate}")
        print(f"  Shape: {gate_info['shape']} | Size: {gate_info['size_bytes'] / 1024:.1f} KB")
        print(f"  This routes {gate_info['shape'][1]} hidden dims to {gate_info['shape'][0]} experts")

    # Simulate expert selection for multiple forward passes
    print(f"\n🎯 Simulating Expert Selection:")
    print(f"  Model: {NUM_LAYERS} layers, {NUM_EXPERTS} experts/layer, {NUM_ACTIVE} active/token")

    np.random.seed(42)
    selected_experts_by_layer = {}
    all_selected = set()

    # Simulate routing with Zipf-like distribution (some experts are "popular")
    for layer_idx in range(NUM_LAYERS):
        # Zipf distribution: expert popularity decreases with index
        expert_probs = 1.0 / (np.arange(NUM_EXPERTS) + 1) ** 0.8
        expert_probs /= expert_probs.sum()
        
        # Add some noise to simulate input variation
        scores = expert_probs + np.random.randn(NUM_EXPERTS) * 0.1
        top_k = np.argsort(scores)[-NUM_ACTIVE:][::-1]
        
        selected_experts_by_layer[layer_idx] = top_k.tolist()
        all_selected.update((layer_idx, int(e)) for e in top_k)

    # Display results for first few layers
    print(f"Selected experts per layer (showing first 8 layers):")
    for layer_idx in range(8):
        experts = selected_experts_by_layer[layer_idx]
        print(f"  Layer {layer_idx:2d}: {experts}")
    print(f"  ... ({NUM_LAYERS - 8} more layers)")

    # Calculate memory requirements
    # Find size of one expert by looking at layer 0
    expert_layer_0_bytes = sum(info["size_bytes"] for key, info in tensor_info.items() 
                              if "switch_mlp" in key and "layers.0." in key)
    per_expert_bytes = expert_layer_0_bytes / NUM_EXPERTS
    active_experts_total = NUM_ACTIVE * NUM_LAYERS
    active_expert_memory = per_expert_bytes * active_experts_total

    print(f"\n📐 Memory Analysis:")
    print(f"  Per-expert size:        {per_expert_bytes / 1024:.1f} KB")
    print(f"  Per-layer ({NUM_ACTIVE} experts): {per_expert_bytes * NUM_ACTIVE / 1e6:.2f} MB") 
    print(f"  All layers ({active_experts_total} experts): {active_expert_memory / 1e9:.3f} GB")
    print(f"  Unique expert pairs:    {len(all_selected):,}")

    print(f"\n{'*'*60}")
    print(f"  ✅ Active experts need {active_expert_memory / 1e9:.3f} GB")
    print(f"     ({NUM_ACTIVE}/{NUM_EXPERTS} = {NUM_ACTIVE/NUM_EXPERTS*100:.1f}% per layer)")
    print(f"{'*'*60}")

    return selected_experts_by_layer, per_expert_bytes


# ══════════════════════════════════════════════════════════════════════════
# Phase 3: LRU Cache Simulation (MUST COMPLETE)
# ══════════════════════════════════════════════════════════════════════════
class ExpertLRUCacheSimulator:
    """Simulates LRU cache for expert loading without actual I/O."""
    
    def __init__(self, capacity, per_expert_bytes):
        self.capacity = capacity
        self.per_expert_bytes = per_expert_bytes
        self.cache = OrderedDict()  # (layer, expert_idx) -> timestamp
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.load_ops = 0
        
    def get_expert(self, layer_idx, expert_idx, timestamp):
        """Simulate loading an expert (layer_idx, expert_idx)."""
        key = (layer_idx, expert_idx)
        
        if key in self.cache:
            # Cache hit - move to end (most recently used)
            self.cache.move_to_end(key)
            self.cache[key] = timestamp
            self.hits += 1
            return "HIT"
        
        # Cache miss - need to load
        self.misses += 1
        self.load_ops += 1
        
        # Evict LRU if at capacity
        while len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.evictions += 1
        
        # Add to cache
        self.cache[key] = timestamp
        return "MISS"
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "load_operations": self.load_ops,
            "total_requests": total,
        }


def phase3_cache_simulation(selected_experts_by_layer, per_expert_bytes):
    banner(3, "Dynamic Expert Loading with LRU Cache")

    cache = ExpertLRUCacheSimulator(LRU_CACHE_SIZE, per_expert_bytes)
    num_steps = 100  # Simulate 100 forward passes

    print(f"🎮 Cache Simulation Configuration:")
    print(f"  LRU cache capacity:     {LRU_CACHE_SIZE} experts")
    print(f"  Simulation steps:       {num_steps} forward passes")
    print(f"  Experts per pass:       {NUM_ACTIVE} per layer × {NUM_LAYERS} layers = {NUM_ACTIVE * NUM_LAYERS}")
    print(f"  Total expert requests:  {num_steps * NUM_ACTIVE * NUM_LAYERS:,}")
    print()

    # Simulate multiple forward passes with different expert selections
    np.random.seed(42)
    t0 = time.time()

    for step in range(num_steps):
        for layer_idx in range(NUM_LAYERS):
            # Use some variation in expert selection (realistic inference)
            if step % 10 == 0:
                # Every 10 steps, use completely random selection (new input type)
                selected = np.random.choice(NUM_EXPERTS, size=NUM_ACTIVE, replace=False)
            else:
                # Usually use similar experts with some variation
                base_experts = selected_experts_by_layer[layer_idx]
                # Keep 70% of experts, replace 30% randomly
                keep_count = int(NUM_ACTIVE * 0.7)
                new_count = NUM_ACTIVE - keep_count
                
                kept = base_experts[:keep_count]
                new_ones = np.random.choice([e for e in range(NUM_EXPERTS) if e not in base_experts], 
                                          size=new_count, replace=False)
                selected = kept + new_ones.tolist()

            # Load each selected expert through the cache
            for expert_idx in selected:
                cache.get_expert(layer_idx, int(expert_idx), step)

        # Progress report
        if (step + 1) % 25 == 0:
            stats = cache.stats()
            print(f"  Step {step+1:3d}: hit_rate={stats['hit_rate']:.1f}% "
                  f"cache={stats['cache_size']}/{stats['capacity']} "
                  f"evictions={stats['evictions']:,}")

    elapsed = time.time() - t0
    final_stats = cache.stats()

    # Calculate memory usage
    cache_memory_bytes = LRU_CACHE_SIZE * per_expert_bytes

    print(f"\n{'*'*60}")
    print(f"  ✅ Cache Simulation Results ({num_steps} forward passes)")
    print(f"     Total requests:      {final_stats['total_requests']:,}")
    print(f"     Cache hits:          {final_stats['hits']:,}")
    print(f"     Cache misses:        {final_stats['misses']:,}")
    print(f"     Hit rate:            {final_stats['hit_rate']:.1f}%")
    print(f"     Evictions:           {final_stats['evictions']:,}")
    print(f"     Load operations:     {final_stats['load_operations']:,}")
    print(f"     Cache memory:        {cache_memory_bytes / 1e6:.1f} MB")
    print(f"     Simulation time:     {elapsed:.3f}s")
    print(f"{'*'*60}")

    return final_stats


# ══════════════════════════════════════════════════════════════════════════
# Phase 4: Results Summary (BEST EFFORT)
# ══════════════════════════════════════════════════════════════════════════
def phase4_summary(tensor_info, expert_bytes, non_expert_bytes, per_expert_bytes, cache_stats):
    banner(4, "Benchmark Summary")

    total_bytes = expert_bytes + non_expert_bytes
    
    # Memory calculations
    active_expert_mem = per_expert_bytes * NUM_ACTIVE * NUM_LAYERS
    dynamic_total = non_expert_bytes + active_expert_mem
    cache_memory = non_expert_bytes + (LRU_CACHE_SIZE * per_expert_bytes)

    # I/O calculations
    expert_load_savings = (NUM_EXPERTS - NUM_ACTIVE) / NUM_EXPERTS

    results = {
        "model": "Qwen3-Next-80B-A3B-Instruct-MLX-4bit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "architecture": {
            "layers": NUM_LAYERS,
            "experts_per_layer": NUM_EXPERTS,
            "active_per_token": NUM_ACTIVE,
            "utilization_pct": round(NUM_ACTIVE / NUM_EXPERTS * 100, 1)
        },
        "memory_breakdown_gb": {
            "full_model": round(total_bytes / 1e9, 2),
            "expert_weights": round(expert_bytes / 1e9, 2),
            "non_expert_weights": round(non_expert_bytes / 1e9, 2),
            "dynamic_loading": round(dynamic_total / 1e9, 3),
            "lru_cache": round(cache_memory / 1e9, 3)
        },
        "savings": {
            "memory_reduction_pct": round((1 - dynamic_total / total_bytes) * 100, 1),
            "lru_cache_reduction_pct": round((1 - cache_memory / total_bytes) * 100, 1),
            "expert_i/o_reduction_pct": round(expert_load_savings * 100, 1)
        },
        "cache_performance": cache_stats,
        "per_expert_size_kb": round(per_expert_bytes / 1024, 1)
    }

    print(f"{'='*60}")
    print(f"  🎯 DYNAMIC EXPERT LOADING RESULTS")
    print(f"{'='*60}")
    
    print(f"\n  📊 Model Analysis:")
    print(f"    Full model size:      {total_bytes / 1e9:.2f} GB")
    print(f"    Expert weights:       {expert_bytes / 1e9:.2f} GB ({expert_bytes/total_bytes*100:.1f}%)")
    print(f"    Non-expert weights:   {non_expert_bytes / 1e9:.2f} GB")
    print(f"    Expert utilization:   {NUM_ACTIVE}/{NUM_EXPERTS} ({NUM_ACTIVE/NUM_EXPERTS*100:.1f}%) per layer")
    
    print(f"\n  💾 Memory Usage:")
    print(f"    Dynamic loading:      {dynamic_total / 1e9:.3f} GB per forward pass")
    print(f"    LRU cache ({LRU_CACHE_SIZE}):      {cache_memory / 1e9:.3f} GB steady state")
    print(f"    Memory reduction:     {results['savings']['memory_reduction_pct']}% (dynamic)")
    print(f"                          {results['savings']['lru_cache_reduction_pct']}% (with cache)")
    
    print(f"\n  🚀 Cache Performance:")
    print(f"    Hit rate:             {cache_stats['hit_rate']:.1f}%")
    print(f"    Load operations:      {cache_stats['load_operations']:,} (vs {cache_stats['total_requests']:,} requests)")
    print(f"    I/O reduction:        {100 - (cache_stats['load_operations']/cache_stats['total_requests']*100):.1f}%")
    
    print(f"\n  ⚡ Key Benefits:")
    print(f"    • Only load {NUM_ACTIVE/NUM_EXPERTS*100:.1f}% of expert weights per token")
    print(f"    • {results['savings']['memory_reduction_pct']}% memory reduction vs full loading")
    print(f"    • {cache_stats['hit_rate']:.1f}% cache hit rate reduces I/O")
    print(f"    • Scales to any number of experts ({NUM_EXPERTS} → 1024+ possible)")
    
    print(f"\n{'='*60}")

    # Save detailed results
    results_path = os.path.join(OUTPUT_DIR, "benchmark-results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Detailed results saved to: {results_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  🚀 ExpertFlow: Dynamic Expert Loading Demo                ║")
    print("║     Qwen3-Next-80B-A3B (512 experts/layer → 10 active)     ║")  
    print("║     Proof-of-Concept Implementation                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        # Phase 1: Analyze model structure (MUST COMPLETE)
        tensor_info, expert_bytes, non_expert_bytes = phase1_analyze()

        # Phase 2: Simulate expert routing (MUST COMPLETE) 
        selected_experts, per_expert_bytes = phase2_router(tensor_info)

        # Phase 3: LRU cache simulation (MUST COMPLETE)
        cache_stats = phase3_cache_simulation(selected_experts, per_expert_bytes)

        # Phase 4: Results summary (BEST EFFORT)
        results = phase4_summary(tensor_info, expert_bytes, non_expert_bytes, per_expert_bytes, cache_stats)

        print(f"\n✅ ALL PHASES COMPLETED SUCCESSFULLY!")
        print(f"\n🎉 Proof of Concept Summary:")
        print(f"   Dynamic expert loading can reduce memory usage by {results['savings']['memory_reduction_pct']}%")
        print(f"   while maintaining {NUM_ACTIVE}/{NUM_EXPERTS} expert utilization per layer.")
        print(f"   LRU caching achieves {cache_stats['hit_rate']:.1f}% hit rate for realistic workloads.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())