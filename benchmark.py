#!/usr/bin/env python3
"""
ExpertFlow Benchmark: Dynamic vs Full Loading
=============================================
Side-by-side comparison on Apple M5 Max (128GB)
"""

import os, sys, time, json, glob, struct, gc, resource
import numpy as np
import mlx.core as mx
from collections import OrderedDict

MODEL_DIR = "/Users/jhammant/.lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit"
NUM_LAYERS = 48
NUM_EXPERTS = 512
NUM_ACTIVE = 10
RESULTS = {}

def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)

def banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# Parse model structure
banner("MODEL ANALYSIS")
shard_files = sorted(glob.glob(f"{MODEL_DIR}/model*.safetensors"))
print(f"Shards: {len(shard_files)}")

tensor_info = {}
for fpath in shard_files:
    with open(fpath, "rb") as fp:
        header_size = struct.unpack("<Q", fp.read(8))[0]
        header_json = fp.read(header_size)
        metadata = json.loads(header_json)
    for key, info in metadata.items():
        if key == "__metadata__": continue
        offsets = info["data_offsets"]
        tensor_info[key] = {"size": offsets[1] - offsets[0], "is_expert": "switch_mlp" in key}

total_bytes = sum(v["size"] for v in tensor_info.values())
expert_bytes = sum(v["size"] for v in tensor_info.values() if v["is_expert"])
backbone_bytes = total_bytes - expert_bytes
print(f"Total model: {total_bytes/1e9:.2f} GB")
print(f"Expert weights: {expert_bytes/1e9:.2f} GB ({expert_bytes/total_bytes*100:.1f}%)")
print(f"Backbone weights: {backbone_bytes/1e9:.2f} GB ({backbone_bytes/total_bytes*100:.1f}%)")


# BENCHMARK 1: FULL MODEL LOAD
banner("BENCHMARK 1: FULL MODEL LOAD (all 44.8 GB)")
gc.collect()
rss_before = get_rss_mb()
print(f"RSS before: {rss_before:.0f} MB")

t0 = time.time()
full_weights = {}
for i, fpath in enumerate(shard_files):
    st = time.time()
    shard = mx.load(fpath)
    for k, v in shard.items():
        mx.eval(v)
    full_weights.update(shard)
    elapsed = time.time() - st
    print(f"  Shard {i+1}/{len(shard_files)}: {len(shard)} tensors in {elapsed:.1f}s")

full_load_time = time.time() - t0
rss_after_full = get_rss_mb()
full_mem = rss_after_full - rss_before

print(f"\n  Full Load: {full_load_time:.1f}s | Memory: {full_mem/1024:.1f} GB")

# Simulate 100 expert accesses (in-memory)
print(f"  Running 100 inference steps (all in memory)...")
np.random.seed(42)
t0 = time.time()
for step in range(100):
    for layer in range(NUM_LAYERS):
        experts = np.random.randint(0, NUM_EXPERTS, NUM_ACTIVE)
        for exp_id in experts:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                for suffix in ["weight", "scales", "biases"]:
                    key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"
                    if key in full_weights:
                        _ = full_weights[key]
full_access_time = time.time() - t0
print(f"  100 steps: {full_access_time:.3f}s ({full_access_time/100*1000:.1f}ms/step)")

RESULTS["full_load"] = {
    "load_time_s": round(full_load_time, 1),
    "memory_gb": round(full_mem/1024, 1),
    "access_per_step_ms": round(full_access_time/100*1000, 1),
    "tensors": len(full_weights)
}

del full_weights
gc.collect()
mx.metal.clear_cache()
time.sleep(3)


# BENCHMARK 2: DYNAMIC LOADING
banner("BENCHMARK 2: DYNAMIC LOADING (ExpertFlow)")
gc.collect()
rss_before_dyn = get_rss_mb()
print(f"RSS before: {rss_before_dyn:.0f} MB")

# Memory-map (lazy)
t0 = time.time()
mmap_shards = {}
for fpath in shard_files:
    mmap_shards.update(mx.load(fpath))
mmap_time = time.time() - t0
print(f"  Memory-map: {mmap_time:.3f}s (lazy)")

# Load backbone only
t0 = time.time()
backbone = {}
for key, arr in mmap_shards.items():
    if "switch_mlp" not in key:
        mx.eval(arr)
        backbone[key] = arr
backbone_time = time.time() - t0
rss_after_backbone = get_rss_mb()
backbone_mem = rss_after_backbone - rss_before_dyn
print(f"  Backbone load: {backbone_time:.1f}s | Memory: {backbone_mem/1024:.1f} GB")

# LRU Expert Cache
class ExpertLRU:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get_or_load(self, key, mmap_tensor):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        mx.eval(mmap_tensor)
        self.cache[key] = mmap_tensor
        self.misses += 1
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return mmap_tensor

# Correlated expert selection (realistic)
print(f"  Running 100 inference steps (dynamic + LRU cache=100)...")
cache = ExpertLRU(capacity=100)
np.random.seed(42)

layer_prefs = {}
for layer in range(NUM_LAYERS):
    prefs = np.random.exponential(1.0, NUM_EXPERTS)
    layer_prefs[layer] = prefs / prefs.sum()

t0 = time.time()
for step in range(100):
    for layer in range(NUM_LAYERS):
        experts = np.random.choice(NUM_EXPERTS, NUM_ACTIVE, replace=False, p=layer_prefs[layer])
        for exp_id in experts:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                for suffix in ["weight", "scales", "biases"]:
                    key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"
                    if key in mmap_shards:
                        _ = cache.get_or_load(key, mmap_shards[key])

dynamic_access_time = time.time() - t0
rss_after_dynamic = get_rss_mb()
dynamic_mem = rss_after_dynamic - rss_before_dyn
hit_rate = cache.hits / (cache.hits + cache.misses) * 100

print(f"  100 steps: {dynamic_access_time:.3f}s ({dynamic_access_time/100*1000:.1f}ms/step)")
print(f"  Cache: {cache.hits:,} hits / {cache.misses:,} misses ({hit_rate:.1f}% hit rate)")
print(f"  Memory: {dynamic_mem/1024:.1f} GB")

RESULTS["dynamic_load"] = {
    "mmap_time_s": round(mmap_time, 3),
    "backbone_load_time_s": round(backbone_time, 1),
    "memory_gb": round(dynamic_mem/1024, 1),
    "access_per_step_ms": round(dynamic_access_time/100*1000, 1),
    "cache_hit_rate": round(hit_rate, 1),
    "cache_hits": cache.hits,
    "cache_misses": cache.misses,
    "backbone_tensors": len(backbone)
}


# COMPARISON
banner("HEAD-TO-HEAD COMPARISON")
fl = RESULTS["full_load"]
dl = RESULTS["dynamic_load"]

mem_save = (1 - dl["memory_gb"] / fl["memory_gb"]) * 100 if fl["memory_gb"] > 0 else 0
time_save = (1 - dl["backbone_load_time_s"] / fl["load_time_s"]) * 100

print(f"{'Metric':<35} {'Full Load':>15} {'Dynamic':>15} {'Savings':>10}")
print(f"{'-'*75}")
print(f"{'Load time':<35} {fl['load_time_s']:>13.1f}s {dl['backbone_load_time_s']:>13.1f}s {time_save:>9.0f}%")
print(f"{'Peak memory':<35} {fl['memory_gb']:>12.1f}GB {dl['memory_gb']:>12.1f}GB {mem_save:>9.0f}%")
print(f"{'Access latency (ms/step)':<35} {fl['access_per_step_ms']:>14.1f} {dl['access_per_step_ms']:>14.1f} {'':>10}")
print(f"{'Tensors loaded upfront':<35} {fl['tensors']:>15,} {dl['backbone_tensors']:>15,} {'':>10}")
print(f"{'Cache hit rate':<35} {'N/A':>15} {dl['cache_hit_rate']:>13.1f}% {'':>10}")

print(f"\n{'='*75}")
print(f"  VERDICT: Dynamic loading saves {mem_save:.0f}% memory")
print(f"  {dl['memory_gb']:.1f} GB vs {fl['memory_gb']:.1f} GB — that's {fl['memory_gb']-dl['memory_gb']:.1f} GB freed")
print(f"  Cache hit rate: {dl['cache_hit_rate']:.1f}% (correlated expert selection)")
print(f"{'='*75}")

RESULTS["comparison"] = {
    "memory_savings_pct": round(mem_save, 1),
    "load_time_savings_pct": round(time_save, 1),
    "memory_freed_gb": round(fl["memory_gb"] - dl["memory_gb"], 1)
}

outpath = os.path.expanduser("~/dev/expertflow/benchmark-results.json")
with open(outpath, "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nResults: {outpath}")
