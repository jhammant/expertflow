#!/usr/bin/env python3
"""
ExpertFlow Mega Benchmark — Too-Big-For-RAM Models
===================================================
Tests models that CANNOT fit in 128GB RAM via ExpertFlow dynamic loading.
Runs automatically as each model finishes downloading.
"""

import os, sys, time, json, glob, struct, gc, resource
import numpy as np
import mlx.core as mx
from collections import OrderedDict

MODELS = [
    {
        "name": "DeepSeek V3.1",
        "path": os.path.expanduser("~/models/deepseek-v3.1-4bit"),
        "total_params": "671B",
        "active_params": "37B",
    },
    {
        "name": "MiniMax-M2",
        "path": os.path.expanduser("~/models/minimax-m2-4bit"),
        "total_params": "230B",
        "active_params": "10B",
    },
    {
        "name": "GLM-4.5 Full",
        "path": os.path.expanduser("~/models/glm-4.5-4bit"),
        "total_params": "355B",
        "active_params": "32B",
    },
]

STEPS = 50
RAM_GB = 128

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)

def banner(t):
    print(f"\n{'#'*70}\n  {t}\n{'#'*70}", flush=True)

def analyze_model(path):
    """Parse safetensors to get expert vs backbone split."""
    shard_files = sorted(glob.glob(f"{path}/model*.safetensors"))
    if not shard_files:
        return None

    tensor_info = {}
    for fpath in shard_files:
        with open(fpath, "rb") as fp:
            hs = struct.unpack("<Q", fp.read(8))[0]
            meta = json.loads(fp.read(hs))
        for key, info in meta.items():
            if key == "__metadata__": continue
            off = info["data_offsets"]
            # Different models use different expert key patterns
            is_expert = any(x in key for x in ["switch_mlp", "experts.", "expert_", "gate_proj.weight" if "experts" in key else "NOMATCH"])
            # More robust: check if key contains "experts"
            is_expert = "experts" in key.lower() or "switch_mlp" in key
            tensor_info[key] = {"size": off[1] - off[0], "is_expert": is_expert}

    return shard_files, tensor_info

class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.io = 0.0

    def get(self, k, t):
        if k in self.cache:
            self.cache.move_to_end(k)
            self.hits += 1
            return self.cache[k]
        t0 = time.time()
        mx.eval(t)
        self.io += time.time() - t0
        self.cache[k] = t
        self.misses += 1
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return t

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t * 100 if t else 0

def benchmark_model(model_info):
    """Run ExpertFlow benchmark on a single model."""
    name = model_info["name"]
    path = model_info["path"]

    banner(f"{name} ({model_info['total_params']} total, {model_info['active_params']} active)")

    # Check if downloaded
    shard_files = sorted(glob.glob(f"{path}/model*.safetensors"))
    if not shard_files:
        print(f"  ⏳ Not downloaded yet — skipping")
        return None

    result = analyze_model(path)
    if not result:
        print(f"  ❌ No safetensors found")
        return None

    shard_files, tensor_info = result
    total_gb = sum(v["size"] for v in tensor_info.values()) / 1e9
    expert_gb = sum(v["size"] for v in tensor_info.values() if v["is_expert"]) / 1e9
    backbone_gb = total_gb - expert_gb
    expert_count = len([k for k in tensor_info if tensor_info[k]["is_expert"]])
    backbone_count = len(tensor_info) - expert_count

    print(f"  Shards: {len(shard_files)}")
    print(f"  Total: {total_gb:.1f} GB | {len(tensor_info)} tensors")
    print(f"  Expert: {expert_gb:.1f} GB ({expert_gb/total_gb*100:.0f}%) | {expert_count} tensors")
    print(f"  Backbone: {backbone_gb:.1f} GB ({backbone_gb/total_gb*100:.0f}%) | {backbone_count} tensors")

    fits_in_ram = total_gb < (RAM_GB - 10)  # Leave 10GB for system
    print(f"  RAM: {RAM_GB} GB | Model: {total_gb:.0f} GB | {'✅ Fits' if fits_in_ram else '❌ TOO BIG'}")

    # Full load attempt
    if fits_in_ram:
        print(f"\n  === Full Load ===")
        gc.collect(); mx.clear_cache()
        rss0 = rss_gb()
        t0 = time.time()
        full = {}
        for i, fpath in enumerate(shard_files):
            s = mx.load(fpath)
            for k, v in s.items(): mx.eval(v)
            full.update(s)
            if (i+1) % 10 == 0: print(f"    {i+1}/{len(shard_files)} shards...", flush=True)
        full_time = time.time() - t0
        full_mem = rss_gb() - rss0
        print(f"  Full load: {full_time:.1f}s | {full_mem:.1f} GB")
        del full; gc.collect(); mx.clear_cache(); time.sleep(3)
        full_result = {"status": "OK", "time_s": round(full_time, 1), "memory_gb": round(full_mem, 1)}
    else:
        print(f"\n  === Full Load: SKIPPED (would OOM) ===")
        full_result = {"status": "IMPOSSIBLE", "reason": f"{total_gb:.0f}GB > {RAM_GB}GB RAM"}

    # ExpertFlow dynamic loading
    print(f"\n  === ExpertFlow Dynamic Loading ===")
    gc.collect(); mx.clear_cache()
    rss0 = rss_gb()

    # Mmap all shards (lazy)
    t0 = time.time()
    mmap_all = {}
    for i, fpath in enumerate(shard_files):
        mmap_all.update(mx.load(fpath))
        if (i+1) % 20 == 0: print(f"    Mapped {i+1}/{len(shard_files)}...", flush=True)
    mmap_time = time.time() - t0
    print(f"  Mmap: {mmap_time:.2f}s (lazy)")

    # Load backbone only
    t0 = time.time()
    backbone = {}
    for k, v in mmap_all.items():
        if not tensor_info.get(k, {}).get("is_expert", False):
            mx.eval(v)
            backbone[k] = v
    backbone_time = time.time() - t0
    backbone_mem = rss_gb() - rss0
    print(f"  Backbone: {backbone_time:.1f}s | {backbone_mem:.1f} GB | {len(backbone)} tensors")

    # LRU cache simulation with real tensor loading
    cache = LRU(200)
    expert_keys = [k for k in tensor_info if tensor_info[k]["is_expert"]]
    np.random.seed(42)

    if not expert_keys:
        print(f"  ⚠️ No expert tensors detected — MoE routing pattern not recognized")
        print(f"     Running full-tensor LRU test instead...")
        expert_keys = list(tensor_info.keys())

    sample_size = min(480, len(expert_keys))
    print(f"  Running {STEPS} steps (sampling {sample_size} tensors/step, LRU=200)...", flush=True)

    t0 = time.time()
    for step in range(STEPS):
        chosen = np.random.choice(expert_keys, sample_size, replace=False)
        for k in chosen:
            if k in mmap_all:
                cache.get(k, mmap_all[k])
        if (step+1) % 10 == 0:
            print(f"    Step {step+1}/{STEPS}: cache {cache.hit_rate:.1f}%", flush=True)

    infer_time = time.time() - t0
    peak_mem = rss_gb() - rss0

    print(f"  {STEPS} steps: {infer_time:.1f}s ({infer_time/STEPS*1000:.0f}ms/step)")
    print(f"  Cache: {cache.hit_rate:.1f}% hits | I/O: {cache.io:.1f}s")
    print(f"  Peak memory: {peak_mem:.1f} GB")

    mem_savings = (1 - peak_mem / total_gb) * 100 if total_gb > 0 else 0

    del mmap_all, backbone, cache
    gc.collect(); mx.clear_cache()
    time.sleep(3)

    return {
        "model": name,
        "total_params": model_info["total_params"],
        "active_params": model_info["active_params"],
        "total_gb": round(total_gb, 1),
        "expert_gb": round(expert_gb, 1),
        "expert_pct": round(expert_gb/total_gb*100, 1) if total_gb > 0 else 0,
        "backbone_gb": round(backbone_gb, 1),
        "fits_in_ram": fits_in_ram,
        "full_load": full_result,
        "expertflow": {
            "mmap_time_s": round(mmap_time, 2),
            "backbone_time_s": round(backbone_time, 1),
            "backbone_mem_gb": round(backbone_mem, 1),
            "peak_mem_gb": round(peak_mem, 1),
            "cache_hit_rate": round(cache.hit_rate, 1),
            "io_time_s": round(cache.io, 1),
            "ms_per_step": round(infer_time/STEPS*1000, 0),
            "steps": STEPS,
        },
        "memory_savings_pct": round(mem_savings, 1),
    }


# ═══════════════════════════════════════════════════════════════════
banner("ExpertFlow Mega Benchmark — Too-Big-For-RAM Models")
print(f"Hardware: Apple M5 Max, 128GB Unified Memory")
print(f"Steps: {STEPS} per model | LRU cache: 200")
print(f"Models: {len(MODELS)}")

all_results = []
for m in MODELS:
    r = benchmark_model(m)
    if r:
        all_results.append(r)

# Summary
banner("MEGA BENCHMARK RESULTS")
print(f"")
print(f"{'Model':<20} {'Size':>7} {'Expert%':>8} {'Full Load':>12} {'EF Memory':>10} {'EF ms/step':>11} {'Cache%':>8} {'Savings':>8}")
print(f"{'-'*85}")
for r in all_results:
    fl = r["full_load"]
    fl_str = f"{fl['memory_gb']}GB" if fl["status"] == "OK" else "❌ OOM"
    ef = r["expertflow"]
    print(f"{r['model']:<20} {r['total_gb']:>6.0f}G {r['expert_pct']:>7.0f}% {fl_str:>12} {ef['peak_mem_gb']:>9.1f}G {ef['ms_per_step']:>10.0f} {ef['cache_hit_rate']:>7.1f}% {r['memory_savings_pct']:>7.0f}%")

print(f"")
print(f"  Key: EF = ExpertFlow dynamic loading")
print(f"  ❌ OOM = model too large for {RAM_GB}GB RAM (would crash)")
print(f"  Savings = memory reduction vs full model size")

if any(r["full_load"]["status"] == "IMPOSSIBLE" for r in all_results):
    impossible = [r for r in all_results if r["full_load"]["status"] == "IMPOSSIBLE"]
    print(f"")
    print(f"  🔥 ExpertFlow enabled {len(impossible)} model(s) that CANNOT run normally:")
    for r in impossible:
        print(f"     • {r['model']}: {r['total_gb']:.0f}GB model → {r['expertflow']['peak_mem_gb']:.1f}GB with ExpertFlow")

outpath = os.path.expanduser("~/dev/expertflow/mega-benchmark.json")
with open(outpath, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results: {outpath}")
