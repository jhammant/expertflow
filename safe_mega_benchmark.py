#!/usr/bin/env python3
"""
ExpertFlow Safe Mega Benchmark
===============================
Analyzes model structure from safetensor headers (no tensor materialization).
Simulates expert loading patterns and measures real NVMe bandwidth.
Memory-safe: never loads more than 1 expert tensor at a time.
"""

import os, sys, time, json, glob, struct, gc, resource
import numpy as np

MODELS = [
    {"name": "DeepSeek V3.1", "path": os.path.expanduser("~/models/deepseek-v3.1-4bit"),
     "total_params": "671B", "active_params": "37B", "expert_key": "experts",
     "num_experts": 256, "num_active": 8, "num_layers": 61},
    {"name": "MiniMax-M2", "path": os.path.expanduser("~/models/minimax-m2-4bit"),
     "total_params": "230B", "active_params": "10B", "expert_key": "experts",
     "num_experts": 128, "num_active": 8, "num_layers": 56},
    {"name": "GLM-4.5 Full", "path": os.path.expanduser("~/models/glm-4.5-4bit"),
     "total_params": "355B", "active_params": "32B", "expert_key": "experts",
     "num_experts": 256, "num_active": 8, "num_layers": 62},
]

RAM_GB = 128

def get_free_gb():
    """Get free memory in GB on macOS."""
    import subprocess
    out = subprocess.check_output(["vm_stat"]).decode()
    free = inactive = 0
    for line in out.split("\n"):
        if "Pages free" in line:
            free = int(line.split()[-1].rstrip("."))
        elif "Pages inactive" in line:
            inactive = int(line.split()[-1].rstrip("."))
    return (free + inactive) * 16384 / 1e9

def banner(t):
    print(f"\n{'#'*70}\n  {t}\n{'#'*70}", flush=True)

def analyze_model_headers(path, expert_key):
    """Parse safetensor headers WITHOUT loading any tensor data."""
    shard_files = sorted(glob.glob(f"{path}/model*.safetensors"))
    if not shard_files:
        return None
    
    tensors = {}  # key -> {size, is_expert, shard_path, offset, shape}
    
    for fpath in shard_files:
        with open(fpath, "rb") as fp:
            hs = struct.unpack("<Q", fp.read(8))[0]
            meta = json.loads(fp.read(hs))
            header_end = 8 + hs
        
        for key, info in meta.items():
            if key == "__metadata__":
                continue
            off = info["data_offsets"]
            tensors[key] = {
                "size": off[1] - off[0],
                "is_expert": any(x in key for x in ["switch_mlp", "block_sparse_moe", "shared_expert"]),
                "shard": fpath,
                "offset_start": header_end + off[0],
                "offset_end": header_end + off[1],
                "shape": info.get("shape", []),
                "dtype": info.get("dtype", "unknown"),
            }
    
    return shard_files, tensors

def measure_nvme_bandwidth(shard_files, num_reads=10, read_size_mb=64):
    """Measure actual NVMe sequential read bandwidth by reading raw bytes."""
    print(f"  Measuring NVMe bandwidth ({num_reads} reads × {read_size_mb}MB)...", flush=True)
    
    read_bytes = read_size_mb * 1024 * 1024
    times = []
    
    for i in range(num_reads):
        fpath = shard_files[i % len(shard_files)]
        fsize = os.path.getsize(fpath)
        offset = min(i * read_bytes, max(0, fsize - read_bytes))
        
        t0 = time.time()
        with open(fpath, "rb") as fp:
            fp.seek(offset)
            data = fp.read(read_bytes)
        elapsed = time.time() - t0
        times.append(elapsed)
        
        del data
    
    avg_time = np.mean(times)
    bandwidth_gbs = (read_size_mb / 1024) / avg_time
    print(f"  NVMe bandwidth: {bandwidth_gbs:.1f} GB/s (avg {avg_time*1000:.1f}ms per {read_size_mb}MB)", flush=True)
    return bandwidth_gbs

def simulate_expert_loading(tensors, config, nvme_bandwidth_gbs):
    """
    Simulate ExpertFlow dynamic loading based on actual tensor sizes and NVMe speed.
    No actual tensor data is loaded — purely analytical.
    """
    expert_tensors = {k: v for k, v in tensors.items() if v["is_expert"]}
    backbone_tensors = {k: v for k, v in tensors.items() if not v["is_expert"]}
    
    expert_gb = sum(v["size"] for v in expert_tensors.values()) / 1e9
    backbone_gb = sum(v["size"] for v in backbone_tensors.values()) / 1e9
    total_gb = expert_gb + backbone_gb
    
    num_experts = config["num_experts"]
    num_active = config["num_active"]
    num_layers = config["num_layers"]
    
    # Group expert tensors by layer
    # Modern MoE models pack all experts into one tensor: shape [num_experts, hidden, dim]
    # So we calculate per-expert size by dividing packed tensor by num_experts
    layer_expert_bytes = {}  # layer_idx -> total expert bytes in that layer
    for key, info in expert_tensors.items():
        parts = key.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try: layer_idx = int(parts[i + 1])
                except: pass
        if layer_idx is not None:
            layer_expert_bytes[layer_idx] = layer_expert_bytes.get(layer_idx, 0) + info["size"]
    
    unique_layers = len(layer_expert_bytes) if layer_expert_bytes else num_layers
    
    # Per-expert size = total expert bytes per layer / num_experts
    if layer_expert_bytes:
        avg_layer_expert_bytes = np.mean(list(layer_expert_bytes.values()))
        avg_expert_bytes = avg_layer_expert_bytes / num_experts  # single expert slice
        unique_experts = num_experts
    else:
        avg_expert_bytes = expert_gb * 1e9 / max(1, num_experts * num_layers)
        unique_experts = num_experts
    
    # Create synthetic expert groups for simulation
    expert_groups = {}
    for l in range(unique_layers):
        for e in range(unique_experts):
            expert_groups[(l, e)] = avg_expert_bytes
    
    avg_expert_mb = avg_expert_bytes / 1e6
    
    print(f"  Expert groups: {len(expert_groups)} (layer, expert) pairs")
    print(f"  Unique experts: {unique_experts}, Unique layers: {unique_layers}")
    print(f"  Avg expert group: {avg_expert_mb:.2f} MB")
    
    # Simulate LRU caching across different cache sizes
    results = {}
    for cache_size in [50, 100, 200, 500]:
        np.random.seed(42)
        cache = set()
        cache_order = []
        hits = misses = 0
        io_bytes = 0
        steps = 200
        
        for step in range(steps):
            # Simulate router: select num_active experts per layer
            for layer in range(min(unique_layers, num_layers)):
                selected = np.random.choice(unique_experts, size=num_active, replace=False)
                for expert_id in selected:
                    key = (layer, int(expert_id))
                    if key in cache:
                        hits += 1
                        cache_order.remove(key)
                        cache_order.append(key)
                    else:
                        misses += 1
                        io_bytes += avg_expert_bytes
                        cache.add(key)
                        cache_order.append(key)
                        while len(cache) > cache_size:
                            evicted = cache_order.pop(0)
                            cache.discard(evicted)
        
        total = hits + misses
        hit_rate = hits / total * 100 if total > 0 else 0
        io_gb = io_bytes / 1e9
        io_time = io_gb / nvme_bandwidth_gbs
        ms_per_step = (io_time / steps) * 1000
        active_mem_gb = cache_size * avg_expert_bytes / 1e9
        
        results[f"LRU-{cache_size}"] = {
            "cache_size": cache_size,
            "hit_rate": round(hit_rate, 1),
            "miss_rate": round(100 - hit_rate, 1),
            "io_gb": round(io_gb, 1),
            "io_time_s": round(io_time, 2),
            "ms_per_step": round(ms_per_step, 1),
            "active_memory_gb": round(active_mem_gb, 1),
            "steps": steps,
        }
    
    return {
        "total_gb": round(total_gb, 1),
        "expert_gb": round(expert_gb, 1),
        "backbone_gb": round(backbone_gb, 1),
        "expert_pct": round(expert_gb / total_gb * 100, 1) if total_gb > 0 else 0,
        "expert_tensors": len(expert_tensors),
        "backbone_tensors": len(backbone_tensors),
        "expert_groups": len(expert_groups),
        "avg_expert_mb": round(avg_expert_mb, 2),
        "unique_experts": unique_experts,
        "fits_in_ram": total_gb < (RAM_GB - 10),
        "caching": results,
    }


def measure_single_expert_load(shard_files, tensors, expert_key):
    """Load ONE expert tensor to measure real load time, then immediately free it."""
    import mlx.core as mx
    
    expert_keys = [k for k in tensors if tensors[k]["is_expert"]]
    if not expert_keys:
        return None
    
    # Pick the first expert tensor
    key = expert_keys[0]
    info = tensors[key]
    size_mb = info["size"] / 1e6
    
    # Load just this one tensor
    shard = info["shard"]
    t0 = time.time()
    loaded = mx.load(shard)
    mx.eval(loaded[key])
    load_time = time.time() - t0
    
    # Get actual size
    tensor = loaded[key]
    
    result = {
        "tensor_key": key,
        "size_mb": round(size_mb, 2),
        "load_time_ms": round(load_time * 1000, 1),
        "bandwidth_gbs": round(size_mb / 1000 / load_time, 2) if load_time > 0 else 0,
    }
    
    del loaded, tensor
    gc.collect()
    mx.clear_cache()
    
    return result


def benchmark_model(model_info):
    """Full benchmark for one model — memory-safe."""
    name = model_info["name"]
    path = model_info["path"]
    expert_key = model_info["expert_key"]
    
    banner(f"{name} ({model_info['total_params']} total, {model_info['active_params']} active)")
    
    # Check free memory
    free = get_free_gb()
    print(f"  Free memory: {free:.1f} GB", flush=True)
    
    # Check download status
    shard_files = sorted(glob.glob(f"{path}/model*.safetensors"))
    if not shard_files:
        print(f"  ⏳ Not downloaded yet")
        return None
    
    # Phase 1: Header analysis (zero memory cost)
    print(f"\n  === Phase 1: Header Analysis ===", flush=True)
    result = analyze_model_headers(path, expert_key)
    if not result:
        return None
    
    shard_files, tensors = result
    total_gb = sum(v["size"] for v in tensors.values()) / 1e9
    expert_gb = sum(v["size"] for v in tensors.values() if v["is_expert"]) / 1e9
    backbone_gb = total_gb - expert_gb
    
    print(f"  Shards: {len(shard_files)}")
    print(f"  Total: {total_gb:.1f} GB | {len(tensors)} tensors")
    print(f"  Expert: {expert_gb:.1f} GB ({expert_gb/total_gb*100:.0f}%)")
    print(f"  Backbone: {backbone_gb:.1f} GB ({backbone_gb/total_gb*100:.0f}%)")
    print(f"  RAM: {RAM_GB} GB | Model: {total_gb:.0f} GB | {'✅ Fits' if total_gb < RAM_GB - 10 else '❌ TOO BIG'}")
    
    # Phase 2: NVMe bandwidth measurement
    print(f"\n  === Phase 2: NVMe Bandwidth ===", flush=True)
    nvme_bw = measure_nvme_bandwidth(shard_files)
    
    # Phase 3: Single expert load test (real, but just one tensor)
    print(f"\n  === Phase 3: Single Expert Load Test ===", flush=True)
    free_before = get_free_gb()
    if free_before > 20:
        single = measure_single_expert_load(shard_files, tensors, expert_key)
        if single:
            print(f"  Loaded: {single['tensor_key'][:60]}...")
            print(f"  Size: {single['size_mb']:.1f} MB | Time: {single['load_time_ms']:.0f}ms | BW: {single['bandwidth_gbs']:.1f} GB/s")
        free_after = get_free_gb()
        print(f"  Memory: {free_before:.1f}→{free_after:.1f} GB (delta: {free_before-free_after:.1f} GB)")
    else:
        single = None
        print(f"  ⚠️ Skipped — only {free_before:.1f} GB free")
    
    # Phase 4: Simulated expert loading analysis
    print(f"\n  === Phase 4: Simulated Expert Loading ===", flush=True)
    sim = simulate_expert_loading(tensors, model_info, nvme_bw)
    
    # Print cache analysis
    print(f"\n  {'Cache':>10} {'Hit Rate':>10} {'Miss Rate':>10} {'I/O':>8} {'ms/step':>10} {'Mem':>8}")
    print(f"  {'-'*60}")
    for strat, r in sim["caching"].items():
        print(f"  {strat:>10} {r['hit_rate']:>9.1f}% {r['miss_rate']:>9.1f}% {r['io_gb']:>7.1f}G {r['ms_per_step']:>9.1f} {r['active_memory_gb']:>7.1f}G")
    
    # Memory savings
    best_cache = sim["caching"].get("LRU-200", sim["caching"].get("LRU-100", {}))
    ef_mem = sim["backbone_gb"] + best_cache.get("active_memory_gb", 0)
    savings = (1 - ef_mem / total_gb) * 100 if total_gb > 0 else 0
    
    print(f"\n  📊 ExpertFlow Summary:")
    print(f"     Full model:     {total_gb:.0f} GB (would {'OOM' if total_gb > RAM_GB else 'fit'})")
    print(f"     Backbone only:  {backbone_gb:.1f} GB")
    print(f"     ExpertFlow:     {ef_mem:.1f} GB (backbone + LRU-200 cache)")
    print(f"     Memory savings: {savings:.0f}%")
    print(f"     NVMe bandwidth: {nvme_bw:.1f} GB/s")
    
    return {
        "model": name,
        "total_params": model_info["total_params"],
        "active_params": model_info["active_params"],
        "analysis": sim,
        "nvme_bandwidth_gbs": round(nvme_bw, 1),
        "single_expert_load": single,
        "full_load": {"status": "IMPOSSIBLE" if total_gb > RAM_GB - 10 else "POSSIBLE",
                      "would_need_gb": round(total_gb, 1)},
        "expertflow": {
            "backbone_gb": round(backbone_gb, 1),
            "total_with_cache_gb": round(ef_mem, 1),
            "memory_savings_pct": round(savings, 1),
        },
    }


# ════════════════════════════════════════════════════════════════
banner("ExpertFlow Safe Mega Benchmark")
print(f"Hardware: Apple M5 Max, {RAM_GB}GB Unified Memory")
print(f"Strategy: Header analysis + NVMe bandwidth + simulated caching")
print(f"Memory-safe: max 1 expert tensor loaded at a time")
print(f"Free memory: {get_free_gb():.1f} GB")

all_results = []
for m in MODELS:
    r = benchmark_model(m)
    if r:
        all_results.append(r)
    gc.collect()
    time.sleep(5)

# Summary
banner("MEGA BENCHMARK RESULTS")
print(f"\n{'Model':<18} {'Size':>7} {'Expert%':>8} {'Full?':>8} {'EF Mem':>8} {'Savings':>9} {'NVMe':>8}")
print(f"{'-'*70}")
for r in all_results:
    a = r["analysis"]
    fl = "❌ OOM" if r["full_load"]["status"] == "IMPOSSIBLE" else "✅"
    print(f"{r['model']:<18} {a['total_gb']:>6.0f}G {a['expert_pct']:>7.0f}% {fl:>8} {r['expertflow']['total_with_cache_gb']:>7.1f}G {r['expertflow']['memory_savings_pct']:>8.0f}% {r['nvme_bandwidth_gbs']:>7.1f}")

if any(r["full_load"]["status"] == "IMPOSSIBLE" for r in all_results):
    impossible = [r for r in all_results if r["full_load"]["status"] == "IMPOSSIBLE"]
    print(f"\n  🔥 ExpertFlow enables {len(impossible)} model(s) that CANNOT run normally:")
    for r in impossible:
        print(f"     • {r['model']}: {r['analysis']['total_gb']:.0f}GB → {r['expertflow']['total_with_cache_gb']:.1f}GB with ExpertFlow")

outpath = os.path.expanduser("~/dev/expertflow/mega-benchmark.json")
with open(outpath, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults: {outpath}")
