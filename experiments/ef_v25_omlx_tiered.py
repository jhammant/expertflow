#!/usr/bin/env python3
"""
ExpertFlow v25 — oMLX Tiered Cache Experiment
===============================================
Uses oMLX's PagedSSDCacheManager to implement two-tier KV caching:
  - Hot tier: GPU/RAM for active KV cache blocks
  - Cold tier: SSD for evicted blocks (restored on cache hit)

Tests with the existing GLM-4.5 4-bit MLX model.
oMLX handles the KV cache tiering; ExpertFlow handles expert offloading.
"""

import os
import sys
import time
import json
import subprocess

# Try to use oMLX's engine for serving
def test_omlx_serve(model_path, prompt="The capital of France is", max_tokens=15):
    """Test oMLX server with the model."""

    venv_python = os.path.expanduser("~/dev/expertflow/.venv/bin/python")

    # Check if we can use oMLX's engine directly
    test_code = f'''
import os, sys, time
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
mx.set_memory_limit(int(100 * 1024**3))
mx.set_cache_limit(int(4 * 1024**3))

from omlx import EngineConfig, EngineCore
from omlx.cache import (
    PagedCacheManager, PagedSSDCacheManager,
    TieredCacheManager, CacheConfig
)

model_path = "{model_path}"
print("Setting up oMLX engine...")

# Configure SSD cache
ssd_cache_dir = os.path.expanduser("~/.omlx/ssd_cache")
os.makedirs(ssd_cache_dir, exist_ok=True)

# Try to create engine with tiered caching
try:
    config = EngineConfig(
        model_path=model_path,
        max_tokens=2048,
    )
    engine = EngineCore(config)
    print(f"Engine created: {{type(engine)}}")

    # Generate
    from omlx import Request, SamplingParams
    req = Request(
        prompt="{prompt}",
        sampling_params=SamplingParams(max_tokens={max_tokens}, temperature=0.0),
    )

    print("Generating...")
    t0 = time.time()
    outputs = list(engine.generate(req))
    t1 = time.time()

    for out in outputs:
        print(f"Output: {{out}}")
    print(f"Time: {{t1-t0:.1f}}s")

except Exception as e:
    print(f"Engine approach failed: {{e}}")
    import traceback
    traceback.print_exc()

    # Fallback: try oMLX CLI
    print("\\nTrying oMLX CLI serve...")
    print("Run: omlx serve --model {model_path} --port 8080")
    print("Then: curl http://localhost:8080/v1/completions -d '{{"model":"glm-4.5","prompt":"{prompt}","max_tokens":{max_tokens}}}'")
'''

    result = subprocess.run(
        [venv_python, "-c", test_code],
        capture_output=True, text=True, timeout=300
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    return result.returncode == 0


def test_omlx_ssd_cache():
    """Test oMLX's SSD cache directly."""
    venv_python = os.path.expanduser("~/dev/expertflow/.venv/bin/python")

    test_code = '''
from omlx.cache import PagedSSDCacheManager, PagedSSDCacheStats
import os, tempfile

ssd_dir = os.path.expanduser("~/.omlx/ssd_cache")
os.makedirs(ssd_dir, exist_ok=True)

try:
    mgr = PagedSSDCacheManager(
        cache_dir=ssd_dir,
        max_size_gb=10.0,
        block_size=256,
    )
    print(f"SSD Cache Manager created: {type(mgr)}")
    stats = mgr.stats()
    print(f"Stats: {stats}")
    print("SSD cache is operational!")
except Exception as e:
    print(f"SSD cache test failed: {e}")
    import traceback
    traceback.print_exc()
'''

    result = subprocess.run(
        [venv_python, "-c", test_code],
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-300:])


def main():
    print("=" * 60)
    print("  ExpertFlow v25 — oMLX Tiered Cache Test")
    print("=" * 60)

    model_path = os.path.expanduser("~/models/glm-4.5-4bit")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"  Model: {model_path}")
    print()

    # Test 1: SSD cache subsystem
    print("--- Test 1: SSD Cache Subsystem ---")
    test_omlx_ssd_cache()
    print()

    # Test 2: Full engine with tiered caching
    print("--- Test 2: Full Engine ---")
    test_omlx_serve(model_path)


if __name__ == "__main__":
    main()
