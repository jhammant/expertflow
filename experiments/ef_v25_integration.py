#!/usr/bin/env python3
"""
ExpertFlow v25 — Integration Benchmark
========================================
Tests all viable approaches for running GLM-4.5 (355B MoE) on M5 Max 128GB:

1. llama.cpp --cpu-moe (GGUF TQ1_0, 84GB)
2. llama.cpp --cpu-moe (GGUF IQ2_XXS, 116GB) [if available]
3. ExpertFlow MLX engine with LRU cache (existing 4-bit, 185GB)
4. ExpertFlow MLX engine with Belady-approximate cache (v25)
5. oMLX server with SSD KV cache tiering [if installed]

Outputs a comparison table and JSON results.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
MODELS = {
    "glm45_tq1": Path.home() / "models/glm-4.5-gguf/GLM-4.5-UD-TQ1_0.gguf",
    "glm45_iq2": Path.home() / "models/glm-4.5-gguf/UD-IQ2_XXS",
    "glm45_mlx4bit": Path.home() / "models/glm-4.5-4bit",
}

PROMPT = "The capital of France is"
N_TOKENS = 15

def check_tools():
    """Check which tools are available."""
    tools = {}
    tools['llama-cli'] = bool(subprocess.run(
        ["which", "llama-cli"], capture_output=True).returncode == 0)
    tools['omlx'] = bool(subprocess.run(
        ["which", "omlx"], capture_output=True).returncode == 0)
    tools['mlx_lm'] = False
    try:
        # Use the project venv
        venv_python = str(Path.home() / "dev/expertflow/.venv/bin/python")
        result = subprocess.run(
            [venv_python, "-c", "import mlx_lm"],
            capture_output=True, timeout=10)
        tools['mlx_lm'] = result.returncode == 0
    except:
        pass
    return tools

def check_models():
    """Check which models are available."""
    available = {}
    for name, path in MODELS.items():
        if path.is_file():
            size_gb = path.stat().st_size / (1024**3)
            available[name] = {"path": str(path), "size_gb": round(size_gb, 1)}
        elif path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*.gguf") if f.is_file())
            if total == 0:
                total = sum(f.stat().st_size for f in path.rglob("*.safetensors") if f.is_file())
            if total > 0:
                available[name] = {"path": str(path), "size_gb": round(total / (1024**3), 1)}
    return available


def bench_llamacpp(model_path, config_name, extra_args=None, timeout_s=600):
    """Benchmark llama.cpp with given config."""
    cmd = [
        "llama-cli",
        "-m", str(model_path),
        "-p", PROMPT,
        "-n", str(N_TOKENS),
        "--no-display-prompt",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd[:6])}... {' '.join(cmd[6:])}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.time() - start

        # Parse llama.cpp output for timing info
        output = result.stdout + result.stderr
        tok_s = None
        for line in output.split('\n'):
            if 'eval time' in line.lower() and 'token' in line.lower():
                # Try to extract tokens/s
                parts = line.split()
                for i, p in enumerate(parts):
                    if 'token' in p.lower() and i > 0:
                        try:
                            tok_s = float(parts[i-1])
                        except:
                            pass

        return {
            "config": config_name,
            "success": result.returncode == 0,
            "elapsed_s": round(elapsed, 1),
            "tok_s": tok_s,
            "output": result.stdout[:500],
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "config": config_name,
            "success": False,
            "elapsed_s": timeout_s,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "config": config_name,
            "success": False,
            "error": str(e),
        }


def bench_mlx_ef(model_path, script, extra_args=None, timeout_s=600):
    """Benchmark ExpertFlow MLX engine."""
    venv_python = str(Path.home() / "dev/expertflow/.venv/bin/python")
    cmd = [
        venv_python, str(script),
        "--model", str(model_path),
        "--prompt", PROMPT,
        "--max-tokens", str(N_TOKENS),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {os.path.basename(str(script))} ...")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.time() - start

        return {
            "config": f"mlx_{os.path.basename(str(script)).replace('.py','')}",
            "success": result.returncode == 0,
            "elapsed_s": round(elapsed, 1),
            "output": result.stdout[-1000:],
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "config": f"mlx_{os.path.basename(str(script))}",
            "success": False,
            "elapsed_s": timeout_s,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "config": f"mlx_{os.path.basename(str(script))}",
            "success": False,
            "error": str(e),
        }


def main():
    print("=" * 70)
    print("  ExpertFlow v25 — Integration Benchmark")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Prompt: {PROMPT!r}")
    print(f"  Tokens: {N_TOKENS}")
    print()

    tools = check_tools()
    print("  Tools:")
    for name, avail in tools.items():
        print(f"    {name}: {'OK' if avail else 'NOT AVAILABLE'}")

    models = check_models()
    print("\n  Models:")
    for name, info in models.items():
        print(f"    {name}: {info['size_gb']} GB at {info['path']}")
    if not models:
        print("    No models found!")
        return

    print()
    results = []

    # --- llama.cpp benchmarks ---
    if tools['llama-cli']:
        for model_name in ['glm45_tq1', 'glm45_iq2']:
            if model_name not in models:
                print(f"  Skipping {model_name} (not downloaded)")
                continue

            model_path = models[model_name]['path']

            # Config: --cpu-moe (recommended)
            print(f"\n--- llama.cpp {model_name} --cpu-moe ---")
            r = bench_llamacpp(model_path, f"llama_{model_name}_cpumoe",
                              ["-ngl", "999", "--cpu-moe"])
            results.append(r)
            print(f"  Result: {'OK' if r['success'] else 'FAIL'} in {r.get('elapsed_s','?')}s")

            # Config: all GPU
            print(f"\n--- llama.cpp {model_name} all-GPU ---")
            r = bench_llamacpp(model_path, f"llama_{model_name}_allgpu",
                              ["-ngl", "999"], timeout_s=300)
            results.append(r)
            print(f"  Result: {'OK' if r['success'] else 'FAIL'} in {r.get('elapsed_s','?')}s")

    # --- MLX ExpertFlow benchmarks ---
    if tools['mlx_lm'] and 'glm45_mlx4bit' in models:
        model_path = models['glm45_mlx4bit']['path']

        # v24 LRU baseline
        v24_script = RESULTS_DIR / "ef_v24_lean.py"
        if v24_script.exists():
            print(f"\n--- ExpertFlow v24 (LRU) ---")
            r = bench_mlx_ef(model_path, v24_script, ["--cache-budget", "300"])
            results.append(r)
            print(f"  Result: {'OK' if r['success'] else 'FAIL'} in {r.get('elapsed_s','?')}s")

        # v25 Belady cache
        v25_script = RESULTS_DIR / "ef_v25_belady_cache.py"
        if v25_script.exists():
            print(f"\n--- ExpertFlow v25 (Belady) ---")
            r = bench_mlx_ef(model_path, v25_script, ["--cache-budget", "300"])
            results.append(r)
            print(f"  Result: {'OK' if r['success'] else 'FAIL'} in {r.get('elapsed_s','?')}s")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<40} {'Status':<10} {'Time':<10} {'tok/s':<10}")
    print("  " + "-" * 66)
    for r in results:
        status = "OK" if r.get('success') else "FAIL"
        elapsed = f"{r.get('elapsed_s', '?')}s"
        tok_s = f"{r['tok_s']:.2f}" if r.get('tok_s') else "?"
        print(f"  {r.get('config','?'):<40} {status:<10} {elapsed:<10} {tok_s:<10}")

    # Save results
    outfile = RESULTS_DIR / f"v25_integration_{time.strftime('%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump({"timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                   "results": results}, f, indent=2)
    print(f"\n  Results saved: {outfile}")


if __name__ == "__main__":
    main()
