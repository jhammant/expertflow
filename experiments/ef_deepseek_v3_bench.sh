#!/bin/bash
# DeepSeek V3 671B (IQ1_S, 186GB) on M5 Max 128GB
# The headline demo: "671B model, 128GB Mac, actually works"
#
# Architecture: 61 layers, 256 routed experts + 1 shared, 8 active/token
# MLA (Multi-head Latent Attention) for compressed KV cache
#
# Safety: 186GB model on 128GB RAM = mmap streaming required
# NEVER attempt all-GPU (-ngl 999 without --cpu-moe) — exceeds GPU allocation
# Monitor swap — abort if >40GB

set -euo pipefail

MODEL_DIR="$HOME/models/deepseek-v3-gguf/UD-IQ1_S"

# Check model exists
if [ ! -d "$MODEL_DIR" ] || [ "$(ls "$MODEL_DIR"/*.gguf 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "Download:"
    echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('unsloth/DeepSeek-V3-0324-GGUF', allow_patterns=['UD-IQ1_S/*'], local_dir='$HOME/models/deepseek-v3-gguf')\""
    exit 1
fi

MODEL=$(ls "$MODEL_DIR"/*.gguf | sort | head -1)
MODEL_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)

check_swap() {
    local swap_mb
    swap_mb=$(sysctl vm.swapusage | awk '{gsub(/M/,"",$7); print int($7)}')
    echo "  Swap: ${swap_mb}MB"
    if [ "$swap_mb" -gt 40000 ]; then
        echo "  ABORT: Swap exceeded 40GB safety limit!"
        exit 1
    fi
}

echo "============================================"
echo "  DeepSeek V3 671B (IQ1_S) on M5 Max 128GB"
echo "============================================"
echo "  Model: $(basename "$MODEL")"
echo "  Size:  $MODEL_SIZE"
echo "  Arch:  61 layers, 256 experts, 8 active/tok"
check_swap
echo ""

# ─── Test 1: Quick generation (verify model loads and produces text) ───
echo "--- Test 1: Quick generation (--cpu-moe) ---"
echo "  Config: -ngl 999 --cpu-moe -n 20"
echo ""
llama-completion -m "$MODEL" \
    -ngl 999 --cpu-moe \
    -p "The meaning of life is" \
    -n 20 --temp 0 --single-turn \
    2>&1 | tail -15
echo ""
check_swap

# ─── Test 2: Prompt processing benchmark ───
echo ""
echo "--- Test 2: Prompt processing (pp32, pp128) ---"
llama-bench -m "$MODEL" \
    -ngl 999 -ncmoe 999 \
    -p 32,128 -n 0 -r 1 \
    2>&1 | tail -5
check_swap

# ─── Test 3: Generation benchmark ───
echo ""
echo "--- Test 3: Token generation (tg20, tg50) ---"
llama-bench -m "$MODEL" \
    -ngl 999 -ncmoe 999 \
    -p 0 -n 20,50 -r 1 \
    2>&1 | tail -5
check_swap

# ─── Test 4: Combined pp+tg ───
echo ""
echo "--- Test 4: Combined pp128+tg20 ---"
llama-bench -m "$MODEL" \
    -ngl 999 -ncmoe 999 \
    -p 128 -n 20 -r 1 \
    2>&1 | tail -5
check_swap

# ─── Test 5: Longer generation (quality check) ───
echo ""
echo "--- Test 5: 50-token generation ---"
llama-completion -m "$MODEL" \
    -ngl 999 --cpu-moe \
    -p "Paris, the capital of France, has a rich history spanning" \
    -n 50 --temp 0 --single-turn \
    2>&1 | tail -15
check_swap

echo ""
echo "============================================"
echo "  DeepSeek V3 671B — ALL TESTS COMPLETE"
echo "============================================"
echo ""
echo "  This is a 671B parameter model running on a"
echo "  128GB MacBook Pro. The model exceeds system RAM"
echo "  (186GB > 128GB) but runs via mmap streaming with"
echo "  expert routing to CPU for MoE layers."
echo ""
