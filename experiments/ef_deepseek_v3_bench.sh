#!/bin/bash
# DeepSeek V3 671B (IQ1_S, 186GB) on M5 Max 128GB
# The headline demo: "671B model, 128GB Mac, actually works"
#
# Safety: 186GB model on 128GB RAM = needs mmap streaming
# NEVER load without --cpu-moe on this model
# Monitor swap — abort if >40GB

set -e

MODEL_DIR="$HOME/models/deepseek-v3-gguf/UD-IQ1_S"
MODEL_PATTERN="$MODEL_DIR/DeepSeek-V3-0324-UD-IQ1_S"

# Check model exists
if [ ! -d "$MODEL_DIR" ] || [ $(ls "$MODEL_DIR"/*.gguf 2>/dev/null | wc -l) -eq 0 ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "Download with: huggingface-cli download unsloth/DeepSeek-V3-0324-GGUF --include 'UD-IQ1_S/*' --local-dir ~/models/deepseek-v3-gguf"
    exit 1
fi

# Find the first split file (llama.cpp auto-loads splits)
MODEL=$(ls "$MODEL_DIR"/*.gguf | head -1)
echo "Model: $MODEL"
echo "Size: $(du -sh "$MODEL_DIR" | cut -f1)"

# Check swap before starting
SWAP=$(sysctl vm.swapusage | awk '{print $7}' | tr -d 'M.')
echo "Swap before: ${SWAP}M"

echo ""
echo "============================================"
echo "  DeepSeek V3 671B (IQ1_S) Benchmark"
echo "  128GB M5 Max — Expert Streaming Demo"
echo "============================================"

# Test 1: All layers on GPU, experts on CPU (primary config)
echo ""
echo "--- Test 1: GPU + CPU-MoE (recommended) ---"
echo "Config: -ngl 999 --cpu-moe"
llama-completion -m "$MODEL" \
    -ngl 999 --cpu-moe \
    -n 20 -p "The meaning of life is" \
    --temp 0 \
    2>&1 | grep -E "(t/s|timings|load|offload|model|perf)"

# Check swap
SWAP=$(sysctl vm.swapusage | awk '{print $7}' | tr -d 'M.')
echo "Swap after test 1: ${SWAP}M"
if [ "${SWAP%.*}" -gt 40000 ]; then
    echo "WARNING: Swap exceeded 40GB! Aborting."
    exit 1
fi

# Test 2: Benchmark with llama-bench
echo ""
echo "--- Test 2: llama-bench pp32/tg10 ---"
llama-bench -m "$MODEL" \
    -ngl 999 --cpu-moe \
    -p 32 -n 10 -r 1 \
    2>&1 | tail -5

# Check swap again
SWAP=$(sysctl vm.swapusage | awk '{print $7}' | tr -d 'M.')
echo "Swap after test 2: ${SWAP}M"

echo ""
echo "============================================"
echo "  DeepSeek V3 671B — COMPLETE"
echo "============================================"
