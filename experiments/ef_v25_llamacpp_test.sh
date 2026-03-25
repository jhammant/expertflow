#!/usr/bin/env bash
# ExpertFlow v25 — llama.cpp MoE offloading quick test with OLMoE
# OLMoE: 7B total (1B active), 64 experts, 8 active per token
set -euo pipefail

MODEL="${1:-$HOME/models/olmoe-gguf/olmoe-1b-7b-0924-q4_k_m.gguf}"
PROMPT="The capital of France is"
N_PREDICT=20
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  llama.cpp MoE Offloading Test — $(basename "$MODEL")"
echo "============================================================"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "ERROR: $MODEL not found"
    exit 1
fi

echo "Model: $(du -h "$MODEL" | cut -f1)"
echo ""

# Test 1: Standard (all GPU)
echo "--- Test 1: All GPU (-ngl 999) ---"
time llama-cli -m "$MODEL" -ngl 999 -p "$PROMPT" -n $N_PREDICT --no-display-prompt 2>&1 | grep -E "^|eval time|load time|sample time"
echo ""

# Test 2: CPU-MOE (experts on CPU, attention on GPU)
echo "--- Test 2: CPU-MOE (experts CPU, attention GPU) ---"
time llama-cli -m "$MODEL" -ngl 999 --cpu-moe -p "$PROMPT" -n $N_PREDICT --no-display-prompt 2>&1 | grep -E "^|eval time|load time|sample time"
echo ""

# Test 3: All CPU
echo "--- Test 3: All CPU (-ngl 0) ---"
time llama-cli -m "$MODEL" -ngl 0 -p "$PROMPT" -n $N_PREDICT --no-display-prompt 2>&1 | grep -E "^|eval time|load time|sample time"
echo ""

# Test 4: Partial expert offload
echo "--- Test 4: Partial expert offload (--n-cpu-moe 16) ---"
time llama-cli -m "$MODEL" -ngl 999 --n-cpu-moe 16 -p "$PROMPT" -n $N_PREDICT --no-display-prompt 2>&1 | grep -E "^|eval time|load time|sample time"
echo ""

echo "============================================================"
echo "  Test complete"
echo "============================================================"
