#!/usr/bin/env bash
# ExpertFlow v25 — llama.cpp --cpu-moe benchmark for GLM-4.5
# Tests attention-on-GPU + experts-on-CPU with various configurations
set -euo pipefail

MODEL="${1:-$HOME/models/glm-4.5-gguf/GLM-4.5-UD-TQ1_0.gguf}"
PROMPT="The capital of France is"
N_PREDICT=20
RESULTS_DIR="$HOME/dev/expertflow/experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  ExpertFlow v25 — llama.cpp MoE Offloading Benchmark"
echo "============================================================"
echo "  Model: $(basename "$MODEL")"
echo "  Time:  $(date)"
echo "  RAM:   $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1073741824}')"
echo ""

# Check model exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Download with: huggingface-cli download unsloth/GLM-4.5-GGUF GLM-4.5-UD-TQ1_0.gguf --local-dir ~/models/glm-4.5-gguf"
    exit 1
fi

echo "Model size: $(du -h "$MODEL" | cut -f1)"
echo ""

# Config 1: Full GPU (baseline - will likely OOM or be slow)
echo "--- Config 1: All layers GPU (baseline) ---"
echo "  llama-cli -m $MODEL -ngl 999 -p '$PROMPT' -n $N_PREDICT"
timeout 300 llama-cli \
    -m "$MODEL" \
    -ngl 999 \
    -p "$PROMPT" \
    -n $N_PREDICT \
    --no-display-prompt \
    2>&1 | tee "${RESULTS_DIR}/v25_config1_${TIMESTAMP}.log" || echo "  Config 1 failed (likely OOM)"
echo ""

# Config 2: All experts on CPU (recommended)
echo "--- Config 2: Experts on CPU (--cpu-moe) ---"
echo "  llama-cli -m $MODEL -ngl 999 --cpu-moe -p '$PROMPT' -n $N_PREDICT"
timeout 600 llama-cli \
    -m "$MODEL" \
    -ngl 999 \
    --cpu-moe \
    -p "$PROMPT" \
    -n $N_PREDICT \
    --no-display-prompt \
    2>&1 | tee "${RESULTS_DIR}/v25_config2_${TIMESTAMP}.log" || echo "  Config 2 failed"
echo ""

# Config 3: Partial expert offload (first 46 layers CPU, rest GPU)
echo "--- Config 3: Partial expert offload (46 layers CPU) ---"
echo "  llama-cli -m $MODEL -ngl 999 --n-cpu-moe 46 -p '$PROMPT' -n $N_PREDICT"
timeout 600 llama-cli \
    -m "$MODEL" \
    -ngl 999 \
    --n-cpu-moe 46 \
    -p "$PROMPT" \
    -n $N_PREDICT \
    --no-display-prompt \
    2>&1 | tee "${RESULTS_DIR}/v25_config3_${TIMESTAMP}.log" || echo "  Config 3 failed"
echo ""

# Config 4: Fine-grained override - experts CPU for high layers only
echo "--- Config 4: Override tensor (layers 40+ experts on CPU) ---"
echo "  llama-cli -m $MODEL -ngl 999 -ot '[4-9][0-9]\.ffn_.*_exps\.=CPU' -p '$PROMPT' -n $N_PREDICT"
timeout 600 llama-cli \
    -m "$MODEL" \
    -ngl 999 \
    -ot "[4-9][0-9]\.ffn_.*_exps\.=CPU" \
    -p "$PROMPT" \
    -n $N_PREDICT \
    --no-display-prompt \
    2>&1 | tee "${RESULTS_DIR}/v25_config4_${TIMESTAMP}.log" || echo "  Config 4 failed"
echo ""

# Config 5: CPU-only (no GPU at all, as a baseline)
echo "--- Config 5: CPU-only (ngl=0) ---"
echo "  llama-cli -m $MODEL -ngl 0 -p '$PROMPT' -n $N_PREDICT"
timeout 600 llama-cli \
    -m "$MODEL" \
    -ngl 0 \
    -p "$PROMPT" \
    -n $N_PREDICT \
    --no-display-prompt \
    2>&1 | tee "${RESULTS_DIR}/v25_config5_${TIMESTAMP}.log" || echo "  Config 5 failed"
echo ""

echo "============================================================"
echo "  Benchmark complete. Results in ${RESULTS_DIR}/v25_config*_${TIMESTAMP}.log"
echo "============================================================"
