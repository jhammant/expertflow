# M5 Optimisation Notes

Based on Apple ML Research: "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"
and ModelFit.io benchmarks (March 2026).

## Key M5 Architecture Facts

### Neural Accelerators (NEW)
- Embedded in **every GPU core** (40 on M5 Max)
- Dedicated matrix-multiplication operations
- Accessed via **Metal 4 TensorOps** + **Metal Performance Primitives**
- Requires **macOS 26.2 or later**

### Memory Bandwidth
- M5 base: 153 GB/s (vs M4: 120 GB/s, +28%)
- M5 Max: 614 GB/s (vs M4 Max: 546 GB/s, +12%)

### Performance Impact on LLM Inference
- **TTFT (Time to First Token):** 3.3–4x faster (compute-bound → Neural Accelerators help)
- **Token Generation:** 19–27% faster (bandwidth-bound → modest improvement)
- **Image gen (FLUX):** 3.8x faster

## What This Means for FlashMoE

### 1. Expert FFN Compute → Neural Accelerators
The expert FFN/MLP layers are large matrix multiplications — exactly what Neural Accelerators
are designed for. FlashMoE should dispatch expert compute through Metal 4 TensorOps instead
of generic Metal compute shaders.

**Action:** Add a `NeuralAccelerator` compute backend that uses Metal Performance Primitives
for expert FFN execution. Fall back to standard Metal for non-matmul ops.

### 2. MLX as Primary Backend (Replace llama.cpp)
Apple's MLX framework is:
- 20–30% faster than llama.cpp on Apple Silicon
- Up to 50% faster than Ollama
- Native support for Neural Accelerators via TensorOps
- Native quantization support (4-bit, MXFP4)
- Unified memory aware — no copies needed

**Action:** Replace the llama.cpp FFI bridge with an MLX Python/Swift bridge.
The `bridge/llamacpp.rs` and FFI hooks should be replaced with MLX integration.

MLX models available on HuggingFace (`mlx-community/`):
- `mlx-community/Qwen3-30B-A3B-MLX-4bit` — 17.31GB, perfect MoE test
- `mlx-community/Qwen3-14B-MLX-4bit` — 9.16GB
- `mlx-community/Llama-3.3-70B-Instruct-4bit` — ~40GB

### 3. Revised Memory Budget (M5 Max 128GB)
With 614 GB/s bandwidth and 128GB unified memory:

| Model | Q4 Size | Experts Resident | Cold Storage | Token Gen |
|-------|---------|-----------------|--------------|-----------|
| Qwen3-30B MoE | 17GB | All (fits in RAM) | None needed | ~95 t/s |
| Qwen3 235B MoE | ~110GB | ~70% hot | ~30% on NVMe | 15-25 t/s |
| DeepSeek V3 1-bit | ~100GB | ~80% hot | ~20% on NVMe | 10-18 t/s |

For models that fit entirely in 128GB, FlashMoE's value is in:
- **Dynamic pinning** — keep frequently-used experts page-locked
- **Prefetch for context switches** — when routing pattern changes
- **Multi-model hot-swap** — hold 2-3 models simultaneously

### 4. Prefetcher Tuning for M5 NVMe
M5 Max SSD: 7.4 GB/s sequential read
- Expert size (typical MoE): 200-500MB per expert
- Prefetch time: 27-68ms per expert
- With 2-layer lookahead: ~55-136ms budget (sufficient for 20+ t/s generation)

**Action:** Increase default `prefetch_lookahead` from 1 to 2 on M5 Max.
Add bandwidth-aware prefetch scheduling — avoid saturating the SSD bus.

### 5. MoE Sweet Spot on M5
Apple's benchmarks show MoE models are the sweet spot:
- Qwen3-30B (3B active params, 4-bit): TTFT under 3 seconds, 17GB memory
- Only 3B parameters active per token → bandwidth-efficient
- FlashMoE adds value by keeping the RIGHT experts hot based on conversation context

**Action:** Optimise the temperature tracker for MoE routing patterns.
Track per-conversation expert affinity (e.g., "coding" conversations use different
experts than "creative writing" — pre-warm based on system prompt analysis).

### 6. Multi-Model Orchestration
128GB enables holding multiple models simultaneously:
- Primary: 70B Q4 (40GB) for complex tasks
- Draft: 7B (4.4GB) for speculative decoding
- Specialist: 14B code model (9GB) for coding tasks
- Total: ~53GB, leaving 75GB for KV cache + system

**Action:** Add a `ModelOrchestrator` that manages multiple loaded models
and routes requests based on task type. Use the draft model for speculative
decoding to improve effective throughput.

## Implementation Priority

1. **MLX bridge** — Replace llama.cpp with MLX (biggest perf win)
2. **Neural Accelerator dispatch** — Route expert FFN through TensorOps
3. **M5 memory profiles** — Auto-detect hardware, set budgets
4. **Multi-model orchestration** — Hold multiple models hot
5. **Conversation-aware prefetching** — Pre-warm experts based on prompt analysis
