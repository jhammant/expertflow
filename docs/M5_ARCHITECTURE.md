# M5 Architecture Notes for ExpertFlow

## Sources
- [Apple ML Research: Exploring LLMs with MLX and M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [ModelFit: M5 Pro & Max Local LLM Analysis](https://modelfit.io/blog/m5-pro-max-local-llm-2026/)
- [Apple Developer: Boost Graphics Performance with M5/A19 GPUs](https://youtu.be/_5yEcJfB6nk)

## Key Architecture Decisions

### 1. Target MLX, Not llama.cpp

MLX is 20-50% faster than Ollama on Apple Silicon. ExpertFlow Phase 7 should replace the
llama.cpp FFI bridge with an MLX Python/Swift bridge:

- MLX uses Metal 4 TensorOps natively → Neural Accelerators "just work"
- MLX has native quantization support (BF16, 4-bit, MXFP4)
- mlx-community on HuggingFace has pre-quantized MoE models ready to use
- LM Studio already ships MLX backend

### 2. Two-Phase Inference Strategy

**Prompt processing (TTFT):** Compute-bound → Neural Accelerators give 3-4× speedup for free.
ExpertFlow does NOT need to optimise this phase. Let MLX/Metal handle it.

**Token generation:** Bandwidth-bound → 614 GB/s on M5 Max.
THIS is where ExpertFlow adds value:
- Expert prefetching hides SSD latency (7.4 GB/s NVMe)
- Temperature tracking keeps hot experts pinned in unified memory  
- Eviction of cold experts frees bandwidth for active computation

### 3. Memory Budget Calculations (M5 Max 128GB)

```
Model: DeepSeek V3 685B (1-bit quant) ≈ 100GB on disk
  - Router + shared layers (always resident): ~5GB
  - Expert pool: ~95GB of experts
  - With 128GB: ALL experts fit in RAM → no SSD streaming needed!
  
Model: Qwen3-235B MoE (4-bit quant) ≈ 110GB on disk  
  - Exceeds RAM by ~20GB → ExpertFlow manages overflow
  - Active experts: ~12GB per layer (3B active params)
  - At any given time: ~30GB hot, ~80GB warm/cold
  - SSD streaming budget: 7.4 GB/s → can load a cold expert (~500MB) in ~68ms

Model: Future 1T+ MoE models
  - Will exceed 128GB significantly
  - ExpertFlow's full scheduling pipeline essential
```

### 4. MoE Expert Sizes

Typical expert sizes (4-bit quantized):
- 8B model experts: ~50-100MB each
- 30B model experts: ~200-400MB each  
- 235B model experts: ~400-800MB each

At 7.4 GB/s NVMe, loading a cold expert takes:
- 100MB → 14ms
- 400MB → 54ms  
- 800MB → 108ms

With 1-2 layer lookahead prefetching, we can hide most of this latency.

### 5. Neural Engine vs GPU Neural Accelerators

These are DIFFERENT hardware:
- **GPU Neural Accelerators** (NEW in M5): In every GPU core. Matrix multiply units.
  Activated via Metal 4 TensorOps. Used by MLX automatically.
- **Neural Engine** (existing since M1): Separate 16-core unit. 38 TOPS.
  Currently used by CoreML. Could run expert FFN while GPU handles attention.

Phase 5 research: Can we dispatch expert MLP to Neural Engine while GPU
runs attention + routing? This would be true heterogeneous compute.

### 6. macOS Requirements

- macOS 26.2+ for Neural Accelerator support in Metal 4
- MLX automatically detects and uses Neural Accelerators when available
- Older macOS versions: MLX still works, just without Neural Accelerator speedup
