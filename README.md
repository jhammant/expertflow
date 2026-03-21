# ⚡ ExpertFlow

**Dynamic MoE expert streaming for Apple Silicon — run 685B parameter models on a laptop.**

Rust-native heterogeneous compute scheduler. Keeps hot experts in unified memory, streams cold experts from NVMe on demand. GPU + ANE + SSD orchestration.

## The Problem

Mixture-of-Experts models like DeepSeek V3 (685B) and Qwen3 235B are too large for RAM. Current solutions either:
- Rely on macOS swap → kernel panics, system freezes
- Use static layer offloading → misses the dynamic nature of MoE routing  
- Ignore the Neural Engine → 38 TOPS of compute sitting idle

## The Solution

ExpertFlow dynamically schedules expert computation across all Apple Silicon compute units:

```
┌─────────────────────────────────────────┐
│               ExpertFlow                  │
│                                         │
│  GPU (Metal)     ANE (future)   CPU     │
│  ┌───────────┐  ┌──────────┐  ┌─────┐  │
│  │ Attention  │  │ Expert   │  │Sched│  │
│  │ Router     │  │ FFN/MLP  │  │Fetch│  │
│  │ Embeddings │  │ Compute  │  │Evict│  │
│  └─────┬─────┘  └────┬─────┘  └──┬──┘  │
│        └──────────────┴───────────┘     │
│           Unified Memory (128GB)        │
│                   │                     │
│           ┌───────┴───────┐             │
│           │  NVMe SSD     │             │
│           │  7.4 GB/s     │             │
│           │  Cold Experts │             │
│           └───────────────┘             │
└─────────────────────────────────────────┘
```

## Key Techniques

### 1. Lookahead Router Prefetching
The MoE router (tiny, always resident) predicts which experts are needed 1-2 layers ahead. Async SSD reads start BEFORE the expert is needed.

### 2. Dynamic Expert Temperature Tracking
Every expert has a temperature score (recency × frequency). Hot experts stay pinned. Cold experts get evicted via `madvise(MADV_DONTNEED)`.

### 3. Heterogeneous Compute Dispatch
- **GPU (Metal):** Attention, router, embeddings (bandwidth-bound)
- **ANE (future):** Expert FFN/MLP (compute-bound matrix multiply)
- **CPU:** Scheduling, async prefetch, memory management

### 4. Zero-Copy Memory Management
Direct `mmap` of GGUF/safetensors files. `madvise(MADV_WILLNEED)` for prefetch, `MADV_DONTNEED` for eviction. No copies — the kernel pages directly from NVMe into unified memory.

## Performance Targets (M5 Max, 128GB)

| Model | Params | Q4 Size | Vanilla mmap | ExpertFlow |
|-------|--------|---------|--------------|----------|
| Qwen3 235B (MoE) | 235B | ~110GB | 8-15 tok/s | 15-25 tok/s |
| DeepSeek V3 (1-bit) | 685B | ~100GB | 5-10 tok/s | 10-18 tok/s |
| DeepSeek V3 (Q2) | 685B | ~170GB | 2-4 tok/s | 6-12 tok/s |
| DeepSeek V3 (Q4) | 685B | ~350GB | 1-3 tok/s | 4-8 tok/s |

## Architecture

ExpertFlow wraps llama.cpp via FFI — we handle scheduling, llama.cpp handles compute.

```
┌──────────────────────────────────────┐
│           ExpertFlow (Rust)            │
│                                      │
│  src/                                │
│  ├── main.rs              # CLI      │
│  ├── lib.rs               # Library  │
│  ├── core/                           │
│  │   ├── scheduler.rs     # Expert scheduling engine    │
│  │   ├── prefetcher.rs    # Async SSD prefetch          │
│  │   ├── evictor.rs       # Temperature eviction        │
│  │   └── memory.rs        # mmap + madvise manager      │
│  ├── bridge/                         │
│  │   ├── llamacpp.rs      # FFI to llama.cpp            │
│  │   ├── expert_hook.rs   # Intercept expert loading    │
│  │   └── router_hook.rs   # Tap router predictions      │
│  ├── model/                          │
│  │   ├── gguf.rs          # GGUF expert map extraction  │
│  │   └── config.rs        # MoE model configs           │
│  └── profiler/                       │
│      ├── heatmap.rs       # Expert activation heatmap   │
│      └── bench.rs         # Benchmark suite             │
│                                      │
│  Cargo.toml                          │
│  build.rs          # Compile llama.cpp as dep           │
└──────────┬───────────────────────────┘
           │ FFI (C bindings)
┌──────────┴───────────────────────────┐
│   llama.cpp (git submodule)          │
│   GGML + Metal + MoE inference       │
└──────────────────────────────────────┘
```

## Quick Start

```bash
cargo install expertflow

# Profile expert activation patterns
expertflow profile --model ./DeepSeek-V3-Q4.gguf --samples 100

# Run inference with dynamic expert streaming  
expertflow run --model ./DeepSeek-V3-Q4.gguf --ram-budget 100

# Benchmark against vanilla mmap baseline
expertflow bench --model ./DeepSeek-V3-Q4.gguf
```

## Building

```bash
git clone https://github.com/jhammant/expertflow
cd expertflow
cargo build --release
```

## Requirements

- Apple Silicon Mac (M1+ basic, M5 Max recommended)
- macOS 15+ (Sequoia or later)
- Rust 1.75+ 
- Xcode Command Line Tools (for Metal SDK)

## M5 GPU Architecture Insights

ExpertFlow is designed around the specific hardware characteristics of Apple's M5 chips, informed by [Apple's MLX research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) and independent benchmarks.

### Why M5 Changes Everything for MoE

The M5 introduces **Neural Accelerators** — dedicated matrix-multiplication units embedded in every GPU core (40 on M5 Max). This creates a two-speed inference regime:

| Phase | Bottleneck | M5 vs M4 | ExpertFlow Strategy |
|-------|-----------|----------|-------------------|
| Prompt processing (TTFT) | **Compute** | **3.3–4× faster** | Neural Accelerators handle this natively — no ExpertFlow intervention needed |
| Token generation | **Memory bandwidth** | ~15–27% faster | **This is where ExpertFlow shines** — smart prefetch/eviction keeps hot experts in unified memory |

### Key Hardware Numbers (M5 Max 128GB)

- **Memory bandwidth:** 614 GB/s (vs 546 GB/s M4 Max)
- **NVMe read:** ~7.4 GB/s (sequential)
- **Unified memory:** 128GB — fits 70B Q4 (~40GB) with 88GB to spare
- **GPU cores:** 40, each with a Neural Accelerator
- **Neural Engine:** 16-core, 38 TOPS (separate from GPU Neural Accelerators)

### Design Implications

1. **MLX over llama.cpp** — Apple's [MLX framework](https://mlx-framework.org) is 20–50% faster than Ollama/llama.cpp on Apple Silicon. ExpertFlow should target MLX as the primary compute backend via Metal 4 TensorOps.

2. **Focus on token generation** — TTFT is already 3-4× faster with Neural Accelerators. The real win is in sustained generation, where bandwidth is the bottleneck and smart expert caching provides the most uplift.

3. **MoE sweet spot** — Qwen3-30B (3B active params, 4-bit quantized) uses only 17GB and gets TTFT under 3 seconds on M5. With 128GB, ExpertFlow can keep *all* experts for multiple MoE layers resident simultaneously.

4. **Heterogeneous dispatch opportunity** — GPU Neural Accelerators handle dense matmuls (attention, embeddings). The separate 16-core Neural Engine could potentially run expert FFN/MLP in parallel. This is Phase 5 (research).

5. **macOS 26.2 required** — Neural Accelerator support needs the latest macOS beta for Metal 4 TensorOps.

### MLX Performance Reference (MacBook Pro M5, 24GB)

| Model | TTFT Speedup vs M4 | Gen Speedup | Memory |
|-------|-------------------|-------------|--------|
| Qwen3-1.7B BF16 | 3.57× | 1.27× | 4.4 GB |
| Qwen3-8B BF16 | 3.62× | 1.24× | 17.5 GB |
| Qwen3-14B 4-bit | 4.06× | 1.19× | 9.2 GB |
| Qwen3-30B MoE 4-bit | 3.52× | 1.25× | 17.3 GB |
| GPT-OSS-20B MXFP4 | 3.33× | 1.24× | 12.1 GB |

*Source: [Apple ML Research, March 2026](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)*

## The Speculative Stack Vision

ExpertFlow is designed to be one layer of a three-phase inference pipeline for interactive MoE on Apple Silicon:

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  SpecPrefill     │───▶│  ExpertFlow        │───▶│  MTP Decode      │
│  Sparse prefill  │    │  Expert stream   │    │  Multi-token     │
│  3-5× TTFT ↓     │    │  2-3× gen ↑      │    │  30-60% tok/s ↑  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

- **[SpecPrefill](https://doi.org/10.5281/zenodo.19120919)** (Green 2026) — Uses a 2B draft model to score prompt token importance via attention, sparse-prefills only 20% into the target. 3.7–5.5× TTFT reduction on Qwen3.5-122B (M2 Ultra). Built on vllm-mlx.
- **ExpertFlow** — Dynamic expert caching during generation. Hot experts stay resident, cold experts stream from NVMe.
- **MTP** — Multi-Token Prediction speculative decoding for throughput.

The draft model used by SpecPrefill generates MoE router activations as a free side-effect — ExpertFlow can use these to predict which experts the target model will need 1-2 layers ahead. See [docs/SPECPREFILL-INTEGRATION.md](docs/SPECPREFILL-INTEGRATION.md).

## Prior Art

- **[SpecPrefill](https://doi.org/10.5281/zenodo.19120919)** — Attention-based sparse prefill. 3.7–5.5× TTFT on M2 Ultra. Complementary to ExpertFlow (handles prefill, not generation).
- **[HOBBIT](https://arxiv.org/abs/2411.01433)** — Mixed precision expert offloading, 9.93× speedup. NVIDIA only.
- **[Krasis](https://github.com/brontoguana/krasis)** — Hybrid GPU/CPU streaming. NVIDIA + Linux only.
- **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** — CPU-side MoE offloading. Basic, no smart scheduling.
- **[Orion](https://arxiv.org/abs/2603.06728)** — First open ANE programming system for transformers.
- **[MLX](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)** — Apple's ML framework with M5 Neural Accelerators + Metal 4 TensorOps.
- **[jangq.ai](https://jangq.ai)** — Custom 2-3 bit MLX quantisations that beat 4-bit on MMLU.
- **[mlx.studio](https://mlx.studio)** — Paged KV cache + hybrid model support for MLX.

**ExpertFlow is the first to combine dynamic expert scheduling + SSD streaming + draft-model-as-oracle on Apple Silicon.**

## Roadmap

- [x] Architecture design
- [x] Scheduler, prefetcher, evictor, simulator (2,384 lines Rust)
- [ ] Phase 1: mmap + madvise expert pinning/prefetch on real GGUF
- [ ] Phase 2: Dynamic scheduler with router lookahead
- [ ] Phase 3: Temperature-based eviction
- [ ] Phase 4: Metal 4 compute integration (TensorOps + Neural Accelerators)
- [ ] Phase 5: ANE dispatch (research — requires macOS 26.2)
- [ ] Phase 6: GGUF expert extraction
- [x] **Phase 7: MLX backend** — Python bridge + JSON-over-stdio protocol, Metal 4 TensorOps placeholder
- [ ] Phase 8: SpecPrefill integration (sparse prefill via draft attention scoring)
- [ ] Phase 9: Draft-as-oracle (use draft MoE router to predict target expert needs)
- [x] **Phase 10: Persistent expert cache** — Disk-backed cache for warm restarts (~/.expertflow/cache.json)
- [x] **Phase 11: Adaptive expert quantization** — Hot experts stay Q4, cold experts demote to Q2 for memory savings
- [x] **Phase 12: KV cache-aware memory budget** — Dynamic expert budget adjusts based on KV cache pressure
- [x] **Phase 13: vMLX bridge** — OpenAI-compatible API integration with vMLX serving framework
- [x] **Phase 14: Hybrid architecture support** — MoE+SSM models (Nemotron-H, Jamba, GatedDeltaNet)
- [ ] Benchmarks on M5 Max 128GB

## License

MIT
