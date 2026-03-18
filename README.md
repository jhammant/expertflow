# ⚡ FlashMoE

**Dynamic MoE expert streaming for Apple Silicon — run 685B parameter models on a laptop.**

GPU + ANE + SSD offloading via MLX. Heterogeneous compute scheduling that keeps hot experts in memory and streams cold experts from NVMe on demand.

## The Problem

Mixture-of-Experts (MoE) models like DeepSeek V3 (685B) and Qwen3 235B are too large to fit entirely in RAM, even on high-end machines. Current solutions either:
- Rely on macOS swap (causes kernel panics, system freezes)
- Use static layer offloading (misses the dynamic nature of MoE routing)
- Ignore the Neural Engine entirely (leaving 38 TOPS of compute idle)

## The Solution

FlashMoE dynamically schedules expert computation across all three Apple Silicon compute units:

```
┌─────────────────────────────────────────┐
│               FlashMoE                │
│                                         │
│  GPU (Metal)     ANE (Orion)    CPU     │
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
The MoE router network (tiny, always in RAM) predicts which experts are needed 1-2 layers ahead. We start async SSD reads BEFORE the expert is needed.

### 2. Dynamic Expert Temperature Tracking
Every expert has a "temperature" score based on recency and frequency of activation. Hot experts stay pinned in RAM. Cold experts get evicted to make room.

### 3. Heterogeneous Compute Dispatch
- **GPU:** Attention layers, router, embeddings (bandwidth-bound ops)
- **ANE:** Expert FFN/MLP computation (compute-bound matrix multiplies — perfect for ANE's 38 TOPS)
- **CPU:** Scheduling, async SSD prefetch, memory management

### 4. Adaptive Eviction Policy
When RAM fills up, evict the coldest expert. Track per-expert activation patterns and adapt to the conversation — code generation activates different experts than creative writing.

## Performance Targets (M5 Max, 128GB RAM)

| Model | Params | Q4 Size | Without HotSwap | With HotSwap |
|-------|--------|---------|-----------------|--------------|
| Qwen3 235B (MoE) | 235B | ~110GB | 8-15 tok/s | 15-25 tok/s |
| DeepSeek V3 (1-bit) | 685B | ~100GB | 5-10 tok/s | 10-18 tok/s |
| DeepSeek V3 (Q2) | 685B | ~170GB | 2-4 tok/s | 6-12 tok/s |
| DeepSeek V3 (Q4) | 685B | ~350GB | 1-3 tok/s | 4-8 tok/s |
| Llama 3.1 405B | 405B | ~230GB | 2-5 tok/s | N/A (dense) |

*MoE models benefit most — dense models like Llama 405B can't use expert-level scheduling.*

## Prior Art

- **HOBBIT** (arxiv 2411.01433) — Mixed precision expert offloading, 9.93x speedup. NVIDIA only.
- **Krasis** (brontoguana) — Hybrid GPU/CPU streaming runtime. NVIDIA + Linux only.
- **ik_llama.cpp** — CPU-side MoE offloading fork. Basic, no smart scheduling.
- **Orion** (arxiv 2603.06728) — First open ANE programming system. Proves ANE is viable for transformers.
- **MLX Neural Accelerators** (Apple, 2026) — M5 GPU neural accelerators for matrix multiply.

**FlashMoE is the first to combine dynamic expert scheduling + ANE compute + SSD streaming on Apple Silicon.**

## Architecture

```
flashmoe/
├── core/
│   ├── scheduler.py      # Dynamic expert scheduling engine
│   ├── prefetcher.py     # Async SSD prefetch manager
│   ├── evictor.py        # Temperature-based eviction policy
│   └── memory.py         # Unified memory manager (madvise)
├── compute/
│   ├── gpu.py            # Metal/MLX GPU dispatch
│   ├── ane.py            # ANE dispatch (via Orion or CoreML)
│   └── router.py         # MoE router lookahead
├── profiler/
│   ├── heatmap.py        # Expert activation heatmap generator
│   ├── benchmark.py      # Performance benchmarking suite
│   └── visualise.py      # Real-time expert flow visualisation
├── models/
│   ├── deepseek_v3.py    # DeepSeek V3 MoE config
│   ├── qwen3_235b.py     # Qwen3 235B MoE config
│   └── base.py           # Base MoE model interface
└── cli/
    ├── run.py            # Main inference CLI
    ├── profile.py        # Profile expert activation patterns
    └── bench.py          # Benchmark suite
```

## Quick Start

```bash
pip install flashmoe

# Profile expert activation patterns on a model
flashmoe profile --model deepseek-v3-q4 --samples 100

# Run inference with dynamic expert streaming
flashmoe run --model deepseek-v3-q4 --ram-budget 100GB

# Benchmark against vanilla MLX/llama.cpp
flashmoe bench --model deepseek-v3-q4 --compare mlx,llamacpp
```

## Requirements

- Apple Silicon Mac (M1+ for basic, M5 Max recommended for large models)
- macOS 15+ (Sequoia or later)
- Python 3.11+
- MLX 0.20+

## Status

🚧 **Pre-alpha** — Architecture defined, building core scheduler. Hardware testing begins March 23 on M5 Max.

## References

- [HOBBIT: Mixed Precision Expert Offloading](https://arxiv.org/abs/2411.01433)
- [Orion: Programming Apple's Neural Engine](https://arxiv.org/abs/2603.06728)
- [MLX Neural Accelerators on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Krasis: Hybrid LLM Runtime](https://github.com/brontoguana/krasis)
- [llama.cpp MoE Offloading](https://github.com/ggml-org/llama.cpp/issues/19825)

## License

MIT
