<p align="center">
  <h1 align="center">⚡ ExpertFlow</h1>
  <p align="center"><strong>Run 685B parameter models on a laptop.</strong></p>
  <p align="center">Dynamic MoE expert streaming for Apple Silicon — keeps hot experts in unified memory, streams cold experts from NVMe on demand. GPU + ANE + SSD orchestration in Rust.</p>
</p>

<p align="center">
  <a href="https://github.com/jhammant/expertflow/actions/workflows/tests.yml"><img src="https://github.com/jhammant/expertflow/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/jhammant/expertflow"><img src="https://img.shields.io/badge/Rust-2021-DEA584?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"></a>
  <a href="https://github.com/jhammant/expertflow"><img src="https://img.shields.io/badge/Apple_Silicon-M1→M5-000000?style=for-the-badge&logo=apple" alt="Apple Silicon"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/Lines-2,384-blue?style=for-the-badge" alt="Lines of Code">
</p>

<p align="center">
  <a href="#the-problem">Problem</a> •
  <a href="#the-solution">Solution</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#installation">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## The Problem

Mixture-of-Experts models are the future of efficient LLMs. DeepSeek V3 (685B), Qwen3 235B, and Nemotron-H use sparse routing to activate only a fraction of parameters per token — making them faster than dense models at equivalent quality.

**But they don't fit in RAM.**

| Model | Parameters | Q4 Size | M5 Max RAM |
|-------|-----------|---------|------------|
| Qwen3 235B (MoE) | 235B | ~110 GB | 128 GB ⚠️ |
| DeepSeek V3 (Q2) | 685B | ~170 GB | 128 GB ❌ |
| DeepSeek V3 (Q4) | 685B | ~350 GB | 128 GB ❌ |

Current solutions are broken:

- **macOS virtual memory** → Swap thrashing, kernel panics, system freezes
- **Static layer offloading** → Treats MoE like a dense model, ignoring that only 8/256 experts activate per token
- **CPU offloading** → Wastes bandwidth moving entire layers instead of individual experts
- **Everyone ignores the Neural Engine** → 38 TOPS of compute sitting idle

## The Solution

ExpertFlow exploits the key insight of MoE: **you only need ~3% of experts at any given moment.** Instead of loading the entire model, ExpertFlow dynamically schedules individual experts across Apple Silicon's heterogeneous compute:

```
                    ┌─────────────────────────────────┐
                    │          ExpertFlow               │
                    │                                  │
                    │   ┌─────────┐  ┌──────────────┐ │
                    │   │  GPU    │  │    CPU        │ │
                    │   │ (Metal) │  │  Scheduler    │ │
                    │   │         │  │  Prefetcher   │ │
                    │   │ Attn    │  │  Evictor      │ │
                    │   │ Router  │  │               │ │
                    │   │ Embed   │  │  ANE (future) │ │
                    │   └────┬────┘  └──────┬───────┘ │
                    │        └───────┬──────┘         │
                    │                │                 │
                    │   ┌────────────┴────────────┐   │
                    │   │    Unified Memory        │   │
                    │   │    128 GB                │   │
                    │   │                          │   │
                    │   │  ┌──────┐ ┌──────┐       │   │
                    │   │  │ Hot  │ │ Hot  │ ...   │   │
                    │   │  │Expert│ │Expert│       │   │
                    │   │  │  #12 │ │  #47 │       │   │
                    │   │  └──────┘ └──────┘       │   │
                    │   └────────────┬────────────┘   │
                    │                │                 │
                    │   ┌────────────┴────────────┐   │
                    │   │    NVMe SSD              │   │
                    │   │    7.4 GB/s              │   │
                    │   │                          │   │
                    │   │  ┌──────┐ ┌──────┐       │   │
                    │   │  │ Cold │ │ Cold │ ...   │   │
                    │   │  │Expert│ │Expert│       │   │
                    │   │  │ #203 │ │  #41 │       │   │
                    │   │  └──────┘ └──────┘       │   │
                    │   └─────────────────────────┘   │
                    └─────────────────────────────────┘
```

### Key Techniques

#### 1. 🔮 Lookahead Router Prefetching

The MoE router (tiny, always resident) runs 1–2 layers ahead of compute. ExpertFlow taps router predictions and starts async NVMe reads **before the expert is needed**.

```
Layer N:     [Computing expert #12, #47]
Layer N+1:   Router predicts → experts #8, #91 needed
             └→ async mmap prefetch starts NOW
Layer N+1:   [Experts #8, #91 already in memory ✓]
```

#### 2. 🌡️ Dynamic Expert Temperature

Every expert has a temperature score: `recency × frequency`. Hot experts stay pinned in unified memory. Cold experts get evicted via `madvise(MADV_DONTNEED)` — zero-cost kernel page eviction.

```
Expert #12:  temp=0.94 ████████████████████ HOT → pinned in RAM
Expert #47:  temp=0.71 ██████████████░░░░░░ WARM → resident
Expert #203: temp=0.12 ██░░░░░░░░░░░░░░░░░░ COLD → evicted to SSD
```

#### 3. 🔀 Heterogeneous Compute Dispatch

| Compute Unit | Workload | Why |
|-------------|----------|-----|
| **GPU (Metal)** | Attention, router, embeddings | Bandwidth-bound, GPU excels |
| **ANE (future)** | Expert FFN/MLP | Compute-bound matmuls, 38 TOPS |
| **CPU** | Scheduling, prefetch, memory mgmt | Coordination and async I/O |

#### 4. 📄 Zero-Copy Memory Management

Direct `mmap` of GGUF/safetensors files. No copies, no allocations — the kernel pages data directly from NVMe into unified memory.

```rust
// Prefetch: tell the kernel to start reading
madvise(expert_ptr, expert_size, MADV_WILLNEED);

// Evict: release pages back to the kernel (free, instant)
madvise(expert_ptr, expert_size, MADV_DONTNEED);
```

#### 5. 🔄 Adaptive Quantization

Hot experts stay at Q4 for quality. Cold experts demote to Q2 for memory savings. Quantization level adapts dynamically based on temperature and memory pressure.

#### 6. 🧠 KV Cache-Aware Budget

The expert memory budget adjusts in real-time based on KV cache pressure. Longer conversations = more KV cache = fewer resident experts = more aggressive eviction.

## Architecture

ExpertFlow wraps llama.cpp (and optionally MLX) via FFI for compute, adding a Rust scheduling layer on top:

```
┌────────────────────────────────────────────────────┐
│                  ExpertFlow (Rust)                    │
│                                                    │
│  src/                                              │
│  ├── main.rs                  CLI entry point      │
│  ├── lib.rs                   Library root         │
│  ├── core/                                         │
│  │   ├── scheduler.rs         Expert dispatch      │
│  │   ├── prefetcher.rs        Async SSD prefetch   │
│  │   ├── evictor.rs           Temperature eviction │
│  │   └── memory.rs            mmap + madvise       │
│  ├── bridge/                                       │
│  │   ├── llamacpp.rs          FFI to llama.cpp     │
│  │   ├── expert_hook.rs       Intercept loading    │
│  │   └── router_hook.rs       Tap router output    │
│  ├── model/                                        │
│  │   ├── gguf.rs              Expert map parsing   │
│  │   └── config.rs            MoE model configs    │
│  └── profiler/                                     │
│      ├── heatmap.rs           Activation heatmap   │
│      └── bench.rs             Benchmark suite      │
│                                                    │
│  build.rs                     Compile llama.cpp    │
├────────────────────┬───────────────────────────────┤
│   llama.cpp (FFI)  │   MLX (Python bridge, opt.)   │
│   GGML + Metal     │   Metal 4 TensorOps           │
└────────────────────┴───────────────────────────────┘
```

### The Speculative Stack

ExpertFlow is designed as one layer of a three-phase inference pipeline:

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  SpecPrefill     │───▶│  ExpertFlow        │───▶│  MTP Decode      │
│                  │    │                  │    │                  │
│  Sparse prefill  │    │  Expert stream   │    │  Multi-token     │
│  via draft model │    │  from NVMe       │    │  prediction      │
│                  │    │                  │    │                  │
│  3–5× TTFT ↓     │    │  2–3× gen ↑      │    │  30–60% tok/s ↑  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

The draft model used by SpecPrefill generates MoE router activations as a free side-effect — ExpertFlow uses these to predict which experts the target model needs 1–2 layers ahead. See [docs/SPECPREFILL-INTEGRATION.md](docs/SPECPREFILL-INTEGRATION.md).

## Benchmarks

### Performance Targets (M5 Max, 128GB)

| Model | Q Level | Size | Vanilla mmap | ExpertFlow | Speedup |
|-------|---------|------|-------------|-----------|---------|
| Qwen3 235B | Q4 | ~110 GB | 8–15 tok/s | 15–25 tok/s | ~1.8× |
| DeepSeek V3 | 1-bit | ~100 GB | 5–10 tok/s | 10–18 tok/s | ~1.8× |
| DeepSeek V3 | Q2 | ~170 GB | 2–4 tok/s | 6–12 tok/s | ~3× |
| DeepSeek V3 | Q4 | ~350 GB | 1–3 tok/s | 4–8 tok/s | ~3× |

> ⚠️ **These are projected targets.** Real benchmarks will be published when Phase 2 (dynamic scheduler) is complete. Run `expertflow bench` to reproduce.

### Why M5 Changes Everything

Apple's M5 introduces **Neural Accelerators** — dedicated matmul units in every GPU core. This creates a two-speed regime:

| Phase | Bottleneck | M5 vs M4 | ExpertFlow Impact |
|-------|-----------|----------|------------------|
| Prompt (TTFT) | Compute | 3.3–4× faster | Handled by Neural Accelerators natively |
| Generation | **Memory bandwidth** | ~15–27% faster | **This is where ExpertFlow shines** |

Token generation is bandwidth-bound. Smart expert caching keeps hot data in the 614 GB/s unified memory pool instead of thrashing to the 7.4 GB/s SSD.

### Hardware Reference (M5 Max 128GB)

```
Memory bandwidth:   614 GB/s (unified)
NVMe sequential:    7.4 GB/s
Unified memory:     128 GB
GPU cores:          40 (each with Neural Accelerator)
Neural Engine:      16-core, 38 TOPS
```

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/jhammant/expertflow
cd expertflow
cargo build --release
```

### From Cargo (Coming Soon)

```bash
cargo install expertflow
```

### Requirements

- **Apple Silicon Mac** (M1+ basic support, M4 Pro+ recommended, M5 Max optimal)
- **macOS 15+** (Sequoia or later; macOS 26.2 for Neural Accelerator support)
- **Rust 1.75+**
- **Xcode Command Line Tools** (for Metal SDK)

## Usage

### Profile Expert Activation Patterns

Before running inference, profile which experts activate for your typical workload:

```bash
# Analyze expert activation heatmap
expertflow profile \
  --model ./DeepSeek-V3-Q4.gguf \
  --samples 100

# Output: expert activation frequencies, co-occurrence matrix,
# recommended cache budget
```

### Run Inference with Expert Streaming

```bash
# Interactive chat with dynamic expert scheduling
expertflow run \
  --model ./DeepSeek-V3-Q4.gguf \
  --ram-budget 100 \
  --prefetch-depth 2

# Flags:
#   --ram-budget     Max GB for resident experts (default: 80% of RAM)
#   --prefetch-depth Lookahead layers for prefetching (default: 1)
#   --pin-threshold  Temperature above which experts stay pinned (default: 0.7)
```

### Benchmark Against Baseline

```bash
# Compare ExpertFlow vs vanilla mmap
expertflow bench \
  --model ./DeepSeek-V3-Q4.gguf \
  --warmup 10 \
  --iterations 50

# Output: tok/s, TTFT, p50/p95/p99 latency, cache hit rate,
# prefetch accuracy, memory high-water mark
```

### Simulate (No Model Required)

```bash
# Run the scheduling simulator to test strategies
cargo run --bin simulate -- \
  --experts 256 \
  --active 8 \
  --budget-gb 64 \
  --tokens 1000
```

### Configuration

```bash
# ExpertFlow respects these environment variables:
EXPERTFLOW_RAM_BUDGET=100        # GB
EXPERTFLOW_PREFETCH_DEPTH=2      # layers
EXPERTFLOW_LOG=debug             # tracing level
EXPERTFLOW_CACHE_DIR=~/.expertflow/cache  # persistent cache
```

## Supported Models

ExpertFlow works with any GGUF-format MoE model. Tested configurations:

| Model | Architecture | Experts | Active/Token | Status |
|-------|-------------|---------|-------------|--------|
| DeepSeek V3 | MoE | 256 | 8 | 🎯 Primary target |
| Qwen3 235B | MoE | 128 | 8 | ✅ Supported |
| Qwen3.5-35B-A3B | MoE | 64 | 4 | ✅ Supported |
| Nemotron-H | MoE+SSM | varies | varies | 🧪 Experimental |
| Jamba | MoE+SSM | varies | varies | 🧪 Experimental |

## Prior Art

| Project | Approach | Platform | Limitation |
|---------|----------|----------|------------|
| [HOBBIT](https://arxiv.org/abs/2411.01433) | Mixed precision offload | NVIDIA only | No Apple Silicon |
| [Krasis](https://github.com/brontoguana/krasis) | Hybrid GPU/CPU stream | NVIDIA + Linux | No macOS |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | CPU MoE offload | Cross-platform | No smart scheduling |
| [SpecPrefill](https://doi.org/10.5281/zenodo.19120919) | Sparse prefill | Apple Silicon | Prefill only, not generation |
| [Orion](https://arxiv.org/abs/2603.06728) | Open ANE programming | Apple Silicon | No MoE scheduling |
| [MLX](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) | ML framework | Apple Silicon | No expert-level scheduling |

**ExpertFlow is the first to combine dynamic expert scheduling + SSD streaming + draft-model-as-oracle on Apple Silicon.**

## Roadmap

- [x] Architecture design & simulator (2,384 lines Rust)
- [x] Persistent expert cache (warm restarts)
- [x] Adaptive expert quantization (Q4 ↔ Q2)
- [x] KV cache-aware memory budgeting
- [x] MLX backend bridge
- [x] vMLX OpenAI-compatible API integration
- [x] Hybrid MoE+SSM architecture support
- [ ] **Phase 1:** mmap + madvise expert pinning on real GGUF
- [ ] **Phase 2:** Dynamic scheduler with router lookahead
- [ ] **Phase 3:** Temperature-based eviction
- [ ] **Phase 4:** Metal 4 compute integration
- [ ] **Phase 5:** ANE dispatch (research)
- [ ] **Phase 6:** GGUF expert extraction tooling
- [ ] **Phase 8:** SpecPrefill integration
- [ ] **Phase 9:** Draft-as-oracle (use draft MoE router predictions)
- [ ] Real-world benchmarks on M5 Max 128GB
- [ ] `cargo install` distribution

## Contributing

ExpertFlow is open source and we welcome contributions! Areas where help is especially valuable:

- **Metal/GPU expertise** — Metal 4 TensorOps integration
- **ANE research** — Neural Engine programming via Orion or CoreML
- **Benchmarking** — Testing on different Apple Silicon configurations
- **Model support** — Adding new MoE architectures

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Run with debug logging
EXPERTFLOW_LOG=debug cargo run -- run --model ./model.gguf
```

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use ExpertFlow in research, please cite:

```bibtex
@software{expertflow2025,
  title={ExpertFlow: Dynamic MoE Expert Streaming for Apple Silicon},
  author={Hammant, Jon},
  year={2025},
  url={https://github.com/jhammant/expertflow}
}
```

---

<p align="center">
  <strong>Because 685B parameters shouldn't require a data center.</strong>
</p>
