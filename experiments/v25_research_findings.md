# ExpertFlow v25 — Research Findings: Running GLM-4.5 (355B MoE) on M5 Max 128GB

## Model Facts
- **GLM-4.5**: 355B total params, 32B active per token
- **Architecture**: 92 layers, 160 routed experts + 1 shared, 8 active per token
- **BF16 size**: ~717 GB
- **MLX 4-bit**: ~185 GB (existing at `~/models/glm-4.5-4bit`)
- **GGUF IQ2_XXS**: 116 GB | **GGUF TQ1_0**: 84 GB (downloading)

## Approach Evaluation

### 1. JANG Mixed-Precision Quantization — BLOCKED
- **Status**: Real tool, `pip install jang` (v2.2.0), `pip install vmlx` (v1.3.5)
- **How it works**: 3-tier quantization — attention stays 4-8 bit, MLP/experts crushed to 2-3 bit
- **GLM-4.5 estimate**: JANG_2L profile → ~107.5 GB (fits in 128GB!)
- **Blocker**: Requires BF16 source model (717GB) for conversion. Can't re-quantize from existing 4-bit.
- **GLM architecture**: Not explicitly listed as supported, but uses standard transformer patterns
- **Pre-quantized**: No GLM-4.5 JANG models on HuggingFace
- **Action**: Could request from JANGQ-AI community, or download BF16 if disk allows

### 2. oMLX Tiered KV Cache — INSTALLED, TESTING
- **Status**: Real, installed v0.2.20 from source (`github.com/jundot/omlx`)
- **How it works**: PagedSSDCacheManager — hot blocks in RAM, cold blocks on SSD
- **TTFT improvement**: 30-90s → 1-3s on warm cache hits (agentic scenarios)
- **Integration**: Works with any MLX model, including our 4-bit GLM-4.5
- **SSD cache tested**: `PagedSSDCacheManager` works, supports `save_block`/`load_block`/`evict`
- **Key insight**: Complements ExpertFlow — oMLX handles KV cache tiering, EF handles expert offloading

### 3. llama.cpp --cpu-moe — MOST PROMISING, DOWNLOADING
- **Status**: Shipped and working in llama.cpp b8500
- **Flags**: `--cpu-moe`, `--n-cpu-moe N`, `-ot "exps=CPU"`
- **How it works**: All layers on GPU (`-ngl 999`), then expert tensors overridden to CPU compute
- **Apple Silicon nuance**: No PCIe bus transfer (unified memory) — just routing compute to CPU vs GPU
- **GLM-4.5 GGUF**: Supported since Aug 2025 (PR #14939)
- **Model options**:
  - TQ1_0 (84GB): Leaves 44GB for KV cache + OS — **downloading now**
  - IQ2_XXS (116GB): Better quality, only 12GB headroom — tight
- **Quality concern**: 1-bit quantization of 355B MoE will have significant degradation
- **Alternative**: GLM-4.5-Air (106B) at Q4 fits comfortably with much better quality

### 4. FlashMoE ML Cache Replacement — CONCEPT IMPLEMENTED
- **Status**: Paper is real (arXiv 2601.17063), but no code released
- **How it works**: Small FFN (3-layer, 128 hidden, ~113KB) predicts which cached expert to evict
- **Training**: Labels from Belady's optimal algorithm on routing traces
- **Improvement**: +21% cache hit rate over LRU, 22% faster generation
- **Implementation**: Created `ef_v25_belady_cache.py` with:
  - BeladyPredictor: 3-layer FFN for eviction scoring
  - BeladyExpertCache: replaces LRU OrderedDict eviction
  - Belady labeling: compute_belady_labels() for offline training
  - Trace collection mode for gathering routing decisions

## Benchmark Results

### GLM-4.5 355B (TQ1_0, 79GB) on M5 Max 128GB — THE MAIN EVENT

| Config | pp32 | tg10 | pp128 | tg50 |
|--------|------|------|-------|------|
| **All GPU** (`-ngl 999`) | **88 t/s** | **24 t/s** | **206 t/s** | **22 t/s** |
| **Experts→CPU** (`-ot "exps=CPU"`) | 8.5 t/s | 4.4 t/s | — | — |
| **All CPU** (`-ngl 0`) | 4.0 t/s | 2.8 t/s | — | — |

**Key finding**: TQ1_0 at 79GB fits within the M5 Max GPU allocation limit (~110GB).
All-GPU is 5.5x faster than CPU-MOE. **22 tok/s is highly usable for interactive chat.**

Comparison: danveloper/flash-moe gets 4.36 t/s on Qwen3.5-397B (M3 Max 48GB).

### llama.cpp MoE Offloading — OLMoE 7B (Q2_K, 2.4GB) on M5 Max 128GB

| Config | Prompt (pp128) | Generate (tg20) |
|--------|---------------|-----------------|
| All GPU (`-ngl 999`) | 5,908 t/s | 387 t/s |
| Experts→CPU (`-ot "exps=CPU"`) | 505 t/s | 61 t/s |
| All CPU (`-ngl 0`) | 274 t/s | 94 t/s |

**Key insight**: On Apple Silicon unified memory, `--cpu-moe` isn't about data transfer — CPU and GPU share memory. It controls which compute engine processes the tensor. For small models that fit in GPU, all-GPU wins. For GLM-4.5 (84GB TQ1_0), `--cpu-moe` is essential because it exceeds the ~90GB GPU allocation limit.

### ExpertFlow MLX — Mixtral 8x7B (4-bit, 24GB) on M5 Max 128GB

| Config | Cache Hit Rate | Steady tok/s |
|--------|---------------|--------------|
| v24 LRU (128 budget) | 54.9% | 0.641 |
| v25 Freq-weighted (128 budget) | 54.3% | 0.642 |
| v25 Belady-trace (128 budget) | 53.5% | 0.643 |

Cache policies perform similarly on Mixtral (only 8 experts). Benefits expected on GLM-4.5 (160 experts) where access patterns have more structure.

### vmlx-engine — Mixtral 8x7B (4-bit) Batch Benchmark

| Metric | Value |
|--------|-------|
| Throughput | 37 t/s (batch of 3) |
| Total throughput | 52 t/s |

This is the MLX batch generator baseline — standard inference without expert streaming.

## Recommended Strategy

### Immediate (what's viable now):
1. **llama.cpp + --cpu-moe + TQ1_0 GGUF** — test once download completes (~84GB)
2. **ExpertFlow v25 Belady cache** — test improved eviction on existing MLX model
3. **oMLX + ExpertFlow combo** — SSD KV cache tiering + expert streaming

### Short-term:
4. **JANG conversion** — if quality at 1-2 bit GGUF is poor, consider downloading BF16 model for JANG_2L conversion (107.5GB, much better quality than uniform 2-bit)

### Pragmatic alternative:
5. **GLM-4.5-Air** at Q4/Q5 — 106B total, fits comfortably in 128GB, proven 31 tok/s on M4 Max

## What's Installed & Ready

| Tool | Version | Status |
|------|---------|--------|
| llama.cpp | b8500 | Installed, `--cpu-moe` confirmed working |
| JANG | 2.2.0 | Installed, needs BF16 source for conversion |
| vmlx | 1.3.5 | Installed (JANG runtime + vmlx-engine) |
| oMLX | 0.2.20 | Installed from source, SSD cache working |
| mlx-lm | 0.31.2 | Installed in venv |

## Files Created

| File | Description |
|------|-------------|
| `experiments/ef_v25_belady_cache.py` | Belady-approximate ML cache replacement (FlashMoE-inspired) |
| `experiments/ef_v25_hybrid_engine.py` | ExpertFlow + oMLX SSD KV cache integration |
| `experiments/ef_v25_llamacpp_moe.sh` | llama.cpp MoE benchmark suite for GLM-4.5 |
| `experiments/ef_v25_llamacpp_test.sh` | Quick llama.cpp MoE test with OLMoE |
| `experiments/ef_v25_omlx_tiered.py` | oMLX tiered cache experiment |
| `experiments/ef_v25_integration.py` | Comprehensive cross-approach benchmark |

## Next Steps (when GLM-4.5 TQ1_0 GGUF download completes)

```bash
# Test 1: llama.cpp with --cpu-moe (most likely to work)
llama-bench -m ~/models/glm-4.5-gguf/GLM-4.5-UD-TQ1_0.gguf -ngl 999 -ot "exps=CPU" -p 128 -n 20

# Test 2: Full benchmark suite
bash experiments/ef_v25_llamacpp_moe.sh ~/models/glm-4.5-gguf/GLM-4.5-UD-TQ1_0.gguf

# Test 3: Compare with ExpertFlow MLX engine
source .venv/bin/activate
python3 experiments/ef_v25_integration.py
```
