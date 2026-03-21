# SpecPrefill + ExpertFlow: The Complete Apple Silicon MoE Stack

## Overview

SpecPrefill (Green, March 2026) and ExpertFlow address **different bottlenecks** in MoE inference on Apple Silicon:

| Phase | Bottleneck | Solution | Speedup |
|-------|-----------|----------|---------|
| **Prefill (TTFT)** | Compute-bound (FLOPs) | SpecPrefill — sparse prefill via draft attention scoring | 3.7–5.5× |
| **Generation (tok/s)** | Memory bandwidth-bound | ExpertFlow — smart expert caching + SSD streaming | 2–3× (projected) |

Combined, these could make 122B+ MoE models genuinely interactive on M5 Max:
- **128K prompt, Qwen3.5-122B:** 19.3 min → 3.5 min (SpecPrefill) 
- **Token generation:** +50-100% throughput (ExpertFlow expert prefetch)

## How SpecPrefill Works

### The Core Insight

During autoregressive generation, the model attends heavily to a **small fraction** of prompt tokens. If you can cheaply identify which tokens matter, you only need to prefill those into the target model.

### Algorithm (3 phases)

1. **Draft prefill** — Run the full prompt through a tiny draft model (e.g., Qwen3.5-2B, 1.4GB at 4-bit). Same tokenizer required.

2. **Lookahead attention scoring** — Run 8 autoregressive decode steps on the draft, capturing attention weights at each layer. Compute per-token importance:
   ```
   importance[i] = mean_over_steps(max_over_layers_and_heads(attention_score[i]))
   ```
   Smoothed with kernel-13 average pooling to prevent isolated-token artifacts.

3. **Sparse prefill** — Select the top 20% of tokens (in non-overlapping chunks for spatial locality). Prefill only those into the target model, preserving original RoPE position IDs so relative position encoding remains correct.

### Why Unified Memory Makes This Work

On discrete GPUs (NVIDIA), the draft model either:
- Shares VRAM with the target → reduces KV cache headroom
- Runs on CPU → PCIe transfer latency for scores

On Apple Silicon's unified memory: **zero data movement**. Draft scoring is pure FLOPs. The cost equation simplifies to:

```
Speedup = C_target / (C_draft + k × C_target)
```

Where `k` = keep fraction (0.2), `C_draft` ≪ `C_target`. When FLOP ratio is ~50× (2B vs 122B), speedup approaches `1/k = 5×`.

### Architecture-Specific Adaptations

| Model | Challenge | Solution |
|-------|-----------|----------|
| **Qwen3.5 (MoE)** | Gated queries (2× width), per-head RMSNorm | Split at midpoint, apply q_norm before RoPE |
| **Nemotron-H (hybrid)** | Only 8/88 layers have attention, no RoPE | Score from attention layers only, skip RoPE patching |
| **GPT-OSS (sliding window)** | RotatingKVCache evicts old entries | Auto-detect + augment selection with last `max_size` positions |

### Benchmarks (M2 Ultra 128GB, 20% keep)

| Model | Draft | 8K | 16K | 32K | 64K | 128K |
|-------|-------|-----|------|------|------|------|
| Qwen3.5-122B (MoE) | 2B | 3.71× | 4.11× | 4.23× | 4.50× | 5.45× |
| Qwen3.5-35B (MoE) | 4B | 1.81× | 1.86× | 1.85× | 1.84× | — |
| Nemotron-H 120B | Nano-4B | 2.10× | 2.17× | 2.19× | 2.19× | — |
| GPT-OSS 120B | 20B | 1.24× | 1.28× | — | — | — |

**Key insight:** Speedup scales with context length (superlinear at 128K) because draft scoring cost grows slower than target prefill cost. GPT-OSS gets minimal benefit because the 20B draft is too large relative to the dense 120B target.

### Quality

- **0/16 regressions** at 20% keep across needle-in-haystack, JSON, code benchmarks
- 10% keep starts getting flaky on structured output tasks
- System prompt always gets full prefill (snapshotted KV state)

### Limitations Flagged by oMLX Dev

1. **Multi-turn KV cache invalidation** — Sparse KV cache from SpecPrefill can't be persisted across turns (different token selection each time). Draft KV cache is cacheable though.
2. **System prompt preservation** — Solved: system prompt gets full prefill with snapshotted KV state. SpecPrefill only applies after the system boundary.

## Integration with ExpertFlow

### The "Speculative Stack" (3-phase inference)

```
Phase 1: SpecPrefill (TTFT)           Phase 2: ExpertFlow (Generation)     Phase 3: MTP (Throughput)
┌─────────────────────┐               ┌────────────────────────┐          ┌──────────────────┐
│ Draft model scores  │               │ Expert temp tracking   │          │ Multi-token       │
│ prompt importance   │──────────────▶│ Lookahead prefetch     │─────────▶│ prediction       │
│ Sparse prefill 20%  │               │ SSD streaming          │          │ Speculative decode│
│ → 3-5× faster TTFT  │               │ → +50-100% gen speed   │          │ → +30-60% tok/s  │
└─────────────────────┘               └────────────────────────┘          └──────────────────┘
```

### Shared Infrastructure

Both SpecPrefill and ExpertFlow benefit from:
- **Draft model as router oracle** — The same 2B draft model used for SpecPrefill scoring could inform ExpertFlow's expert prefetch predictions (the draft's MoE router activations reveal which experts the target will need)
- **MLX unified memory** — Zero-copy for both draft scoring and expert caching
- **Temperature tracking** — ExpertFlow's expert temperature could inform SpecPrefill's token selection (hot experts = hot tokens)

### Implementation Plan

1. **Phase 7 (MLX backend)** — Port ExpertFlow from llama.cpp FFI to MLX. This is a prerequisite for SpecPrefill integration since SpecPrefill is built on vllm-mlx.

2. **Phase 8 (SpecPrefill integration)** — Add `specprefill.py` as a prefill preprocessor in ExpertFlow's inference pipeline. Draft model loading/caching handled by ExpertFlow's memory manager.

3. **Phase 9 (Draft-as-oracle)** — Use draft model's MoE router activations to predict target model expert needs 1-2 layers ahead. This is the novel contribution: **SpecPrefill's draft run generates expert routing predictions as a free side-effect**.

## References

- Green, D. (2026). "SpecPrefill on Unified Memory: Cross-Architecture Sparse Prefill for LLMs on Apple Silicon." DOI: 10.5281/zenodo.19120919
- Yao et al. (2025). "SpecPrefill" (original GPU-based formulation)
- Implementation: [waybarrios/vllm-mlx PR #180](https://github.com/waybarrios/vllm-mlx/pull/180)
- Paper: [HuggingFace](https://huggingface.co/Thump604/specprefill-paper)

## Related Tools Mentioned in Discussion

- **jangq.ai** — Custom 2-3 bit MLX quantizations that beat 4-bit on MMLU
- **mlx.studio** — Paged KV cache + hybrid model support for MLX
- **oMLX** — MLX community fork with additional optimisations
