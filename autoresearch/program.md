# ExpertFlow AutoResearch Program

## Goal
Optimize ExpertFlow inference to:
1. Complete 100% of layers (currently: GLM 91%, DeepSeek 49%)
2. Generate coherent tokens from massive MoE models
3. Minimize time per layer
4. Keep memory under 25GB

## Setup
```bash
source ~/mlx-env/bin/activate
cd ~/dev/expertflow/autoresearch
```

## Run an experiment
```bash
python benchmark.py --model glm
```

## What to modify
Only modify `expertflow_opt.py`. The benchmark is fixed.

## Current bottlenecks
1. **Metal GPU timeout**: CPU fallback works but is slow
2. **Memory pressure**: Process dies around L80-84 on GLM
3. **I/O bound**: Loading expert weights from NVMe dominates time

## Key architecture facts
- DeepSeek V3.1: 61 layers, 256 experts, 8 active, SwiGLU, 7168 hidden
- GLM-4.5: 92 layers, 160 experts, 8 active, SwiGLU
- Expert weights: 4-bit quantized, packed [num_experts, out_dim, packed_in]
- Each expert weight: ~28MB dequantized (float16)
- NVMe bandwidth: 14-20 GB/s measured

## Optimization ideas to try
- [ ] Batch multiple experts into single matmul ops
- [ ] Pre-fetch next layer's expert weights while computing current layer
- [ ] Use GPU for dense layers, CPU only for MoE layers
- [ ] Reduce eval() calls (fewer sync barriers)
- [ ] Process attention on GPU, MoE on CPU (hybrid)
- [ ] Cache frequently-used experts across tokens
- [ ] Use lower precision for intermediate computations
- [ ] Parallelize expert loading with computation
- [ ] Skip eval for non-MoE layers (let graph grow)

## Rules
- Keep memory under 25GB peak
- Must work with both GLM and DeepSeek models
- Track ALL experiments in experiments.jsonl
- Each experiment: modify ONE thing, measure, record
