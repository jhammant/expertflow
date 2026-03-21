# ExpertFlow Academic Paper

This directory contains the academic paper for ExpertFlow, written in LaTeX for submission to arXiv and ML systems conferences.

## Files

- **expertflow.tex** — Main LaTeX source (complete paper, ~10 pages)
- **compile.sh** — Compilation script (generates PDF)

## Compiling

### Prerequisites

Install LaTeX distribution:

**macOS:**
```bash
brew install --cask mactex-no-gui
```

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended
```

### Build

```bash
./compile.sh
```

This will generate `expertflow.pdf` in the same directory.

### Manual Compilation

If the script doesn't work, compile manually:

```bash
pdflatex expertflow.tex
pdflatex expertflow.tex  # Second pass for references
pdflatex expertflow.tex  # Third pass for cross-references
```

## Paper Structure

The paper follows standard ML systems conference format (similar to SOSP, OSDI, USENIX ATC):

1. **Abstract** — Problem, solution, key results
2. **Introduction** — MoE scaling challenge, unified memory opportunity, contributions
3. **Background & Related Work** — HOBBIT, Krasis, SpecPrefill, MLX, Orion
4. **System Design** — Four key techniques + scheduling algorithm
5. **The Speculative Stack** — Three-phase inference pipeline (SpecPrefill → ExpertFlow → MTP)
6. **Implementation** — 3,255 lines Rust/Python, MLX backend, Metal 4 TensorOps
7. **Projected Performance** — M5 Max 128GB, 2-3× speedup projections (NOT empirical benchmarks)
8. **Discussion** — Routing predictability, ANE dispatch, limitations
9. **Conclusion**
10. **References** — 9 citations (SpecPrefill, HOBBIT, Krasis, MLX, DeepSeek V3, Qwen3, etc.)

## Key Features

- **Professional academic tone** — NOT marketing copy
- **Honest about limitations** — Clearly labeled as projections, no access to M5 Max yet
- **Complete architecture diagram** — TikZ-based system diagram (Figure 1)
- **Formal algorithm** — Pseudocode for scheduling algorithm (Algorithm 1)
- **Performance tables** — Projected throughput + memory budget analysis
- **Proper citations** — Inline bibliography with 9 references

## Submission Targets

Potential venues (once empirical benchmarks are available):

- **USENIX ATC** (Annual Technical Conference) — Systems track
- **SOSP** (Symposium on Operating Systems Principles) — Memory management track
- **MLSys** (Conference on Machine Learning and Systems) — Inference systems track
- **arXiv** — Preprint server (can submit immediately)

## License

Same as parent project (MIT).
