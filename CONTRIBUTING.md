# Contributing to ExpertFlow

Thank you for your interest in contributing! This document covers how to set up a development environment and run the test suite.

## Prerequisites

- macOS with Apple Silicon (M1–M5) — required for unified memory architecture
- Python 3.11 or later
- Rust 1.75+ (`rustup update stable`)
- NVMe storage with ≥500 GB free for model weights (for real inference only)

## Python setup

```bash
# Clone the repo
git clone https://github.com/jhammant/expertflow.git
cd expertflow

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pytest
```

## Running the tests

```bash
# Activate your venv first
source .venv/bin/activate

# Run the full suite (84 tests, ~1s)
pytest tests/ -v

# Run a specific file
pytest tests/test_kv_cache.py -v

# Run a specific test
pytest tests/test_gguf_loader.py::TestTensorByteSize::test_q2k_single_block -v
```

All tests run without model weights — they exercise the caching and routing subsystems using simulated data.

## Running the benchmark

```bash
# Quick smoke test (small simulated config, ~5s)
python benchmark_integrated.py --model small --turns 3

# Full DeepSeek V3 simulation (no weights needed)
python benchmark_integrated.py --model deepseek --turns 5

# Compare all eviction policies
python benchmark_integrated.py --model small --compare-policies

# See all options
python benchmark_integrated.py --help
```

## Rust build

```bash
cargo build --release
cargo test
```

## Code style

- Python: follow PEP 8, use type hints on public APIs
- Rust: run `cargo fmt` and `cargo clippy` before committing
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

## Submitting changes

1. Fork the repo and create a feature branch: `feat/my-feature`
2. Make sure `pytest tests/` passes
3. Open a pull request with a clear description of what changed and why

## Project layout

```text
ef_kv_manager.py          — Paged KV cache + budget coordinator
ef_integrated_engine.py   — End-to-end inference engine (simulated)
ef_deepseek_mmap.py       — GGUF parser + expert mmap loader
benchmark_integrated.py   — Multi-turn benchmark CLI
tests/                    — pytest suite
src/                      — Rust kernel (memory guard, mmap helpers)
```
