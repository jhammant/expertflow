#!/usr/bin/env python3
"""
ExpertFlow Integrated Benchmark — Multi-Turn MoE Conversation Simulation
=========================================================================

Benchmarks the full ExpertFlow pipeline:
  - Expert weight caching with NVMe-simulated loading
  - Paged KV cache with block-level eviction
  - Dynamic budget rebalancing between KV and expert caches
  - Simulated MoE routing with Zipf-distributed expert popularity

Simulates a realistic multi-turn conversation with growing context,
measuring at each turn:
  - Expert cache hit rate (target: 85%+)
  - KV cache memory usage and eviction count
  - Estimated tokens/sec for decode
  - Budget coordinator rebalancing decisions
  - NVMe I/O volume

Usage:
  python benchmark_integrated.py [--model deepseek|mixtral|small] [--turns 5]

Output: JSON file with per-turn and aggregate statistics.
"""

import argparse
import json
import os
import sys
import time

try:
    import numpy as np
except ImportError:
    print(
        "Error: numpy is required but not installed.\n"
        "Install it with:  pip install numpy\n"
        "Or inside a venv: pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

__version__ = "0.1.0"


def _positive_int(name):
    """Return an argparse type checker that requires a positive integer."""
    def check(value):
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--{name} must be an integer, got: {value!r}"
            )
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f"--{name} must be > 0, got {ivalue}"
            )
        return ivalue
    return check


def _output_path(value):
    """Validate that the output path is writable."""
    directory = os.path.dirname(os.path.abspath(value))
    if not os.path.isdir(directory):
        raise argparse.ArgumentTypeError(
            f"--output directory does not exist: {directory}"
        )
    if os.path.exists(value) and not os.access(value, os.W_OK):
        raise argparse.ArgumentTypeError(
            f"--output file exists but is not writable: {value}"
        )
    return value

try:
    from ef_integrated_engine import (
        ExpertFlowEngine,
        ModelConfig,
        DEEPSEEK_V3_CONFIG,
        DEEPSEEK_V3_SMALL_SIM,
        MIXTRAL_8X7B_CONFIG,
        EvictionPolicy,
    )
except ImportError as e:
    print(
        f"Error: could not import ExpertFlow engine modules: {e}\n"
        "Make sure you are running from the project root directory\n"
        "and that all dependencies are installed (pip install numpy).",
        file=sys.stderr,
    )
    sys.exit(1)


def format_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    elif b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def run_benchmark(config: ModelConfig, n_turns: int = 5,
                  prompt_tokens: int = 50, response_tokens: int = 100,
                  eviction_policy: EvictionPolicy = EvictionPolicy.FREQUENCY_WEIGHTED,
                  rebalance_every: int = 32,
                  verbose: bool = True) -> dict:
    """Run the full multi-turn benchmark.

    Args:
        config: Model configuration to simulate.
        n_turns: Number of conversation turns.
        prompt_tokens: User prompt tokens per turn.
        response_tokens: Model response tokens per turn.
        eviction_policy: KV cache eviction policy.
        rebalance_every: Rebalance budgets every N decode tokens.
        verbose: Print progress during benchmark.

    Returns:
        Full benchmark results dict.
    """
    if verbose:
        print("=" * 70)
        print(f"  ExpertFlow Integrated Benchmark")
        print("=" * 70)
        print(f"  Model:           {config.name}")
        print(f"  Layers:          {config.n_layers} ({config.n_moe_layers} MoE)")
        print(f"  Experts:         {config.n_experts} per layer, {config.n_active_experts} active")
        print(f"  Total RAM:       {config.total_ram_gb} GB")
        print(f"  Available:       {config.available_budget_gb} GB (KV + expert cache)")
        print(f"  Turns:           {n_turns}")
        print(f"  Per turn:        {prompt_tokens} prompt + {response_tokens} decode tokens")
        print(f"  KV eviction:     {eviction_policy.value}")
        print(f"  Rebalance every: {rebalance_every} tokens")
        print("-" * 70)

    engine = ExpertFlowEngine(
        config=config,
        kv_eviction_policy=eviction_policy,
        initial_kv_fraction=0.15,
    )

    if verbose:
        print(f"  Expert cache:    {engine.expert_cache.budget} slots "
              f"({format_bytes(engine.expert_cache.budget * engine.expert_store.expert_bytes)})")
        print(f"  KV budget:       {format_bytes(engine.coordinator.kv_budget_bytes)}")
        print(f"  Expert budget:   {format_bytes(engine.coordinator.expert_budget_bytes)}")
        print(f"  Expert size:     {format_bytes(engine.expert_store.expert_bytes)}")
        print(f"  KV/tok (all L):  {format_bytes(engine.kv_cache.bytes_per_token_all_layers())}")
        max_tok = engine.kv_cache.max_tokens_in_budget()
        print(f"  Max KV tokens:   {max_tok:,}")
        print("-" * 70)

    turn_results = []
    benchmark_start = time.monotonic()

    for turn in range(n_turns):
        turn_start = time.monotonic()

        if verbose:
            print(f"\n  Turn {turn + 1}/{n_turns}: "
                  f"{prompt_tokens} prompt + {response_tokens} decode tokens")

        result = engine.generate_turn(
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            rebalance_every=rebalance_every,
        )

        # Strip per-token details from turn results to keep output manageable
        token_details = result.pop("token_details")
        turn_time = time.monotonic() - turn_start

        # Compute per-turn hit rate from token details
        decode_details = [t for t in token_details if not t["is_prefill"]]
        hit_rates = [t["expert_hit_rate"] for t in decode_details]
        avg_hit = np.mean(hit_rates) if hit_rates else 0

        # First-token and steady-state split
        if len(decode_details) >= 10:
            warmup_hit = np.mean([t["expert_hit_rate"] for t in decode_details[:10]])
            steady_hit = np.mean([t["expert_hit_rate"] for t in decode_details[10:]])
        else:
            warmup_hit = avg_hit
            steady_hit = avg_hit

        result["turn_idx"] = turn + 1
        result["turn_time_s"] = round(turn_time, 3)
        result["avg_expert_hit_rate"] = round(float(avg_hit), 1)
        result["warmup_hit_rate"] = round(float(warmup_hit), 1)
        result["steady_state_hit_rate"] = round(float(steady_hit), 1)

        turn_results.append(result)

        if verbose:
            print(f"    Time:          {turn_time:.2f}s")
            print(f"    Decode:        {result['decode_tok_s']:.1f} tok/s "
                  f"({result['avg_decode_s']*1000:.1f} ms/tok)")
            print(f"    Expert hit:    {result['avg_expert_hit_rate']:.1f}% avg "
                  f"({result['warmup_hit_rate']:.1f}% warmup, "
                  f"{result['steady_state_hit_rate']:.1f}% steady)")
            print(f"    Expert cache:  {result['expert_cache_size']}/{result['expert_cache_budget']} slots")
            print(f"    KV cache:      {result['kv_ram_mb']:.1f} MB "
                  f"(seq_len={result['kv_seq_len']}, "
                  f"evictions={result['kv_evictions']})")
            print(f"    KV budget:     {result['kv_budget_mb']:.1f} MB")
            print(f"    NVMe loads:    {result['expert_loads_from_nvme']}")
            print(f"    Rebalances:    {result['budget_rebalances']}")

    total_time = time.monotonic() - benchmark_start
    engine_stats = engine.stats()

    # Aggregate statistics
    all_decode_tok_s = [r["decode_tok_s"] for r in turn_results]
    all_hit_rates = [r["avg_expert_hit_rate"] for r in turn_results]

    aggregate = {
        "total_tokens": engine.tokens_generated,
        "total_time_s": round(total_time, 3),
        "overall_tok_s": round(engine.tokens_generated / total_time, 1) if total_time > 0 else 0,
        "avg_decode_tok_s": round(float(np.mean(all_decode_tok_s)), 1),
        "min_decode_tok_s": round(float(np.min(all_decode_tok_s)), 1),
        "max_decode_tok_s": round(float(np.max(all_decode_tok_s)), 1),
        "avg_expert_hit_rate": round(float(np.mean(all_hit_rates)), 1),
        "final_expert_hit_rate": round(engine.expert_cache.total_hit_rate, 1),
        "final_expert_cache_size": engine.expert_cache.size,
        "final_kv_ram_mb": round(engine.kv_cache.total_ram_bytes / 1024**2, 1),
        "total_kv_evictions": engine.kv_cache.total_evictions,
        "total_expert_nvme_loads": engine.expert_store.total_loads,
        "total_nvme_mb_read": round(engine.expert_store.total_bytes_loaded / 1024**2, 1),
        "total_rebalances": len(engine._rebalance_log),
        "final_kv_fraction": round(engine.coordinator.kv_fraction, 3),
        "final_expert_fraction": round(engine.coordinator.expert_fraction, 3),
    }

    benchmark_result = {
        "benchmark": "expertflow_integrated",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": config.name,
            "n_layers": config.n_layers,
            "n_moe_layers": config.n_moe_layers,
            "n_experts": config.n_experts,
            "n_active_experts": config.n_active_experts,
            "total_ram_gb": config.total_ram_gb,
            "available_budget_gb": config.available_budget_gb,
        },
        "parameters": {
            "n_turns": n_turns,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "eviction_policy": eviction_policy.value,
            "rebalance_every": rebalance_every,
        },
        "aggregate": aggregate,
        "turns": turn_results,
        "engine_stats": engine_stats,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  AGGREGATE RESULTS")
        print("=" * 70)
        print(f"  Total tokens:    {aggregate['total_tokens']}")
        print(f"  Total time:      {aggregate['total_time_s']:.1f}s")
        print(f"  Overall tok/s:   {aggregate['overall_tok_s']}")
        print(f"  Avg decode:      {aggregate['avg_decode_tok_s']} tok/s")
        print(f"  Expert hit rate: {aggregate['avg_expert_hit_rate']}% avg, "
              f"{aggregate['final_expert_hit_rate']}% final")
        print(f"  Expert cache:    {aggregate['final_expert_cache_size']} slots")
        print(f"  KV RAM:          {aggregate['final_kv_ram_mb']} MB "
              f"(evictions: {aggregate['total_kv_evictions']})")
        print(f"  NVMe loads:      {aggregate['total_expert_nvme_loads']} "
              f"({aggregate['total_nvme_mb_read']} MB)")
        print(f"  Budget split:    KV {aggregate['final_kv_fraction']*100:.1f}% / "
              f"Expert {aggregate['final_expert_fraction']*100:.1f}%")
        print(f"  Rebalances:      {aggregate['total_rebalances']}")

    return benchmark_result


def run_eviction_policy_comparison(config: ModelConfig, n_turns: int = 3,
                                   verbose: bool = True) -> dict:
    """Compare all eviction policies on the same workload."""
    if verbose:
        print("\n" + "=" * 70)
        print("  EVICTION POLICY COMPARISON")
        print("=" * 70)

    comparison = {}
    for policy in EvictionPolicy:
        if verbose:
            print(f"\n  >>> Policy: {policy.value}")
        result = run_benchmark(
            config=config,
            n_turns=n_turns,
            eviction_policy=policy,
            verbose=verbose,
        )
        comparison[policy.value] = {
            "avg_decode_tok_s": result["aggregate"]["avg_decode_tok_s"],
            "avg_expert_hit_rate": result["aggregate"]["avg_expert_hit_rate"],
            "total_kv_evictions": result["aggregate"]["total_kv_evictions"],
            "total_expert_nvme_loads": result["aggregate"]["total_expert_nvme_loads"],
        }

    if verbose:
        print("\n  " + "-" * 60)
        print(f"  {'Policy':<25} {'Hit Rate':>10} {'tok/s':>8} {'KV Evict':>10} {'NVMe Loads':>12}")
        print("  " + "-" * 60)
        for policy, stats in comparison.items():
            print(f"  {policy:<25} {stats['avg_expert_hit_rate']:>9.1f}% "
                  f"{stats['avg_decode_tok_s']:>7.1f} "
                  f"{stats['total_kv_evictions']:>10} "
                  f"{stats['total_expert_nvme_loads']:>12}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="ExpertFlow integrated benchmark — simulates MoE inference caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python benchmark_integrated.py --model small --turns 3
  python benchmark_integrated.py --model deepseek --turns 5 --output results.json
  python benchmark_integrated.py --model small --compare-policies
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--model", choices=["deepseek", "mixtral", "small"],
        default="small",
        help="Model config to simulate (default: small for quick test)"
    )
    parser.add_argument(
        "--turns", type=_positive_int("turns"), default=5,
        help="Number of conversation turns (default: 5)",
    )
    parser.add_argument(
        "--prompt-tokens", type=_positive_int("prompt-tokens"), default=50,
        help="Prompt tokens per turn (default: 50)",
    )
    parser.add_argument(
        "--response-tokens", type=_positive_int("response-tokens"), default=100,
        help="Response tokens to generate per turn (default: 100)",
    )
    parser.add_argument("--compare-policies", action="store_true",
                        help="Compare all eviction policies side-by-side")
    parser.add_argument(
        "--output", type=_output_path, default=None,
        help="Write JSON results to this file (default: auto-named with timestamp)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    configs = {
        "deepseek": DEEPSEEK_V3_CONFIG,
        "mixtral": MIXTRAL_8X7B_CONFIG,
        "small": DEEPSEEK_V3_SMALL_SIM,
    }
    config = configs[args.model]
    verbose = not args.quiet

    if args.compare_policies:
        comparison = run_eviction_policy_comparison(
            config=config, n_turns=args.turns, verbose=verbose
        )
        result = {
            "benchmark": "eviction_policy_comparison",
            "model": config.name,
            "comparison": comparison,
        }
    else:
        result = run_benchmark(
            config=config,
            n_turns=args.turns,
            prompt_tokens=args.prompt_tokens,
            response_tokens=args.response_tokens,
            verbose=verbose,
        )

    # Save results
    if args.output:
        outpath = args.output
    else:
        outpath = f"benchmark_integrated_{time.strftime('%H%M%S')}.json"

    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\n  Results saved to: {outpath}")


if __name__ == "__main__":
    main()
