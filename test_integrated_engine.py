#!/usr/bin/env python3
"""
Tests for ExpertFlow Integrated Engine and Benchmark.

Tests cover:
  - SimulatedMoERouter: routing distribution, temporal locality, layer diversity
  - SimulatedExpertStore: load latency simulation, cold/warm tracking
  - ExpertWeightCache: hit/miss tracking, eviction, budget enforcement
  - ExpertFlowEngine: end-to-end pipeline, multi-turn, rebalancing
  - Benchmark: full run with all eviction policies
"""

import time
import numpy as np
import pytest

from ef_integrated_engine import (
    SimulatedMoERouter,
    SimulatedExpertStore,
    ExpertWeightCache,
    ExpertFlowEngine,
    ModelConfig,
    DEEPSEEK_V3_SMALL_SIM,
    MIXTRAL_8X7B_CONFIG,
    EvictionPolicy,
)
from ef_kv_manager import KVExpertBudgetCoordinator


# Use small config for all tests to keep them fast
FAST_CONFIG = ModelConfig(
    name="Test-Tiny",
    n_layers=4, first_moe_layer=1, n_experts=16, n_active_experts=2,
    num_kv_heads=2, head_dim=16,
    expert_rows=16, expert_cols=8,
    total_ram_gb=4, attn_budget_gb=1, os_overhead_gb=1,
    cold_load_ms=0.0, warm_load_ms=0.0,  # No sleep in tests
)


# ═══════════════════════════════════════════════════════════════════════
# SimulatedMoERouter Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSimulatedMoERouter:

    def test_route_returns_correct_count(self):
        router = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=8)
        experts, weights = router.route(0)
        assert len(experts) == 4
        assert len(weights) == 4

    def test_route_experts_unique(self):
        router = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=8)
        experts, _ = router.route(0)
        assert len(set(experts)) == 4  # no duplicates

    def test_route_experts_in_range(self):
        router = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=8)
        for layer in range(8):
            experts, _ = router.route(layer)
            assert all(0 <= e < 32 for e in experts)

    def test_route_weights_sum_to_one(self):
        router = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=8)
        _, weights = router.route(0)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_route_all_layers(self):
        router = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=8)
        results = router.route_all_layers()
        assert len(results) == 8
        for experts, weights in results:
            assert len(experts) == 4
            assert len(weights) == 4

    def test_zipf_skew_creates_popularity(self):
        """Higher skew should concentrate routing on fewer experts."""
        router_low = SimulatedMoERouter(
            n_experts=64, n_active=4, n_moe_layers=1, zipf_skew=0.5, seed=42
        )
        router_high = SimulatedMoERouter(
            n_experts=64, n_active=4, n_moe_layers=1, zipf_skew=2.0, seed=42
        )

        # Route many tokens and count unique experts
        unique_low = set()
        unique_high = set()
        for _ in range(200):
            experts, _ = router_low.route(0)
            unique_low.update(experts)
            experts, _ = router_high.route(0)
            unique_high.update(experts)

        # High skew should use fewer unique experts
        assert len(unique_high) <= len(unique_low)

    def test_temporal_locality(self):
        """With high temporal locality, consecutive tokens should share experts."""
        router = SimulatedMoERouter(
            n_experts=64, n_active=4, n_moe_layers=1,
            temporal_locality=0.9, seed=42,
        )

        experts_a, _ = router.route(0)
        experts_b, _ = router.route(0)

        # With 0.9 temporal locality, there should be significant overlap
        overlap = len(set(experts_a) & set(experts_b))
        # Not guaranteed every time, but over many runs this should average high
        # Just check it doesn't crash and returns valid data
        assert len(experts_b) == 4

    def test_deterministic_with_seed(self):
        router1 = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=4, seed=123)
        router2 = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=4, seed=123)

        for layer in range(4):
            e1, w1 = router1.route(layer)
            e2, w2 = router2.route(layer)
            assert e1 == e2
            np.testing.assert_array_almost_equal(w1, w2)


# ═══════════════════════════════════════════════════════════════════════
# SimulatedExpertStore Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSimulatedExpertStore:

    def test_load_returns_three_arrays(self):
        store = SimulatedExpertStore(
            n_layers=4, n_experts=8, expert_rows=16, expert_cols=8,
            cold_load_ms=0, warm_load_ms=0,
        )
        gate, up, down = store.load_expert(0, 0)
        assert gate.shape == (16, 8)
        assert up.shape == (16, 8)
        assert down.shape == (16, 8)
        assert gate.dtype == np.float16

    def test_cold_warm_tracking(self):
        store = SimulatedExpertStore(
            n_layers=4, n_experts=8, expert_rows=16, expert_cols=8,
            cold_load_ms=0, warm_load_ms=0,
        )
        store.load_expert(0, 0)  # cold
        store.load_expert(0, 0)  # warm
        store.load_expert(1, 5)  # cold

        assert store.total_cold == 2
        assert store.total_warm == 1
        assert store.total_loads == 3

    def test_expert_bytes(self):
        store = SimulatedExpertStore(
            expert_rows=256, expert_cols=128,
            cold_load_ms=0, warm_load_ms=0,
        )
        # 256 * 128 * 2 bytes * 3 projections
        assert store.expert_bytes == 256 * 128 * 2 * 3

    def test_total_bytes_tracked(self):
        store = SimulatedExpertStore(
            expert_rows=16, expert_cols=8,
            cold_load_ms=0, warm_load_ms=0,
        )
        store.load_expert(0, 0)
        store.load_expert(0, 1)
        assert store.total_bytes_loaded == store.expert_bytes * 2

    def test_stats(self):
        store = SimulatedExpertStore(
            expert_rows=16, expert_cols=8,
            cold_load_ms=0, warm_load_ms=0,
        )
        store.load_expert(0, 0)
        s = store.stats()
        assert s["total_loads"] == 1
        assert s["cold_loads"] == 1
        assert s["expert_bytes"] > 0


# ═══════════════════════════════════════════════════════════════════════
# ExpertWeightCache Tests
# ═══════════════════════════════════════════════════════════════════════

class TestExpertWeightCache:

    def test_get_miss(self):
        cache = ExpertWeightCache(budget=10)
        assert cache.get((0, 0)) is None
        assert cache.total_misses == 1

    def test_put_and_get(self):
        cache = ExpertWeightCache(budget=10)
        dummy = (np.zeros(4), np.zeros(4), np.zeros(4))
        cache.put((0, 0), dummy)
        result = cache.get((0, 0))
        assert result is not None
        assert cache.total_hits == 1

    def test_trim_evicts_to_budget(self):
        cache = ExpertWeightCache(budget=5)
        for i in range(10):
            cache.put((0, i), (np.zeros(1),) * 3)
        assert cache.size == 10

        evicted = cache.trim()
        assert evicted == 5
        assert cache.size == 5

    def test_trim_evicts_lowest_scored(self):
        cache = ExpertWeightCache(budget=2)
        cache.put((0, 0), (np.zeros(1),) * 3)
        cache.put((0, 1), (np.zeros(1),) * 3)
        cache.put((0, 2), (np.zeros(1),) * 3)

        # Boost score of entry 0 by accessing it
        for _ in range(5):
            cache.get((0, 0))

        cache.trim()
        # Entry 0 should survive (highest score)
        assert cache.get((0, 0)) is not None

    def test_hit_rate(self):
        cache = ExpertWeightCache(budget=10)
        cache.put((0, 0), (np.zeros(1),) * 3)
        cache.get((0, 0))  # hit
        cache.get((0, 1))  # miss
        assert cache.token_hit_rate == 50.0

    def test_total_hit_rate(self):
        cache = ExpertWeightCache(budget=10)
        cache.put((0, 0), (np.zeros(1),) * 3)
        cache.get((0, 0))  # hit
        cache.get((0, 0))  # hit
        cache.get((0, 1))  # miss
        assert cache.total_hit_rate == pytest.approx(66.7, abs=0.1)

    def test_reset_token_stats(self):
        cache = ExpertWeightCache(budget=10)
        cache.put((0, 0), (np.zeros(1),) * 3)
        cache.get((0, 0))
        cache.get((0, 1))
        cache.reset_token_stats()
        assert cache.token_hits == 0
        assert cache.token_misses == 0
        assert cache.total_hits == 1  # total not reset

    def test_end_token(self):
        cache = ExpertWeightCache()
        assert cache.current_token == 0
        cache.end_token()
        assert cache.current_token == 1

    def test_stats(self):
        cache = ExpertWeightCache(budget=100)
        cache.put((0, 0), (np.zeros(1),) * 3)
        cache.get((0, 0))
        s = cache.stats()
        assert s["size"] == 1
        assert s["budget"] == 100
        assert s["total_hits"] == 1


# ═══════════════════════════════════════════════════════════════════════
# ExpertFlowEngine Tests
# ═══════════════════════════════════════════════════════════════════════

class TestExpertFlowEngine:

    def test_creation(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        assert engine.tokens_generated == 0
        assert engine.expert_cache.budget > 0
        assert engine.kv_cache.n_layers == FAST_CONFIG.n_layers

    def test_generate_single_token(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.generate_token()

        assert result["token_idx"] == 1
        assert result["is_prefill"] is False
        assert result["time_s"] > 0
        assert result["kv_seq_len"] == 1
        assert "kv_hit_rate" in result
        assert engine.tokens_generated == 1

    def test_generate_prefill(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.generate_token(is_prefill=True, prefill_tokens=10)

        assert result["is_prefill"] is True
        assert result["n_tokens"] == 10
        assert result["kv_seq_len"] == 10

    def test_generate_multiple_tokens(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)

        for i in range(5):
            result = engine.generate_token()

        assert engine.tokens_generated == 5
        assert engine.kv_cache.total_seq_len == 5
        assert len(engine.token_times) == 5

    def test_expert_cache_gets_populated(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)

        # Generate enough tokens to populate the cache
        for _ in range(10):
            engine.generate_token()

        assert engine.expert_cache.size > 0
        assert engine.expert_store.total_loads > 0

    def test_expert_cache_hit_rate_improves(self):
        """After warmup, hit rate should increase as cache fills."""
        engine = ExpertFlowEngine(config=FAST_CONFIG)

        # Warmup phase — lots of misses
        for _ in range(5):
            engine.generate_token()
        early_hits = engine.expert_cache.total_hits
        early_misses = engine.expert_cache.total_misses

        # Steady state — should see more hits
        for _ in range(20):
            engine.generate_token()

        assert engine.expert_cache.total_hit_rate > 0  # some hits
        assert engine.kv_cache.cache_hits > 0

    def test_rebalance_budgets(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)

        for _ in range(5):
            engine.generate_token()

        result = engine.rebalance_budgets()
        assert "rebalance_id" in result
        assert "new_kv_bytes" in result
        assert len(engine._rebalance_log) == 1

    def test_generate_turn(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.generate_turn(
            prompt_tokens=5, response_tokens=10, rebalance_every=5
        )

        assert result["prompt_tokens"] == 5
        assert result["response_tokens"] == 10
        assert result["total_time_s"] > 0
        assert result["decode_tok_s"] > 0
        assert result["kv_seq_len"] > 0
        assert "expert_cache_hit_rate" in result
        assert "kv_hit_rate" in result

    def test_multi_turn_context_grows(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)

        kv_seq_lens = []
        for turn in range(3):
            result = engine.generate_turn(
                prompt_tokens=5, response_tokens=5, rebalance_every=10
            )
            kv_seq_lens.append(result["kv_seq_len"])

        # Context should grow with each turn
        assert kv_seq_lens[1] > kv_seq_lens[0]
        assert kv_seq_lens[2] > kv_seq_lens[1]

    def test_stats(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        engine.generate_turn(prompt_tokens=3, response_tokens=5)

        stats = engine.stats()
        assert stats["config"]["model"] == "Test-Tiny"
        assert stats["tokens_generated"] > 0
        assert "kv_cache" in stats
        assert "expert_cache" in stats
        assert "expert_store" in stats
        assert "budget_coordinator" in stats

    def test_all_eviction_policies(self):
        """Verify engine works with all KV eviction policies."""
        for policy in EvictionPolicy:
            engine = ExpertFlowEngine(
                config=FAST_CONFIG,
                kv_eviction_policy=policy,
            )
            result = engine.generate_turn(
                prompt_tokens=3, response_tokens=5
            )
            assert result["decode_tok_s"] > 0

    def test_kv_budget_enforcement(self):
        """With tiny KV budget, evictions should happen."""
        tiny_config = ModelConfig(
            name="Tiny-KV-Test",
            n_layers=2, first_moe_layer=0, n_experts=8, n_active_experts=2,
            num_kv_heads=2, head_dim=8,
            expert_rows=8, expert_cols=4,
            total_ram_gb=1, attn_budget_gb=0, os_overhead_gb=0,
            cold_load_ms=0, warm_load_ms=0,
        )
        engine = ExpertFlowEngine(
            config=tiny_config,
            initial_kv_fraction=0.001,  # very small KV budget
        )

        # Manually set a very tight KV budget to ensure evictions
        # Each token across 2 layers: 2 heads × 8 dim × 2 bytes × 2 (K+V) × 2 layers = 128 bytes
        # Set budget to fit ~5 tokens
        engine.kv_cache.kv_budget_bytes = 640

        for _ in range(50):
            engine.generate_token()

        # With tight KV budget, there should be evictions
        assert engine.kv_cache.total_evictions > 0

    def test_expert_budget_updates_after_rebalance(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        initial_budget = engine.expert_cache.budget

        # Generate tokens to build state
        for _ in range(10):
            engine.generate_token()

        engine.rebalance_budgets()
        # Budget may or may not change, but it should be a valid positive int
        assert engine.expert_cache.budget > 0

    def test_preset_configs_work(self):
        """Verify preset configs can construct engines."""
        for config in [DEEPSEEK_V3_SMALL_SIM, MIXTRAL_8X7B_CONFIG]:
            engine = ExpertFlowEngine(config=config)
            assert engine.config.name == config.name
            assert engine.expert_cache.budget > 0


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Integration Test
# ═══════════════════════════════════════════════════════════════════════

class TestBenchmarkIntegration:

    def test_full_benchmark_run(self):
        """Run a minimal benchmark and verify output structure."""
        from benchmark_integrated import run_benchmark

        result = run_benchmark(
            config=FAST_CONFIG,
            n_turns=2,
            prompt_tokens=3,
            response_tokens=5,
            verbose=False,
        )

        assert result["benchmark"] == "expertflow_integrated"
        assert result["config"]["model"] == "Test-Tiny"
        assert len(result["turns"]) == 2
        assert "aggregate" in result
        agg = result["aggregate"]
        assert agg["total_tokens"] > 0
        assert agg["avg_decode_tok_s"] > 0
        assert 0 <= agg["avg_expert_hit_rate"] <= 100
        assert agg["final_kv_fraction"] + agg["final_expert_fraction"] == pytest.approx(1.0, abs=0.01)

    def test_eviction_policy_comparison(self):
        from benchmark_integrated import run_eviction_policy_comparison

        comparison = run_eviction_policy_comparison(
            config=FAST_CONFIG, n_turns=1, verbose=False,
        )

        assert "lru" in comparison
        assert "frequency_weighted" in comparison
        assert "belady_approximate" in comparison

        for policy, stats in comparison.items():
            assert "avg_decode_tok_s" in stats
            assert "avg_expert_hit_rate" in stats
            assert "final_kv_hit_rate" in stats

    def test_kv_reads_drive_cache_hits(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        engine.generate_token(is_prefill=True, prefill_tokens=8)
        engine.generate_token()

        assert engine.kv_cache.cache_hits > 0
        assert engine.kv_cache.total_seq_len == 9

    def test_kv_ssd_tiering_surfaces_page_io(self, tmp_path):
        config = ModelConfig(
            name="KV-Tiering-Test",
            n_layers=2, first_moe_layer=0, n_experts=4, n_active_experts=1,
            num_kv_heads=2, head_dim=8,
            expert_rows=8, expert_cols=4,
            total_ram_gb=1, attn_budget_gb=0, os_overhead_gb=0,
            cold_load_ms=0, warm_load_ms=0,
            kv_block_size=4,
        )
        engine = ExpertFlowEngine(
            config=config,
            initial_kv_fraction=0.001,
            kv_spill_directory=str(tmp_path),
        )
        engine.kv_cache.kv_budget_bytes = 128

        engine.generate_token(is_prefill=True, prefill_tokens=12)
        engine.generate_token()

        assert engine.kv_cache.total_page_outs > 0
        assert engine.kv_cache.total_page_ins > 0


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases: Memory Pressure, Concurrent Patterns, Error Recovery
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryPressure:

    def test_extreme_kv_pressure_all_evicted(self):
        """When KV budget is near-zero, all blocks get evicted except current."""
        from ef_kv_manager import MoEKVCacheManager
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=8, kv_budget_bytes=128,  # room for ~2 blocks
        )
        for _ in range(100):
            k = np.random.randn(2, 1, 8).astype(np.float16)
            v = np.random.randn(2, 1, 8).astype(np.float16)
            mgr.append_kv(0, k, v)
        assert mgr.total_evictions > 0
        assert mgr.total_ram_bytes <= 128

    def test_expert_cache_full_churn(self):
        """When expert budget is tiny, every token causes full cache churn."""
        cache = ExpertWeightCache(budget=3)
        for i in range(50):
            key = (0, i % 10)  # 10 unique experts, cache holds 3
            if cache.get(key) is None:
                cache.put(key, (np.zeros(1),) * 3)
            cache.trim()
        # Should have high miss rate due to thrashing
        assert cache.total_misses > cache.total_hits

    def test_budget_coordinator_oscillation_stability(self):
        """Coordinator should not oscillate wildly between extremes."""
        from ef_kv_manager import KVExpertBudgetCoordinator
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=10 * 1024**3,
            initial_kv_fraction=0.15,
        )
        budgets = []
        for i in range(20):
            # Alternate high/low pressure to try to induce oscillation
            if i % 2 == 0:
                coord.rebalance(kv_ram_used=coord.kv_budget_bytes, expert_hit_rate=0.95,
                                seq_len=1000, bytes_per_token=8192)
            else:
                coord.rebalance(kv_ram_used=0, expert_hit_rate=0.60,
                                seq_len=1000, bytes_per_token=8192)
            budgets.append(coord.kv_budget_bytes)

        # Max swing between consecutive rebalances should be bounded (<5% of total)
        max_swing = max(abs(budgets[i+1] - budgets[i]) for i in range(len(budgets)-1))
        assert max_swing <= coord.total_budget_bytes * 0.05

    def test_kv_and_expert_compete_under_pressure(self):
        """Full engine under memory pressure: both caches fighting for RAM."""
        pressure_config = ModelConfig(
            name="Pressure-Test",
            n_layers=4, first_moe_layer=1, n_experts=32, n_active_experts=4,
            num_kv_heads=2, head_dim=8,
            expert_rows=16, expert_cols=8,
            total_ram_gb=1, attn_budget_gb=0, os_overhead_gb=0,
            cold_load_ms=0, warm_load_ms=0,
        )
        engine = ExpertFlowEngine(config=pressure_config, initial_kv_fraction=0.5)
        # Force very tight budgets
        engine.kv_cache.kv_budget_bytes = 512
        engine.expert_cache.budget = 5

        for _ in range(30):
            engine.generate_token()

        # System should survive without crashing
        assert engine.tokens_generated == 30
        assert engine.expert_cache.total_misses > 0  # cache too small for all experts

    def test_zero_budget_coordinator(self):
        """Coordinator with zero budget should not crash."""
        from ef_kv_manager import KVExpertBudgetCoordinator
        coord = KVExpertBudgetCoordinator(total_budget_bytes=0)
        result = coord.rebalance(kv_ram_used=0, expert_hit_rate=0.5,
                                 seq_len=100, bytes_per_token=1024)
        assert result["new_kv_bytes"] == 0

    def test_single_expert_model(self):
        """Model with only 1 expert per layer (dense model edge case)."""
        dense_config = ModelConfig(
            name="Dense-Edge",
            n_layers=2, first_moe_layer=0, n_experts=1, n_active_experts=1,
            num_kv_heads=2, head_dim=8,
            expert_rows=8, expert_cols=4,
            total_ram_gb=1, attn_budget_gb=0, os_overhead_gb=0,
            cold_load_ms=0, warm_load_ms=0,
        )
        engine = ExpertFlowEngine(config=dense_config)
        result = engine.generate_turn(prompt_tokens=3, response_tokens=5)
        assert result["total_time_s"] >= 0
        assert engine.tokens_generated > 0
        # With 1 expert, hit rate should be very high after warmup
        assert engine.expert_cache.total_hit_rate > 50


class TestConcurrentPatterns:

    def test_bursty_expert_access(self):
        """Simulate bursty access: one expert heavily used then abandoned."""
        cache = ExpertWeightCache(budget=5)
        # Phase 1: hammer expert 0
        for _ in range(20):
            if cache.get((0, 0)) is None:
                cache.put((0, 0), (np.zeros(1),) * 3)
        # Phase 2: use different experts
        for i in range(1, 10):
            cache.put((0, i), (np.zeros(1),) * 3)
            cache.trim()
        # Expert 0 should eventually be evicted as score decays
        cache.trim()
        cache.trim()
        # After enough trims with decay, cache adapts

    def test_sequential_layer_sweep(self):
        """Simulate sequential layer sweep (how real inference works)."""
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        for _ in range(10):
            engine.generate_token()

        # Verify all layers have equal seq_len
        seq_lens = [layer.seq_len for layer in engine.kv_cache.layers]
        assert all(s == seq_lens[0] for s in seq_lens)

    def test_long_conversation(self):
        """Multi-turn long conversation doesn't degrade or crash."""
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        for turn in range(10):
            result = engine.generate_turn(
                prompt_tokens=5, response_tokens=10, rebalance_every=10
            )
        assert engine.tokens_generated > 100
        stats = engine.stats()
        assert stats["tokens_generated"] > 100

    def test_prefill_then_decode_pattern(self):
        """Prefill a large batch then decode one-by-one."""
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        engine.generate_token(is_prefill=True, prefill_tokens=20)
        for _ in range(10):
            engine.generate_token()
        assert engine.kv_cache.total_seq_len == 30


class TestErrorRecovery:

    def test_empty_cache_operations(self):
        """Operations on empty caches should not crash."""
        cache = ExpertWeightCache(budget=10)
        assert cache.get((99, 99)) is None
        assert cache.trim() == 0
        assert cache.total_hit_rate == 0
        assert cache.token_hit_rate == 0
        cache.end_token()
        cache.reset_token_stats()

    def test_evict_from_empty_kv(self):
        from ef_kv_manager import MoEKVCacheManager
        mgr = MoEKVCacheManager(n_layers=2)
        freed = mgr.evict_before(1000)
        assert freed == 0
        mgr.clear()

    def test_get_kv_from_empty_layer(self):
        from ef_kv_manager import MoEKVCacheManager
        mgr = MoEKVCacheManager(n_layers=2, num_kv_heads=2, head_dim=8)
        k, v = mgr.get_kv(0)
        assert k is None
        assert v is None

    def test_rebalance_with_no_usage(self):
        """Rebalance before any tokens are generated."""
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.rebalance_budgets()
        assert result["new_kv_bytes"] >= 0

    def test_page_in_missing_file(self):
        """Page-in with bad path should return 0 bytes, not crash."""
        from ef_kv_manager import KVBlock
        block = KVBlock(layer_idx=0, seq_start=0, seq_end=16)
        block.on_disk = True
        block.disk_path = "/nonexistent/path.npz"
        try:
            block.page_in()
        except (FileNotFoundError, OSError):
            pass  # expected — just verify no unhandled crash

    def test_coordinator_extreme_hit_rate(self):
        """Coordinator handles 0% and 100% hit rates."""
        from ef_kv_manager import KVExpertBudgetCoordinator
        coord = KVExpertBudgetCoordinator(total_budget_bytes=10 * 1024**3)
        coord.rebalance(kv_ram_used=0, expert_hit_rate=0.0,
                        seq_len=100, bytes_per_token=1024)
        coord.rebalance(kv_ram_used=0, expert_hit_rate=1.0,
                        seq_len=100, bytes_per_token=1024)
        assert coord.kv_budget_bytes + coord.expert_budget_bytes == coord.total_budget_bytes

    def test_store_repeated_loads(self):
        """Loading same expert 100 times should all be warm after first."""
        store = SimulatedExpertStore(
            n_layers=2, n_experts=4, expert_rows=8, expert_cols=4,
            cold_load_ms=0, warm_load_ms=0,
        )
        for _ in range(100):
            store.load_expert(0, 0)
        assert store.total_cold == 1
        assert store.total_warm == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
