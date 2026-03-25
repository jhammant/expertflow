"""Tests for MoE expert routing with boundary inputs.

Covers: SimulatedMoERouter edge cases, ExpertWeightCache behavior,
SimulatedExpertStore stats, and boundary conditions.
"""

import numpy as np
import pytest

from ef_integrated_engine import (
    SimulatedMoERouter,
    SimulatedExpertStore,
    ExpertWeightCache,
    ModelConfig,
)


class TestSimulatedMoERouterBoundary:
    """Boundary and edge case tests for expert routing."""

    def test_single_expert_model(self):
        """Router with n_experts=1, n_active=1 must always pick expert 0."""
        router = SimulatedMoERouter(
            n_experts=1, n_active=1, n_moe_layers=2,
        )
        experts, weights = router.route(0)
        assert experts == [0]
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0)

    def test_all_experts_active(self):
        """When n_active == n_experts, all should be selected."""
        router = SimulatedMoERouter(
            n_experts=4, n_active=4, n_moe_layers=1,
        )
        experts, weights = router.route(0)
        assert sorted(experts) == [0, 1, 2, 3]
        assert len(weights) == 4

    def test_no_duplicate_experts(self):
        """Selected experts must be unique within a token."""
        router = SimulatedMoERouter(
            n_experts=256, n_active=8, n_moe_layers=4,
        )
        for layer in range(4):
            experts, _ = router.route(layer)
            assert len(set(experts)) == 8

    def test_routing_weights_sum_to_one(self):
        router = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=2)
        _, weights = router.route(0)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-5)

    def test_route_all_layers(self):
        router = SimulatedMoERouter(
            n_experts=16, n_active=2, n_moe_layers=5,
        )
        all_routes = router.route_all_layers()
        assert len(all_routes) == 5
        for experts, weights in all_routes:
            assert len(experts) == 2
            assert len(weights) == 2

    def test_temporal_locality_boosts_reuse(self):
        """With high temporal locality, consecutive calls should share experts."""
        router = SimulatedMoERouter(
            n_experts=256, n_active=8, n_moe_layers=1,
            temporal_locality=0.9, seed=42,
        )
        first, _ = router.route(0)
        # Run several consecutive calls; reuse should be high
        reuse_count = 0
        for _ in range(10):
            second, _ = router.route(0)
            overlap = len(set(first) & set(second))
            reuse_count += overlap
            first = second
        avg_reuse = reuse_count / 10
        assert avg_reuse > 2  # with 0.9 locality, significant overlap expected

    def test_zero_temporal_locality(self):
        """With zero temporal locality, routing should still work."""
        router = SimulatedMoERouter(
            n_experts=16, n_active=4, n_moe_layers=2,
            temporal_locality=0.0,
        )
        experts, weights = router.route(0)
        assert len(experts) == 4
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-5)

    def test_layer_wrapping(self):
        """layer_idx > n_moe_layers should wrap around."""
        router = SimulatedMoERouter(
            n_experts=8, n_active=2, n_moe_layers=3,
        )
        # layer 5 should use probs[5 % 3] = probs[2]
        experts, _ = router.route(5)
        assert len(experts) == 2

    def test_deterministic_with_seed(self):
        """Same seed should produce identical routing."""
        r1 = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=2, seed=123)
        r2 = SimulatedMoERouter(n_experts=32, n_active=4, n_moe_layers=2, seed=123)
        for layer in range(2):
            e1, w1 = r1.route(layer)
            e2, w2 = r2.route(layer)
            assert e1 == e2


class TestExpertWeightCache:
    """Tests for expert weight cache eviction and scoring."""

    def test_put_get_roundtrip(self):
        cache = ExpertWeightCache(budget=10)
        data = (np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
        cache.put((0, 1), data)
        result = cache.get((0, 1))
        assert result is not None

    def test_miss_returns_none(self):
        cache = ExpertWeightCache(budget=5)
        assert cache.get((99, 99)) is None

    def test_trim_evicts_to_budget(self):
        cache = ExpertWeightCache(budget=3)
        for i in range(10):
            cache.put((0, i), (np.zeros(1),) * 3)
        evicted = cache.trim()
        assert evicted == 7
        assert cache.size == 3

    def test_hit_rate_tracking(self):
        cache = ExpertWeightCache(budget=10)
        cache.put((0, 0), (np.zeros(1),) * 3)

        cache.reset_token_stats()
        cache.get((0, 0))  # hit
        cache.get((0, 1))  # miss
        assert cache.token_hit_rate == pytest.approx(50.0)

    def test_total_hit_rate(self):
        cache = ExpertWeightCache(budget=10)
        cache.put((0, 0), (np.zeros(1),) * 3)
        # 3 hits, 2 misses
        cache.get((0, 0))
        cache.get((0, 0))
        cache.get((0, 0))
        cache.get((1, 1))
        cache.get((2, 2))
        assert cache.total_hit_rate == pytest.approx(60.0)

    def test_frequency_scoring_keeps_popular(self):
        """Frequently accessed experts should survive trim."""
        cache = ExpertWeightCache(budget=2, decay=0.95)
        cache.put((0, 0), (np.zeros(1),) * 3)
        cache.put((0, 1), (np.zeros(1),) * 3)
        cache.put((0, 2), (np.zeros(1),) * 3)

        # Access expert 0 many times to boost its score
        for _ in range(50):
            cache.get((0, 0))

        cache.trim()
        assert cache.size == 2
        assert cache.get((0, 0)) is not None  # popular one survives

    def test_empty_cache_hit_rate_zero(self):
        cache = ExpertWeightCache(budget=5)
        assert cache.total_hit_rate == 0
        assert cache.token_hit_rate == 0


class TestSimulatedExpertStore:
    """Tests for simulated NVMe expert loading."""

    def test_load_returns_three_arrays(self):
        store = SimulatedExpertStore(
            n_layers=4, first_moe_layer=1, n_experts=8,
            cold_load_ms=0.0, warm_load_ms=0.0,
        )
        gate, up, down = store.load_expert(1, 0)
        assert gate.dtype == np.float16
        assert up.shape == gate.shape
        assert down.shape == gate.shape

    def test_stats_tracking(self):
        store = SimulatedExpertStore(
            n_layers=4, first_moe_layer=1, n_experts=8,
            cold_load_ms=0.0, warm_load_ms=0.0,
        )
        store.load_expert(1, 0)
        store.load_expert(1, 0)  # warm load

        stats = store.stats()
        assert stats["total_loads"] == 2
        assert stats["cold_loads"] == 1
        assert stats["warm_loads"] == 1
        assert stats["total_bytes_loaded"] > 0

    def test_expert_bytes_calculation(self):
        store = SimulatedExpertStore(
            n_layers=2, first_moe_layer=0, n_experts=4,
            expert_rows=32, expert_cols=16,
        )
        # 3 projections × rows × cols × 2 bytes (float16)
        expected = 32 * 16 * 2 * 3
        assert store.expert_bytes == expected
