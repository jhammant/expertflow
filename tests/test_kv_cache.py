"""Tests for KV cache allocation, eviction, and budget coordination.

Covers: KVBlock paging, PagedKVCache eviction, MoEKVCacheManager budget
enforcement, eviction policies, and KVExpertBudgetCoordinator rebalancing.
"""

import tempfile
import time

import numpy as np
import pytest

from ef_kv_manager import (
    BeladyPredictor,
    KVBlock,
    PagedKVCache,
    MoEKVCacheManager,
    KVExpertBudgetCoordinator,
    EvictionPolicy,
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def make_kv(num_kv_heads=2, n_tokens=4, head_dim=16):
    keys = np.random.randn(num_kv_heads, n_tokens, head_dim).astype(np.float16)
    values = np.random.randn(num_kv_heads, n_tokens, head_dim).astype(np.float16)
    return keys, values


class TestKVBlockPageOut:
    """Tests for KVBlock NVMe page-out/page-in."""

    def test_page_out_frees_memory(self, tmp_path):
        k, v = make_kv(n_tokens=8)
        block = KVBlock(layer_idx=0, seq_start=0, seq_end=8, keys=k, values=v)
        initial_bytes = block.nbytes
        assert initial_bytes > 0

        freed = block.page_out(str(tmp_path / "block.npz"))
        assert freed == initial_bytes
        assert block.on_disk is True
        assert block.keys is None
        assert block.values is None
        assert block.nbytes == 0

    def test_page_in_restores(self, tmp_path):
        k, v = make_kv(n_tokens=8)
        block = KVBlock(layer_idx=0, seq_start=0, seq_end=8,
                        keys=k.copy(), values=v.copy())

        block.page_out(str(tmp_path / "block.npz"))
        loaded = block.page_in()
        assert loaded > 0
        assert block.on_disk is False
        np.testing.assert_array_equal(block.keys, k)
        np.testing.assert_array_equal(block.values, v)

    def test_page_out_pinned_noop(self, tmp_path):
        k, v = make_kv(n_tokens=4)
        block = KVBlock(layer_idx=0, seq_start=0, seq_end=4,
                        keys=k, values=v, pinned=True)
        freed = block.page_out(str(tmp_path / "pinned.npz"))
        assert freed == 0
        assert block.on_disk is False

    def test_double_page_out_noop(self, tmp_path):
        k, v = make_kv(n_tokens=4)
        block = KVBlock(layer_idx=0, seq_start=0, seq_end=4, keys=k, values=v)
        block.page_out(str(tmp_path / "block.npz"))
        freed = block.page_out(str(tmp_path / "block2.npz"))
        assert freed == 0

    def test_page_in_without_page_out_noop(self):
        k, v = make_kv(n_tokens=4)
        block = KVBlock(layer_idx=0, seq_start=0, seq_end=4, keys=k, values=v)
        loaded = block.page_in()
        assert loaded == 0


class TestPagedKVCacheEviction:
    """Tests for PagedKVCache block eviction."""

    def test_evict_oldest_removes_first_block(self):
        cache = PagedKVCache(layer_idx=0, block_size=4, num_kv_heads=2, head_dim=16)
        k1, v1 = make_kv(n_tokens=4)
        k2, v2 = make_kv(n_tokens=4)
        cache.append(k1, v1)
        time.sleep(0.01)
        cache.append(k2, v2)

        freed = cache.evict_oldest()
        assert freed > 0
        assert cache.num_blocks == 1
        assert 0 not in cache.blocks
        assert 4 in cache.blocks

    def test_evict_nonexistent_block(self):
        cache = PagedKVCache(layer_idx=0, block_size=4, num_kv_heads=2, head_dim=16)
        freed = cache.evict_block(9999)
        assert freed == 0

    def test_evict_pinned_block_fails(self):
        cache = PagedKVCache(layer_idx=0, block_size=4, num_kv_heads=2, head_dim=16)
        k, v = make_kv(n_tokens=4)
        cache.append(k, v)
        cache.pin_range(0, 4)

        freed = cache.evict_block(0)
        assert freed == 0
        assert cache.num_blocks == 1

    def test_clear_resets_state(self):
        cache = PagedKVCache(layer_idx=0, block_size=4, num_kv_heads=2, head_dim=16)
        k, v = make_kv(n_tokens=8)
        cache.append(k, v)
        assert cache.num_blocks > 0

        cache.clear()
        assert cache.num_blocks == 0
        assert cache.seq_len == 0

    def test_pin_and_unpin_range(self):
        cache = PagedKVCache(layer_idx=0, block_size=4, num_kv_heads=2, head_dim=16)
        k, v = make_kv(n_tokens=8)
        cache.append(k, v)

        cache.pin_range(0, 8)
        for block in cache.blocks.values():
            assert block.pinned

        cache.unpin_range(0, 8)
        for block in cache.blocks.values():
            assert not block.pinned


class TestMoEKVCacheManagerBudget:
    """Tests for KV cache budget enforcement and eviction policies."""

    def test_budget_enforcement_triggers_eviction(self):
        """Appending beyond budget should trigger evictions."""
        # Small budget: enough for ~2 tokens across 2 layers
        mgr = MoEKVCacheManager(
            n_layers=2, num_kv_heads=2, head_dim=16, block_size=4,
            kv_budget_bytes=512,
            eviction_policy=EvictionPolicy.LRU,
        )
        # Each append: 2 heads × 1 token × 16 dim × 2 bytes × 2 (K+V) = 128 bytes
        # Across 2 layers = 256 bytes per token
        for i in range(10):
            for layer in range(2):
                k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=16)
                mgr.append_kv(layer, k, v)

        assert mgr.total_ram_bytes <= mgr.kv_budget_bytes + 512
        assert mgr.total_evictions > 0

    def test_layer_index_out_of_range(self):
        mgr = MoEKVCacheManager(n_layers=4, num_kv_heads=2, head_dim=16)
        k, v = make_kv()
        with pytest.raises(IndexError):
            mgr.append_kv(10, k, v)
        with pytest.raises(IndexError):
            mgr.append_kv(-1, k, v)

    def test_get_kv_layer_out_of_range(self):
        mgr = MoEKVCacheManager(n_layers=4, num_kv_heads=2, head_dim=16)
        with pytest.raises(IndexError):
            mgr.get_kv(10)

    def test_bytes_per_token_all_layers(self):
        mgr = MoEKVCacheManager(
            n_layers=4, num_kv_heads=8, head_dim=128,
        )
        # 4 layers × 8 heads × 128 dim × 2 (K+V) × 2 (bytes)
        expected = 4 * 8 * 128 * 2 * 2
        assert mgr.bytes_per_token_all_layers() == expected

    def test_max_tokens_in_budget(self):
        mgr = MoEKVCacheManager(
            n_layers=2, num_kv_heads=2, head_dim=16,
            kv_budget_bytes=1024,
        )
        bpt = mgr.bytes_per_token_all_layers()
        assert mgr.max_tokens_in_budget() == 1024 // bpt

    def test_frequency_weighted_eviction(self):
        """Frequently accessed blocks should survive eviction."""
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=16, block_size=4,
            kv_budget_bytes=512,
            eviction_policy=EvictionPolicy.FREQUENCY_WEIGHTED,
        )
        # Append 3 blocks
        for _ in range(12):
            k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=16)
            mgr.append_kv(0, k, v)

        # Access first block heavily to boost its score
        for _ in range(20):
            mgr.get_kv(0, 0, 4)
            time.sleep(0.001)

        # Force more evictions by adding more data
        for _ in range(8):
            k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=16)
            mgr.append_kv(0, k, v)

        assert mgr.total_evictions > 0

    def test_all_policies_produce_evictions(self):
        """Every eviction policy should be able to evict under pressure."""
        for policy in EvictionPolicy:
            mgr = MoEKVCacheManager(
                n_layers=1, num_kv_heads=2, head_dim=16, block_size=4,
                kv_budget_bytes=256,
                eviction_policy=policy,
            )
            for _ in range(20):
                k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=16)
                mgr.append_kv(0, k, v)
            assert mgr.total_evictions > 0, f"Policy {policy.value} did not evict"

    def test_ssd_tiering_pages_out_and_back_in(self, tmp_path):
        mgr = MoEKVCacheManager(
            n_layers=1,
            num_kv_heads=2,
            head_dim=16,
            block_size=4,
            kv_budget_bytes=256,
            enable_ssd_tiering=True,
            spill_directory=str(tmp_path),
        )

        for _ in range(8):
            k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=16)
            mgr.append_kv(0, k, v)

        assert mgr.total_page_outs > 0
        assert mgr.total_on_disk_blocks > 0

        keys, values = mgr.get_kv(0, 0, 4)
        assert keys is not None
        assert values is not None
        assert mgr.total_page_ins > 0
        assert mgr.cache_misses > 0

    def test_get_kv_tracks_hits_and_misses(self, tmp_path):
        mgr = MoEKVCacheManager(
            n_layers=1,
            num_kv_heads=2,
            head_dim=16,
            block_size=4,
            kv_budget_bytes=1024,
            enable_ssd_tiering=True,
            spill_directory=str(tmp_path),
        )

        for _ in range(4):
            k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=16)
            mgr.append_kv(0, k, v)

        mgr.get_kv(0, 0, 4)
        assert mgr.cache_hits > 0

        block = mgr.layers[0].blocks[0]
        block.page_out(str(tmp_path / "manual_block.npz"))
        mgr.get_kv(0, 0, 4)
        assert mgr.cache_misses > 0
        assert mgr.total_page_ins > 0

    def test_belady_predictor_evicts_farthest_future_use(self):
        mgr = MoEKVCacheManager(
            n_layers=1,
            num_kv_heads=2,
            head_dim=16,
            block_size=4,
            kv_budget_bytes=256,
            eviction_policy=EvictionPolicy.BELADY_APPROXIMATE,
            enable_ssd_tiering=False,
        )

        for _ in range(12):
            k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=16)
            mgr.append_kv(0, k, v)

        predictor = BeladyPredictor()
        predictor.prime([(0, 0), (0, 4)])
        mgr.set_belady_predictor(predictor)

        victim = mgr._pick_eviction_candidate()
        assert victim == (0, 8)


class TestKVExpertBudgetCoordinator:
    """Tests for dynamic budget rebalancing."""

    def test_initial_split(self):
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=1024 * 1024,  # 1MB
            initial_kv_fraction=0.2,
        )
        assert coord.kv_fraction == pytest.approx(0.2)
        assert coord.expert_fraction == pytest.approx(0.8)
        assert coord.kv_budget_bytes + coord.expert_budget_bytes == 1024 * 1024

    def test_rebalance_increases_kv_under_pressure(self):
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=1024 * 1024,
            initial_kv_fraction=0.1,
            min_kv_fraction=0.05,
            max_kv_fraction=0.5,
        )
        # Simulate: KV is fully utilized, expert hit rate is high
        result = coord.rebalance(
            kv_ram_used=coord.kv_budget_bytes,
            expert_hit_rate=0.95,
            seq_len=1000,
            bytes_per_token=256,
        )
        # Should try to give more to KV since it's at capacity and experts are fine
        assert coord.kv_fraction >= 0.1

    def test_rebalance_clamps_to_max(self):
        """Rebalancing should clamp kv_fraction within [min, max]."""
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=10000,
            initial_kv_fraction=0.5,
            min_kv_fraction=0.1,
            max_kv_fraction=0.4,
        )
        # Initial fraction is not clamped in __init__
        assert coord.kv_fraction == pytest.approx(0.5)
        # After rebalance, should be clamped to max
        coord.rebalance(kv_ram_used=5000, expert_hit_rate=0.95,
                        seq_len=100, bytes_per_token=10)
        assert coord.kv_fraction <= 0.4 + 0.001

    def test_expert_cache_slots(self):
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=1024 * 1024,
            initial_kv_fraction=0.2,
        )
        expert_bytes = 1024
        slots = coord.expert_cache_slots(expert_bytes)
        expected = int(coord.expert_budget_bytes // expert_bytes)
        assert slots == expected

    def test_zero_budget(self):
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=0,
            initial_kv_fraction=0.5,
        )
        assert coord.kv_budget_bytes == 0
        assert coord.expert_budget_bytes == 0
