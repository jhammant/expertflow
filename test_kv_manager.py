#!/usr/bin/env python3
"""
Comprehensive tests for ExpertFlow KV Cache Manager.

Tests cover:
  - KVBlock: creation, access tracking, page out/in, pinning
  - PagedKVCache: append, get_kv, eviction, multi-block, pinning
  - MoEKVCacheManager: multi-layer, budget enforcement, eviction policies
  - KVExpertBudgetCoordinator: rebalancing, bounds, expert slot calculation
  - Integration: KV + expert cache memory coordination
"""

import os
import time
import tempfile
import numpy as np
import pytest

from ef_kv_manager import (
    KVBlock,
    PagedKVCache,
    MoEKVCacheManager,
    KVExpertBudgetCoordinator,
    EvictionPolicy,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def make_kv(num_kv_heads=8, n_tokens=16, head_dim=128, dtype=np.float16):
    """Create random K and V tensors for testing."""
    keys = np.random.randn(num_kv_heads, n_tokens, head_dim).astype(dtype)
    values = np.random.randn(num_kv_heads, n_tokens, head_dim).astype(dtype)
    return keys, values


def make_kv_block(layer_idx=0, seq_start=0, seq_end=256,
                  num_kv_heads=8, head_dim=128):
    """Create a KVBlock with random data."""
    n_tokens = seq_end - seq_start
    keys = np.random.randn(num_kv_heads, n_tokens, head_dim).astype(np.float16)
    values = np.random.randn(num_kv_heads, n_tokens, head_dim).astype(np.float16)
    return KVBlock(
        layer_idx=layer_idx,
        seq_start=seq_start,
        seq_end=seq_end,
        keys=keys,
        values=values,
    )


# ═══════════════════════════════════════════════════════════════════════
# KVBlock Tests
# ═══════════════════════════════════════════════════════════════════════

class TestKVBlock:

    def test_creation(self):
        block = make_kv_block(layer_idx=3, seq_start=0, seq_end=256)
        assert block.layer_idx == 3
        assert block.seq_start == 0
        assert block.seq_end == 256
        assert block.block_size == 256
        assert block.keys.shape == (8, 256, 128)
        assert block.values.shape == (8, 256, 128)

    def test_block_id(self):
        block = make_kv_block(layer_idx=5, seq_start=512, seq_end=768)
        assert block.block_id == (5, 512)

    def test_nbytes_in_memory(self):
        block = make_kv_block(num_kv_heads=4, head_dim=64)
        # 4 heads × 256 tokens × 64 dim × 2 bytes × 2 (K+V)
        expected = 4 * 256 * 64 * 2 * 2
        assert block.nbytes == expected

    def test_nbytes_on_disk(self):
        block = make_kv_block()
        block.keys = None
        block.values = None
        block.on_disk = True
        assert block.nbytes == 0

    def test_touch_updates_access(self):
        block = make_kv_block()
        initial_time = block.last_access
        initial_count = block.access_count
        time.sleep(0.01)
        block.touch()
        assert block.last_access > initial_time
        assert block.access_count == initial_count + 1

    def test_pinned_block(self):
        block = make_kv_block()
        block.pinned = True
        assert block.pinned is True

    def test_page_out_and_in(self):
        block = make_kv_block(num_kv_heads=4, head_dim=64)
        original_keys = block.keys.copy()
        original_values = block.values.copy()
        original_bytes = block.nbytes

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name

        try:
            # Page out
            freed = block.page_out(path)
            assert freed == original_bytes
            assert block.on_disk is True
            assert block.keys is None
            assert block.values is None
            assert block.nbytes == 0

            # Page in
            loaded = block.page_in()
            assert loaded > 0
            assert block.on_disk is False
            assert block.keys is not None
            np.testing.assert_array_equal(block.keys, original_keys)
            np.testing.assert_array_equal(block.values, original_values)
        finally:
            os.unlink(path)

    def test_page_out_pinned_fails(self):
        block = make_kv_block()
        block.pinned = True
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            freed = block.page_out(path)
            assert freed == 0
            assert block.on_disk is False
            assert block.keys is not None
        finally:
            os.unlink(path)

    def test_page_in_not_on_disk(self):
        block = make_kv_block()
        loaded = block.page_in()
        assert loaded == 0

    def test_page_out_already_on_disk(self):
        block = make_kv_block()
        block.on_disk = True
        block.keys = None
        block.values = None
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            freed = block.page_out(f.name)
        assert freed == 0


# ═══════════════════════════════════════════════════════════════════════
# PagedKVCache Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPagedKVCache:

    def test_creation(self):
        cache = PagedKVCache(layer_idx=0, block_size=256, num_kv_heads=8, head_dim=128)
        assert cache.layer_idx == 0
        assert cache.block_size == 256
        assert cache.seq_len == 0
        assert cache.num_blocks == 0
        assert cache.nbytes == 0

    def test_single_append(self):
        cache = PagedKVCache(layer_idx=0, block_size=256, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=16, head_dim=64)
        block = cache.append(k, v)

        assert cache.seq_len == 16
        assert cache.num_blocks == 1
        assert block.seq_start == 0
        assert block.seq_end == 16
        assert block.keys.shape == (4, 16, 64)

    def test_multiple_appends_same_block(self):
        cache = PagedKVCache(layer_idx=0, block_size=256, num_kv_heads=4, head_dim=64)

        for _ in range(4):
            k, v = make_kv(num_kv_heads=4, n_tokens=32, head_dim=64)
            cache.append(k, v)

        # 4 × 32 = 128 tokens, fits in one block (size 256)
        assert cache.seq_len == 128
        assert cache.num_blocks == 1

    def test_append_spans_blocks(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)

        # Append 200 tokens — should span 4 blocks (64+64+64+8)
        k, v = make_kv(num_kv_heads=4, n_tokens=200, head_dim=64)
        cache.append(k, v)

        assert cache.seq_len == 200
        # Block starts: 0, 64, 128, 192
        assert cache.num_blocks == 4

    def test_append_fills_exactly_one_block(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=64, head_dim=64)
        cache.append(k, v)

        assert cache.seq_len == 64
        assert cache.num_blocks == 1

        # Next append should go into block 2
        k2, v2 = make_kv(num_kv_heads=4, n_tokens=1, head_dim=64)
        cache.append(k2, v2)
        assert cache.seq_len == 65
        assert cache.num_blocks == 2

    def test_get_kv_full_range(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=100, head_dim=64)
        cache.append(k, v)

        retrieved_k, retrieved_v = cache.get_kv()
        assert retrieved_k.shape == (4, 100, 64)
        assert retrieved_v.shape == (4, 100, 64)
        np.testing.assert_array_equal(retrieved_k, k)
        np.testing.assert_array_equal(retrieved_v, v)

    def test_get_kv_partial_range(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=128, head_dim=64)
        cache.append(k, v)

        # Get only tokens 32-96
        rk, rv = cache.get_kv(seq_start=32, seq_end=96)
        assert rk.shape == (4, 64, 64)
        np.testing.assert_array_equal(rk, k[:, 32:96, :])

    def test_get_kv_empty(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = cache.get_kv()
        assert k is None
        assert v is None

    def test_get_kv_invalid_range(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=32, head_dim=64)
        cache.append(k, v)

        rk, rv = cache.get_kv(seq_start=50, seq_end=50)
        assert rk is None

    def test_evict_block(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=128, head_dim=64)
        cache.append(k, v)

        assert cache.num_blocks == 2
        freed = cache.evict_block(0)
        assert freed > 0
        assert cache.num_blocks == 1
        assert 0 not in cache.blocks
        assert cache.total_evictions == 1

    def test_evict_nonexistent_block(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        freed = cache.evict_block(999)
        assert freed == 0

    def test_evict_pinned_block(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=64, head_dim=64)
        cache.append(k, v)
        cache.pin_range(0, 64)

        freed = cache.evict_block(0)
        assert freed == 0
        assert cache.num_blocks == 1

    def test_evict_oldest(self):
        cache = PagedKVCache(layer_idx=0, block_size=32, num_kv_heads=4, head_dim=64)
        # Add 3 blocks
        for _ in range(3):
            k, v = make_kv(num_kv_heads=4, n_tokens=32, head_dim=64)
            cache.append(k, v)

        assert cache.num_blocks == 3
        freed = cache.evict_oldest()
        assert freed > 0
        assert cache.num_blocks == 2
        assert 0 not in cache.blocks

    def test_pin_and_unpin_range(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=192, head_dim=64)
        cache.append(k, v)

        # Pin first two blocks (0-128)
        cache.pin_range(0, 128)
        assert cache.blocks[0].pinned is True
        assert cache.blocks[64].pinned is True
        assert cache.blocks[128].pinned is False

        # Unpin
        cache.unpin_range(0, 128)
        assert cache.blocks[0].pinned is False
        assert cache.blocks[64].pinned is False

    def test_clear(self):
        cache = PagedKVCache(layer_idx=0, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=128, head_dim=64)
        cache.append(k, v)

        cache.clear()
        assert cache.seq_len == 0
        assert cache.num_blocks == 0
        assert cache.nbytes == 0

    def test_stats(self):
        cache = PagedKVCache(layer_idx=3, block_size=64, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=100, head_dim=64)
        cache.append(k, v)

        s = cache.stats()
        assert s["layer_idx"] == 3
        assert s["seq_len"] == 100
        assert s["num_blocks"] == 2
        assert s["in_memory"] == 2
        assert s["on_disk"] == 0
        assert s["ram_bytes"] > 0
        assert s["total_appends"] == 100

    def test_nbytes_tracks_memory(self):
        cache = PagedKVCache(layer_idx=0, block_size=256, num_kv_heads=4, head_dim=64)
        assert cache.nbytes == 0

        k, v = make_kv(num_kv_heads=4, n_tokens=100, head_dim=64)
        cache.append(k, v)
        # 4 heads × 100 tokens × 64 dim × 2 bytes × 2 (K+V)
        expected = 4 * 100 * 64 * 2 * 2
        assert cache.nbytes == expected

    def test_multiple_appends_data_integrity(self):
        """Verify data survives multiple appends across blocks."""
        cache = PagedKVCache(layer_idx=0, block_size=32, num_kv_heads=2, head_dim=16)
        all_k = []
        all_v = []

        for i in range(5):
            k, v = make_kv(num_kv_heads=2, n_tokens=20, head_dim=16)
            all_k.append(k)
            all_v.append(v)
            cache.append(k, v)

        expected_k = np.concatenate(all_k, axis=1)
        expected_v = np.concatenate(all_v, axis=1)

        rk, rv = cache.get_kv()
        np.testing.assert_array_equal(rk, expected_k)
        np.testing.assert_array_equal(rv, expected_v)


# ═══════════════════════════════════════════════════════════════════════
# MoEKVCacheManager Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMoEKVCacheManager:

    def test_creation(self):
        mgr = MoEKVCacheManager(n_layers=4, num_kv_heads=4, head_dim=64)
        assert len(mgr.layers) == 4
        assert mgr.total_ram_bytes == 0
        assert mgr.total_seq_len == 0
        assert mgr.total_blocks == 0

    def test_append_kv_single_layer(self):
        mgr = MoEKVCacheManager(n_layers=4, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=16, head_dim=64)
        block = mgr.append_kv(0, k, v)

        assert block is not None
        assert mgr.layers[0].seq_len == 16
        assert mgr.total_ram_bytes > 0

    def test_append_kv_all_layers(self):
        mgr = MoEKVCacheManager(n_layers=4, num_kv_heads=4, head_dim=64, block_size=64)

        for layer in range(4):
            k, v = make_kv(num_kv_heads=4, n_tokens=32, head_dim=64)
            mgr.append_kv(layer, k, v)

        assert all(mgr.layers[i].seq_len == 32 for i in range(4))
        # 4 layers × 4 heads × 32 tokens × 64 dim × 2 bytes × 2 (K+V)
        expected = 4 * 4 * 32 * 64 * 2 * 2
        assert mgr.total_ram_bytes == expected

    def test_append_kv_invalid_layer(self):
        mgr = MoEKVCacheManager(n_layers=4)
        k, v = make_kv()
        with pytest.raises(IndexError):
            mgr.append_kv(5, k, v)
        with pytest.raises(IndexError):
            mgr.append_kv(-1, k, v)

    def test_get_kv(self):
        mgr = MoEKVCacheManager(n_layers=2, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=50, head_dim=64)
        mgr.append_kv(0, k, v)

        rk, rv = mgr.get_kv(0)
        np.testing.assert_array_equal(rk, k)
        np.testing.assert_array_equal(rv, v)

    def test_get_kv_invalid_layer(self):
        mgr = MoEKVCacheManager(n_layers=2)
        with pytest.raises(IndexError):
            mgr.get_kv(5)

    def test_budget_enforcement_evicts(self):
        """Test that appending beyond budget triggers eviction."""
        # Tiny budget: 1KB
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=1024,
        )

        # Each token uses: 2 heads × 8 dim × 2 bytes × 2 (K+V) = 64 bytes
        # 1024 / 64 = 16 tokens max
        # Append 32 tokens — should trigger eviction of the first block
        k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k, v)
        assert mgr.total_evictions == 0

        k2, v2 = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k2, v2)
        # Budget was exceeded, so the first block should be evicted
        assert mgr.total_evictions >= 1
        assert mgr.total_ram_bytes <= 1024

    def test_budget_utilization(self):
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=32, kv_budget_bytes=4096,
        )
        assert mgr.budget_utilization == 0.0

        k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k, v)
        assert 0.0 < mgr.budget_utilization <= 1.0

    def test_bytes_per_token_all_layers(self):
        mgr = MoEKVCacheManager(n_layers=4, num_kv_heads=8, head_dim=128)
        # 4 layers × 8 heads × 128 dim × 2 (K+V) × 2 bytes
        expected = 4 * 8 * 128 * 2 * 2
        assert mgr.bytes_per_token_all_layers() == expected

    def test_max_tokens_in_budget(self):
        mgr = MoEKVCacheManager(
            n_layers=4, num_kv_heads=8, head_dim=128,
            kv_budget_bytes=10 * 1024**3,
        )
        max_tokens = mgr.max_tokens_in_budget()
        bpt = mgr.bytes_per_token_all_layers()
        assert max_tokens == (10 * 1024**3) // bpt
        assert max_tokens > 0

    def test_lru_eviction_policy(self):
        """Verify LRU evicts the least recently accessed block."""
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=2048,
            eviction_policy=EvictionPolicy.LRU,
        )

        # Append 3 blocks of 16 tokens each (64 bytes each = 192 total)
        # Budget is 2048 so all fit initially
        for _ in range(3):
            k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
            mgr.append_kv(0, k, v)

        # Touch block at seq_start=16 to make it more recent
        mgr.get_kv(0, seq_start=16, seq_end=32)
        time.sleep(0.01)

        # Force eviction by lowering budget
        mgr.kv_budget_bytes = 64  # only room for 1 block

        k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k, v)

        # Block 0 (oldest, not recently touched) should be evicted first
        assert 0 not in mgr.layers[0].blocks
        assert mgr.total_evictions > 0

    def test_frequency_weighted_eviction(self):
        """Verify frequency-weighted evicts the least frequently accessed block."""
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=4096,
            eviction_policy=EvictionPolicy.FREQUENCY_WEIGHTED,
        )

        # Append 3 blocks
        for _ in range(3):
            k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
            mgr.append_kv(0, k, v)

        # Access block 0 many times to boost its frequency
        for _ in range(10):
            mgr.get_kv(0, seq_start=0, seq_end=16)
            time.sleep(0.001)

        # Shrink budget to force eviction
        mgr.kv_budget_bytes = 64
        k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k, v)

        # Block 0 was accessed most — should survive
        # Block 16 or 32 (less accessed) should be evicted first
        assert mgr.total_evictions > 0

    def test_belady_eviction_with_weights(self):
        """Test Belady-approximate eviction with mock predictor weights."""
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=4096,
            eviction_policy=EvictionPolicy.BELADY_APPROXIMATE,
        )

        # Set up tiny Belady predictor weights
        hidden_dim = 8
        w1 = np.random.randn(3, hidden_dim).astype(np.float32) * 0.1
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        b2 = np.zeros(hidden_dim, dtype=np.float32)
        w3 = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.1
        b3 = np.zeros(1, dtype=np.float32)
        mgr.set_belady_weights((w1, b1, w2, b2, w3, b3))

        # Append 3 blocks
        for _ in range(3):
            k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
            mgr.append_kv(0, k, v)

        # Shrink budget to force eviction
        mgr.kv_budget_bytes = 64
        k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k, v)

        # Eviction should have happened
        assert mgr.total_evictions > 0
        assert mgr.total_bytes_evicted > 0

    def test_belady_falls_back_without_weights(self):
        """Without weights, Belady falls back to frequency-weighted."""
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=128,
            eviction_policy=EvictionPolicy.BELADY_APPROXIMATE,
        )
        # No weights set — should use frequency-weighted fallback
        for _ in range(3):
            k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
            mgr.append_kv(0, k, v)

        assert mgr.total_evictions > 0  # eviction happened without crash

    def test_pin_system_prompt(self):
        mgr = MoEKVCacheManager(
            n_layers=2, num_kv_heads=2, head_dim=8,
            block_size=32, kv_budget_bytes=1024 * 1024,  # 1MB — plenty of room
        )
        k, v = make_kv(num_kv_heads=2, n_tokens=64, head_dim=8)
        mgr.append_kv(0, k, v)
        k, v = make_kv(num_kv_heads=2, n_tokens=64, head_dim=8)
        mgr.append_kv(1, k, v)

        mgr.pin_system_prompt(32)

        # Check that the first block of each layer is pinned
        for layer in mgr.layers:
            assert layer.blocks[0].pinned is True

    def test_evict_before(self):
        mgr = MoEKVCacheManager(
            n_layers=2, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=1024 * 1024,
        )

        for layer_idx in range(2):
            k, v = make_kv(num_kv_heads=2, n_tokens=48, head_dim=8)
            mgr.append_kv(layer_idx, k, v)

        # Each layer has 3 blocks: [0, 16, 32]
        freed = mgr.evict_before(32)
        assert freed > 0
        # Blocks at 0 and 16 should be evicted from both layers
        for layer in mgr.layers:
            assert 0 not in layer.blocks
            assert 16 not in layer.blocks
            assert 32 in layer.blocks

    def test_evict_before_respects_pinned(self):
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=1024 * 1024,
        )
        k, v = make_kv(num_kv_heads=2, n_tokens=48, head_dim=8)
        mgr.append_kv(0, k, v)

        mgr.pin_system_prompt(16)
        mgr.evict_before(32)

        # Block 0 is pinned — should survive
        assert 0 in mgr.layers[0].blocks
        # Block 16 is not pinned — should be evicted
        assert 16 not in mgr.layers[0].blocks

    def test_clear(self):
        mgr = MoEKVCacheManager(n_layers=2, num_kv_heads=2, head_dim=8)
        for i in range(2):
            k, v = make_kv(num_kv_heads=2, n_tokens=32, head_dim=8)
            mgr.append_kv(i, k, v)

        mgr.clear()
        assert mgr.total_ram_bytes == 0
        assert mgr.total_seq_len == 0
        assert mgr.total_blocks == 0

    def test_stats(self):
        mgr = MoEKVCacheManager(
            n_layers=2, num_kv_heads=4, head_dim=64,
            kv_budget_bytes=10 * 1024**3,
            eviction_policy=EvictionPolicy.FREQUENCY_WEIGHTED,
        )
        k, v = make_kv(num_kv_heads=4, n_tokens=50, head_dim=64)
        mgr.append_kv(0, k, v)

        s = mgr.stats()
        assert s["n_layers"] == 2
        assert s["total_ram_bytes"] > 0
        assert s["eviction_policy"] == "frequency_weighted"
        assert s["max_tokens_in_budget"] > 0

    def test_multi_layer_concurrent_append(self):
        """Simulate realistic multi-layer KV append (as in transformer forward pass)."""
        n_layers = 8
        mgr = MoEKVCacheManager(
            n_layers=n_layers, num_kv_heads=4, head_dim=32,
            block_size=128, kv_budget_bytes=1024 * 1024,
        )

        # Simulate 10 tokens of generation
        for token in range(10):
            for layer in range(n_layers):
                k, v = make_kv(num_kv_heads=4, n_tokens=1, head_dim=32)
                mgr.append_kv(layer, k, v)

        # All layers should have same seq_len
        for layer in mgr.layers:
            assert layer.seq_len == 10

    def test_eviction_cleans_up_tracking(self):
        """After eviction, access tracking data should be cleaned up."""
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=128,
        )

        k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k, v)
        mgr.get_kv(0, 0, 16)

        key = (0, 0)
        assert key in mgr._access_counts

        # Force eviction
        mgr.kv_budget_bytes = 1
        k2, v2 = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k2, v2)

        # Old block's tracking should be cleaned up
        assert key not in mgr._access_counts


# ═══════════════════════════════════════════════════════════════════════
# KVExpertBudgetCoordinator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestKVExpertBudgetCoordinator:

    def test_creation(self):
        coord = KVExpertBudgetCoordinator(total_budget_bytes=70 * 1024**3)
        assert coord.total_budget_bytes == 70 * 1024**3
        assert coord.kv_budget_bytes + coord.expert_budget_bytes == coord.total_budget_bytes
        assert abs(coord.kv_fraction - 0.15) < 0.001

    def test_initial_allocation(self):
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=100 * 1024**3,
            initial_kv_fraction=0.20,
        )
        assert coord.kv_budget_bytes == int(100 * 1024**3 * 0.20)
        assert coord.expert_budget_bytes == int(100 * 1024**3 * 0.80)

    def test_fractions_sum_to_one(self):
        coord = KVExpertBudgetCoordinator(total_budget_bytes=70 * 1024**3)
        assert abs(coord.kv_fraction + coord.expert_fraction - 1.0) < 0.001

    def test_rebalance_kv_pressure_high_expert_good(self):
        """When KV is full but expert cache is fine, give KV more room."""
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=100 * 1024**3,
            initial_kv_fraction=0.10,
        )
        old_kv = coord.kv_budget_bytes

        result = coord.rebalance(
            kv_ram_used=int(coord.kv_budget_bytes * 0.95),  # 95% full
            expert_hit_rate=0.90,  # above target
            seq_len=1000,
            bytes_per_token=16384,
        )

        assert result["shift_bytes"] > 0  # KV budget grew
        assert coord.kv_budget_bytes > old_kv

    def test_rebalance_expert_struggling_kv_slack(self):
        """When expert cache is struggling and KV has slack, give experts more."""
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=100 * 1024**3,
            initial_kv_fraction=0.30,
            expert_hit_rate_target=0.85,
        )
        old_expert = coord.expert_budget_bytes

        result = coord.rebalance(
            kv_ram_used=int(coord.kv_budget_bytes * 0.3),  # 30% full
            expert_hit_rate=0.60,  # well below target
            seq_len=100,
            bytes_per_token=16384,
        )

        assert result["shift_bytes"] < 0  # KV budget shrank
        assert coord.expert_budget_bytes > old_expert

    def test_rebalance_respects_min_kv_fraction(self):
        """KV budget should not shrink below min_kv_fraction."""
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=100 * 1024**3,
            initial_kv_fraction=0.06,  # just above min
            min_kv_fraction=0.05,
        )

        # Even with extreme pressure to shrink KV, it stays above min
        for _ in range(50):
            coord.rebalance(
                kv_ram_used=0,
                expert_hit_rate=0.50,
                seq_len=10,
                bytes_per_token=1024,
            )

        min_kv = int(100 * 1024**3 * 0.05)
        assert coord.kv_budget_bytes >= min_kv

    def test_rebalance_respects_max_kv_fraction(self):
        """KV budget should not grow above max_kv_fraction."""
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=100 * 1024**3,
            initial_kv_fraction=0.35,  # near max
            max_kv_fraction=0.40,
        )

        for _ in range(50):
            coord.rebalance(
                kv_ram_used=coord.kv_budget_bytes,
                expert_hit_rate=0.99,
                seq_len=10000,
                bytes_per_token=16384,
            )

        max_kv = int(100 * 1024**3 * 0.40)
        assert coord.kv_budget_bytes <= max_kv

    def test_rebalance_budgets_sum_correctly(self):
        """After rebalance, KV + expert should still equal total."""
        coord = KVExpertBudgetCoordinator(total_budget_bytes=70 * 1024**3)

        for i in range(10):
            coord.rebalance(
                kv_ram_used=int(coord.kv_budget_bytes * (0.3 + i * 0.07)),
                expert_hit_rate=0.80 - i * 0.02,
                seq_len=100 * (i + 1),
                bytes_per_token=8192,
            )
            assert coord.kv_budget_bytes + coord.expert_budget_bytes == coord.total_budget_bytes

    def test_expert_cache_slots(self):
        coord = KVExpertBudgetCoordinator(total_budget_bytes=70 * 1024**3)
        expert_bytes = 14 * 1024 * 1024  # 14MB per expert
        slots = coord.expert_cache_slots(expert_bytes)
        assert slots == coord.expert_budget_bytes // expert_bytes
        assert slots > 0

    def test_expert_cache_slots_zero_bytes(self):
        coord = KVExpertBudgetCoordinator(total_budget_bytes=70 * 1024**3)
        assert coord.expert_cache_slots(0) == 0

    def test_rebalance_history(self):
        coord = KVExpertBudgetCoordinator(total_budget_bytes=70 * 1024**3)
        coord.rebalance(kv_ram_used=0, expert_hit_rate=0.9, seq_len=100, bytes_per_token=1024)
        coord.rebalance(kv_ram_used=0, expert_hit_rate=0.9, seq_len=200, bytes_per_token=1024)

        assert len(coord._history) == 2
        assert coord._rebalance_count == 2

    def test_stats(self):
        coord = KVExpertBudgetCoordinator(total_budget_bytes=70 * 1024**3)
        s = coord.stats()
        assert s["total_budget_gb"] == 70.0
        assert s["kv_budget_gb"] > 0
        assert s["expert_budget_gb"] > 0
        assert abs(s["kv_fraction"] + s["expert_fraction"] - 1.0) < 0.01

    def test_zero_budget(self):
        coord = KVExpertBudgetCoordinator(total_budget_bytes=0)
        assert coord.kv_fraction == 0.0
        assert coord.kv_budget_bytes == 0
        assert coord.expert_budget_bytes == 0


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests — KV + Expert Cache Coordination
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_realistic_deepseek_v3_budget(self):
        """Simulate DeepSeek V3 memory budget on 128GB M5 Max."""
        total_budget = 70 * 1024**3  # 70GB for KV + experts (rest is attn + OS)
        expert_bytes = 24 * 1024 * 1024  # ~24MB per expert (gate+up+down Q2_K)

        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=total_budget,
            initial_kv_fraction=0.14,  # ~10GB KV
            min_kv_fraction=0.05,
            max_kv_fraction=0.30,
            expert_hit_rate_target=0.85,
        )

        # Initial state: ~10GB KV, ~60GB experts
        initial_expert_slots = coord.expert_cache_slots(expert_bytes)
        assert initial_expert_slots > 2000  # Should hold 2000+ experts

        # Create KV manager with initial budget
        mgr = MoEKVCacheManager(
            n_layers=61,
            num_kv_heads=8,
            head_dim=128,
            block_size=256,
            kv_budget_bytes=coord.kv_budget_bytes,
        )

        bpt = mgr.bytes_per_token_all_layers()
        max_tokens = mgr.max_tokens_in_budget()
        assert max_tokens > 0
        assert bpt > 0

    def test_kv_expert_tradeoff_simulation(self):
        """Simulate growing sequence and observe KV/expert budget tradeoff."""
        coord = KVExpertBudgetCoordinator(
            total_budget_bytes=10 * 1024**3,  # 10GB total (small for testing)
            initial_kv_fraction=0.15,
            expert_hit_rate_target=0.85,
        )

        expert_bytes = 14 * 1024 * 1024  # 14MB per expert
        bpt = 4 * 8 * 128 * 2 * 2  # 4 layers, 8 heads, 128 dim

        initial_slots = coord.expert_cache_slots(expert_bytes)

        # Simulate sequence growing — KV needs more, expert hit rate is high
        for step in range(10):
            seq_len = (step + 1) * 500
            kv_used = int(bpt * seq_len * 0.8)

            coord.rebalance(
                kv_ram_used=min(kv_used, coord.kv_budget_bytes),
                expert_hit_rate=0.92,  # good hit rate
                seq_len=seq_len,
                bytes_per_token=bpt,
            )

        # KV budget should have grown (since hit rate was good)
        assert coord.kv_budget_bytes >= int(coord.total_budget_bytes * coord.initial_kv_fraction)
        # Budgets still sum correctly
        assert coord.kv_budget_bytes + coord.expert_budget_bytes == coord.total_budget_bytes

    def test_sliding_window_with_eviction(self):
        """Simulate sliding window attention: evict old context as we generate."""
        mgr = MoEKVCacheManager(
            n_layers=4, num_kv_heads=2, head_dim=16,
            block_size=32, kv_budget_bytes=1024 * 1024,
        )

        # Generate 200 tokens across all layers
        for token in range(200):
            for layer in range(4):
                k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=16)
                mgr.append_kv(layer, k, v)

            # Every 64 tokens, evict context older than 64 tokens ago
            if token > 0 and token % 64 == 0:
                mgr.evict_before(token - 64)

        # Should have some evictions
        assert mgr.total_evictions > 0
        # Recent context should still be accessible
        for layer in range(4):
            k, v = mgr.get_kv(layer, seq_start=180, seq_end=200)
            assert k is not None
            assert k.shape[1] == 20

    def test_pin_system_prompt_survives_pressure(self):
        """System prompt stays pinned even under memory pressure."""
        # Budget: room for system prompt (16 tokens × 2 layers) plus a few more
        # Each token per layer = 2 heads × 8 dim × 2 bytes × 2 (K+V) = 64 bytes
        # System prompt = 16 tokens × 2 layers × 64 = 2048 bytes
        # Budget = 3072 → room for prompt + ~16 extra tokens before evictions start
        mgr = MoEKVCacheManager(
            n_layers=2, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=3072,
        )

        # Add system prompt (16 tokens) and pin BEFORE adding more
        for layer in range(2):
            k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
            mgr.append_kv(layer, k, v)
        mgr.pin_system_prompt(16)

        # Add more tokens to cause pressure — budget will overflow
        for token in range(50):
            for layer in range(2):
                k, v = make_kv(num_kv_heads=2, n_tokens=1, head_dim=8)
                mgr.append_kv(layer, k, v)

        # System prompt block should still exist (pinned, not evictable)
        for layer in mgr.layers:
            assert 0 in layer.blocks
            assert layer.blocks[0].pinned is True
        # Non-pinned blocks should have been evicted
        assert mgr.total_evictions > 0

    def test_all_eviction_policies_work(self):
        """Verify all three eviction policies work end-to-end."""
        for policy in EvictionPolicy:
            mgr = MoEKVCacheManager(
                n_layers=1, num_kv_heads=2, head_dim=8,
                block_size=16, kv_budget_bytes=128,
                eviction_policy=policy,
            )

            if policy == EvictionPolicy.BELADY_APPROXIMATE:
                # Set up mock weights for Belady
                h = 8
                mgr.set_belady_weights((
                    np.random.randn(3, h).astype(np.float32) * 0.1,
                    np.zeros(h, dtype=np.float32),
                    np.random.randn(h, h).astype(np.float32) * 0.1,
                    np.zeros(h, dtype=np.float32),
                    np.random.randn(h, 1).astype(np.float32) * 0.1,
                    np.zeros(1, dtype=np.float32),
                ))

            # Fill and overflow
            for _ in range(5):
                k, v = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
                mgr.append_kv(0, k, v)

            assert mgr.total_evictions > 0, f"Policy {policy} failed to evict"


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_token_append(self):
        cache = PagedKVCache(layer_idx=0, block_size=256, num_kv_heads=4, head_dim=64)
        k, v = make_kv(num_kv_heads=4, n_tokens=1, head_dim=64)
        cache.append(k, v)
        assert cache.seq_len == 1
        rk, rv = cache.get_kv()
        assert rk.shape == (4, 1, 64)

    def test_large_batch_append(self):
        """Append a large batch that spans many blocks."""
        cache = PagedKVCache(layer_idx=0, block_size=32, num_kv_heads=2, head_dim=16)
        k, v = make_kv(num_kv_heads=2, n_tokens=1000, head_dim=16)
        cache.append(k, v)

        assert cache.seq_len == 1000
        assert cache.num_blocks == 32  # ceil(1000/32)
        rk, rv = cache.get_kv()
        np.testing.assert_array_equal(rk, k)

    def test_zero_layers_manager(self):
        mgr = MoEKVCacheManager(n_layers=0)
        assert mgr.total_ram_bytes == 0
        assert mgr.total_seq_len == 0

    def test_evict_from_empty_cache(self):
        mgr = MoEKVCacheManager(n_layers=2)
        freed = mgr.evict_before(100)
        assert freed == 0

    def test_coordinator_repeated_no_change_rebalance(self):
        """When nothing changes, budget should stay stable."""
        coord = KVExpertBudgetCoordinator(total_budget_bytes=70 * 1024**3)
        initial_kv = coord.kv_budget_bytes

        # Moderate usage, at target — no reason to shift
        for _ in range(5):
            coord.rebalance(
                kv_ram_used=int(coord.kv_budget_bytes * 0.6),
                expert_hit_rate=0.85,  # exactly at target
                seq_len=500,
                bytes_per_token=8192,
            )

        # Should be close to initial (small drift OK)
        drift = abs(coord.kv_budget_bytes - initial_kv) / initial_kv
        assert drift < 0.05  # less than 5% drift

    def test_get_kv_after_partial_eviction(self):
        """get_kv should return data from remaining blocks after eviction."""
        mgr = MoEKVCacheManager(
            n_layers=1, num_kv_heads=2, head_dim=8,
            block_size=16, kv_budget_bytes=1024 * 1024,
        )

        k1, v1 = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        k2, v2 = make_kv(num_kv_heads=2, n_tokens=16, head_dim=8)
        mgr.append_kv(0, k1, v1)
        mgr.append_kv(0, k2, v2)

        # Evict first block
        mgr.layers[0].evict_block(0)

        # Getting full range should return only second block
        rk, rv = mgr.get_kv(0, seq_start=16, seq_end=32)
        assert rk is not None
        assert rk.shape == (2, 16, 8)
        np.testing.assert_array_equal(rk, k2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
