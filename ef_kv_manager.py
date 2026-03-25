#!/usr/bin/env python3
"""
ExpertFlow KV Cache Manager for Expert-Level MoE Inference
==========================================================

Manages KV cache memory alongside expert weight caching for MoE models
where both compete for the same unified memory budget (128GB on M5 Max).

Key innovations:
  1. Paged KV cache: fixed-size blocks instead of growing concatenation —
     enables block-level eviction without recomputing entire sequences.
  2. Memory coordinator: dynamically rebalances RAM between KV cache and
     expert weight cache based on sequence length and cache hit rates.
  3. Eviction policies: LRU, frequency-weighted, and Belady-approximate
     eviction that understands MoE routing patterns.
  4. Tiered storage: hot KV blocks in RAM, cold blocks on NVMe (mmap'd),
     restorable without recomputation.

Memory layout integration:
  - Attention + embeddings (pinned): ~30GB — always in RAM
  - Expert cache (hot):              ~60GB — managed by GGUFExpertCache
  - KV cache (dynamic):             ~10GB — managed by THIS module
  - OS + overhead:                   ~28GB
  - Expert weights on NVMe:         ~200GB — mmap'd, paged on demand

The KV budget is dynamic: as sequence length grows, KV cache needs more
RAM, which can be reclaimed from the expert cache budget (fewer cached
experts → more cache misses → slower, but necessary for long contexts).
"""

import time
import threading
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Part 1: KV Block — Fixed-size unit of KV cache storage
# ═══════════════════════════════════════════════════════════════════════

class EvictionPolicy(Enum):
    """Supported cache eviction policies."""
    LRU = "lru"
    FREQUENCY_WEIGHTED = "frequency_weighted"
    BELADY_APPROXIMATE = "belady_approximate"


@dataclass
class KVBlock:
    """Fixed-size block of K/V tensors for a single layer.

    Instead of concatenating KV tensors indefinitely (which fragments
    memory and makes eviction impossible), we store KV in fixed-size
    blocks. Each block holds `block_size` token positions worth of K/V.

    Attributes:
        layer_idx: Which transformer layer this block belongs to.
        seq_start: Starting sequence position (inclusive).
        seq_end: Ending sequence position (exclusive).
        keys: K tensor, shape (num_kv_heads, block_size, head_dim), float16.
        values: V tensor, shape (num_kv_heads, block_size, head_dim), float16.
        last_access: Timestamp of last access (for LRU eviction).
        access_count: Number of times this block was accessed.
        pinned: If True, this block cannot be evicted (e.g. system prompt).
        on_disk: If True, this block has been paged to NVMe.
        disk_path: Path to NVMe-backed storage if paged out.
    """
    layer_idx: int
    seq_start: int
    seq_end: int
    keys: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    last_access: float = field(default_factory=time.monotonic)
    access_count: int = 0
    pinned: bool = False
    on_disk: bool = False
    disk_path: Optional[str] = None

    @property
    def block_size(self) -> int:
        return self.seq_end - self.seq_start

    @property
    def nbytes(self) -> int:
        """Total bytes used by K and V tensors in RAM."""
        if self.on_disk or self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    @property
    def block_id(self) -> tuple:
        """Unique identifier: (layer_idx, seq_start)."""
        return (self.layer_idx, self.seq_start)

    def touch(self):
        """Record an access to this block."""
        self.last_access = time.monotonic()
        self.access_count += 1

    def page_out(self, path: str) -> int:
        """Save K/V to disk and free RAM. Returns bytes freed."""
        if self.pinned or self.on_disk or self.keys is None:
            return 0
        freed = self.nbytes
        np.savez_compressed(path, keys=self.keys, values=self.values)
        self.disk_path = path
        self.keys = None
        self.values = None
        self.on_disk = True
        return freed

    def page_in(self) -> int:
        """Restore K/V from disk into RAM. Returns bytes loaded."""
        if not self.on_disk or self.disk_path is None:
            return 0
        data = np.load(self.disk_path)
        self.keys = data['keys']
        self.values = data['values']
        self.on_disk = False
        self.touch()
        return self.nbytes


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Paged KV Cache — Block-based KV storage per layer
# ═══════════════════════════════════════════════════════════════════════

class PagedKVCache:
    """Paged KV cache for a single transformer layer.

    Stores K/V tensors in fixed-size blocks instead of a single growing
    tensor. This enables:
      - Block-level eviction: free old context without recomputing
      - Memory-bounded operation: cap KV RAM per layer
      - Tiered storage: page cold blocks to NVMe

    Args:
        layer_idx: Which transformer layer this cache serves.
        block_size: Number of token positions per block (default 256).
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
    """

    def __init__(self, layer_idx: int, block_size: int = 256,
                 num_kv_heads: int = 8, head_dim: int = 128):
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.blocks: OrderedDict[int, KVBlock] = OrderedDict()  # seq_start -> KVBlock
        self.seq_len = 0  # total tokens stored (including paged-out)

        # Stats
        self.total_appends = 0
        self.total_evictions = 0

    @property
    def nbytes(self) -> int:
        """Total RAM bytes used by in-memory blocks."""
        return sum(b.nbytes for b in self.blocks.values())

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_in_memory(self) -> int:
        return sum(1 for b in self.blocks.values() if not b.on_disk)

    @property
    def num_on_disk(self) -> int:
        return sum(1 for b in self.blocks.values() if b.on_disk)

    def _current_block_start(self) -> int:
        """Sequence position where the current (partially filled) block starts."""
        return (self.seq_len // self.block_size) * self.block_size

    def append(self, keys: np.ndarray, values: np.ndarray) -> KVBlock:
        """Append new K/V for one or more tokens to the cache.

        Args:
            keys: shape (num_kv_heads, n_tokens, head_dim), float16.
            values: shape (num_kv_heads, n_tokens, head_dim), float16.

        Returns:
            The KVBlock that was written to (may be new or existing).
        """
        n_tokens = keys.shape[1]
        self.total_appends += n_tokens

        tokens_written = 0
        last_block = None

        while tokens_written < n_tokens:
            block_start = self._current_block_start()

            if block_start in self.blocks:
                block = self.blocks[block_start]
                # Append to existing partial block
                existing_len = self.seq_len - block_start
                room = self.block_size - existing_len
                take = min(room, n_tokens - tokens_written)

                new_k = keys[:, tokens_written:tokens_written + take, :]
                new_v = values[:, tokens_written:tokens_written + take, :]

                if block.on_disk:
                    block.page_in()

                block.keys = np.concatenate([block.keys, new_k], axis=1)
                block.values = np.concatenate([block.values, new_v], axis=1)
                block.seq_end = block.seq_start + existing_len + take
                block.touch()
            else:
                # New block
                take = min(self.block_size, n_tokens - tokens_written)
                new_k = keys[:, tokens_written:tokens_written + take, :]
                new_v = values[:, tokens_written:tokens_written + take, :]

                block = KVBlock(
                    layer_idx=self.layer_idx,
                    seq_start=block_start,
                    seq_end=block_start + take,
                    keys=new_k,
                    values=new_v,
                )
                self.blocks[block_start] = block

            tokens_written += take
            self.seq_len += take
            last_block = block

        return last_block

    def get_kv(self, seq_start: int = 0, seq_end: Optional[int] = None
               ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get concatenated K/V tensors for a sequence range.

        Pages in blocks from disk as needed. Touches accessed blocks.

        Returns:
            (keys, values) each of shape (num_kv_heads, seq_len, head_dim),
            or (None, None) if the range is empty.
        """
        if seq_end is None:
            seq_end = self.seq_len
        if seq_end <= seq_start:
            return None, None

        k_parts = []
        v_parts = []

        for block_start, block in self.blocks.items():
            if block.seq_end <= seq_start or block.seq_start >= seq_end:
                continue

            if block.on_disk:
                block.page_in()

            if block.keys is None:
                continue

            block.touch()

            # Slice if only part of the block is needed
            local_start = max(0, seq_start - block.seq_start)
            local_end = min(block.seq_end - block.seq_start,
                            seq_end - block.seq_start)

            k_parts.append(block.keys[:, local_start:local_end, :])
            v_parts.append(block.values[:, local_start:local_end, :])

        if not k_parts:
            return None, None

        return np.concatenate(k_parts, axis=1), np.concatenate(v_parts, axis=1)

    def evict_block(self, seq_start: int) -> int:
        """Evict a specific block from RAM. Returns bytes freed."""
        if seq_start not in self.blocks:
            return 0
        block = self.blocks[seq_start]
        if block.pinned:
            return 0
        freed = block.nbytes
        del self.blocks[seq_start]
        self.total_evictions += 1
        return freed

    def evict_oldest(self) -> int:
        """Evict the oldest non-pinned in-memory block. Returns bytes freed."""
        for block_start, block in self.blocks.items():
            if not block.pinned and not block.on_disk:
                return self.evict_block(block_start)
        return 0

    def pin_range(self, seq_start: int, seq_end: int):
        """Pin blocks covering a sequence range (e.g. system prompt)."""
        for block_start, block in self.blocks.items():
            if block.seq_end > seq_start and block.seq_start < seq_end:
                block.pinned = True

    def unpin_range(self, seq_start: int, seq_end: int):
        """Unpin blocks covering a sequence range."""
        for block_start, block in self.blocks.items():
            if block.seq_end > seq_start and block.seq_start < seq_end:
                block.pinned = False

    def clear(self):
        """Remove all blocks and reset state."""
        self.blocks.clear()
        self.seq_len = 0

    def stats(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "seq_len": self.seq_len,
            "num_blocks": self.num_blocks,
            "in_memory": self.num_in_memory,
            "on_disk": self.num_on_disk,
            "ram_bytes": self.nbytes,
            "ram_mb": round(self.nbytes / 1024**2, 1),
            "total_appends": self.total_appends,
            "total_evictions": self.total_evictions,
        }


# ═══════════════════════════════════════════════════════════════════════
# Part 3: MoE KV Cache Manager — Coordinates all layers + expert cache
# ═══════════════════════════════════════════════════════════════════════

class MoEKVCacheManager:
    """Manages KV cache across all layers for MoE inference.

    Coordinates memory between KV cache and expert weight cache.
    Implements eviction policies that understand the relationship between
    attention (KV) and expert (weight) memory budgets.

    For DeepSeek V3 on 128GB M5 Max:
      - 61 layers total, 58 MoE layers
      - Each layer's KV: num_kv_heads × seq_len × head_dim × 2 × 2 bytes
      - At 4K tokens: ~61 × 128 × 4096 × 128 × 2 × 2 ≈ 16GB KV
      - At 1K tokens: ~4GB — leaving more room for expert cache

    The manager dynamically trades off KV capacity vs expert cache capacity:
      - Short sequences → small KV, large expert cache → high hit rate
      - Long sequences → large KV, smaller expert cache → lower hit rate
      - The breakeven point depends on expert access patterns

    Args:
        n_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads per layer.
        head_dim: Dimension per attention head.
        block_size: Tokens per KV block (default 256).
        kv_budget_bytes: Maximum RAM for KV cache across all layers.
        eviction_policy: Which eviction policy to use.
    """

    def __init__(self, n_layers: int, num_kv_heads: int = 8,
                 head_dim: int = 128, block_size: int = 256,
                 kv_budget_bytes: int = 10 * 1024**3,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.n_layers = n_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.kv_budget_bytes = kv_budget_bytes
        self.eviction_policy = eviction_policy

        # Per-layer paged KV caches
        self.layers: list[PagedKVCache] = [
            PagedKVCache(
                layer_idx=i,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            for i in range(n_layers)
        ]

        # Global access tracking for eviction decisions
        self._access_history: list[tuple[int, int, float]] = []  # (layer, seq_start, time)
        self._access_counts: dict[tuple[int, int], int] = {}  # (layer, seq_start) -> count
        self._gap_ema: dict[tuple[int, int], float] = {}  # exponential moving avg of access gaps
        self._last_access_time: dict[tuple[int, int], float] = {}

        # Belady predictor weights (if using Belady-approximate policy)
        self._belady_weights = None

        # Stats
        self.total_evictions = 0
        self.total_bytes_evicted = 0
        self.total_page_outs = 0
        self.total_page_ins = 0
        self._lock = threading.Lock()

    @property
    def total_ram_bytes(self) -> int:
        """Total RAM used by KV cache across all layers."""
        return sum(layer.nbytes for layer in self.layers)

    @property
    def total_seq_len(self) -> int:
        """Sequence length (same across all layers in standard transformers)."""
        if not self.layers:
            return 0
        return self.layers[0].seq_len

    @property
    def total_blocks(self) -> int:
        return sum(layer.num_blocks for layer in self.layers)

    @property
    def budget_utilization(self) -> float:
        """Fraction of KV budget currently used (0.0 to 1.0+)."""
        if self.kv_budget_bytes <= 0:
            return 0.0
        return self.total_ram_bytes / self.kv_budget_bytes

    def bytes_per_token_all_layers(self) -> int:
        """Bytes consumed per token across all layers.

        For float16 KV: n_layers × num_kv_heads × head_dim × 2 (K+V) × 2 (bytes)
        """
        return self.n_layers * self.num_kv_heads * self.head_dim * 2 * 2

    def max_tokens_in_budget(self) -> int:
        """Maximum sequence length that fits within the KV budget."""
        bpt = self.bytes_per_token_all_layers()
        if bpt <= 0:
            return 0
        return self.kv_budget_bytes // bpt

    def append_kv(self, layer_idx: int, keys: np.ndarray, values: np.ndarray) -> KVBlock:
        """Append KV for new tokens to a specific layer.

        Automatically enforces the memory budget by evicting old blocks
        if the budget would be exceeded.

        Args:
            layer_idx: Which layer to append to.
            keys: shape (num_kv_heads, n_tokens, head_dim), float16.
            values: shape (num_kv_heads, n_tokens, head_dim), float16.

        Returns:
            The KVBlock that was written to.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")

        # Estimate how much RAM this append will need
        new_bytes = keys.nbytes + values.nbytes

        # Evict if we'd exceed budget
        with self._lock:
            while (self.total_ram_bytes + new_bytes > self.kv_budget_bytes
                   and self._evict_one()):
                pass

        block = self.layers[layer_idx].append(keys, values)

        # Track access
        self._record_access(layer_idx, block.seq_start)

        return block

    def get_kv(self, layer_idx: int, seq_start: int = 0,
               seq_end: Optional[int] = None
               ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get K/V tensors for a layer's sequence range.

        Touches accessed blocks to update eviction metadata.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")

        k, v = self.layers[layer_idx].get_kv(seq_start, seq_end)

        # Track access for blocks in range
        if k is not None:
            block_start = (seq_start // self.block_size) * self.block_size
            while block_start < (seq_end or self.layers[layer_idx].seq_len):
                self._record_access(layer_idx, block_start)
                block_start += self.block_size

        return k, v

    def _record_access(self, layer_idx: int, seq_start: int):
        """Record a block access for eviction scoring."""
        key = (layer_idx, seq_start)
        now = time.monotonic()

        count = self._access_counts.get(key, 0) + 1
        self._access_counts[key] = count

        if key in self._last_access_time:
            gap = now - self._last_access_time[key]
            old_ema = self._gap_ema.get(key, gap)
            self._gap_ema[key] = 0.7 * old_ema + 0.3 * gap

        self._last_access_time[key] = now

    def _evict_one(self) -> bool:
        """Evict one block according to the current eviction policy.

        Returns True if a block was evicted, False if nothing evictable.
        """
        candidate = self._pick_eviction_candidate()
        if candidate is None:
            return False

        layer_idx, seq_start = candidate
        freed = self.layers[layer_idx].evict_block(seq_start)
        if freed > 0:
            self.total_evictions += 1
            self.total_bytes_evicted += freed
            # Clean up tracking for evicted block
            key = (layer_idx, seq_start)
            self._access_counts.pop(key, None)
            self._gap_ema.pop(key, None)
            self._last_access_time.pop(key, None)
            return True
        return False

    def _pick_eviction_candidate(self) -> Optional[tuple[int, int]]:
        """Select the best block to evict based on the eviction policy."""
        candidates = []
        for layer in self.layers:
            for seq_start, block in layer.blocks.items():
                if block.pinned or block.on_disk:
                    continue
                candidates.append((layer.layer_idx, seq_start, block))

        if not candidates:
            return None

        if self.eviction_policy == EvictionPolicy.LRU:
            return self._pick_lru(candidates)
        elif self.eviction_policy == EvictionPolicy.FREQUENCY_WEIGHTED:
            return self._pick_frequency_weighted(candidates)
        elif self.eviction_policy == EvictionPolicy.BELADY_APPROXIMATE:
            return self._pick_belady(candidates)
        else:
            return self._pick_lru(candidates)

    def _pick_lru(self, candidates: list) -> tuple[int, int]:
        """Least Recently Used: evict the block with oldest last_access."""
        best = min(candidates, key=lambda c: c[2].last_access)
        return (best[0], best[1])

    def _pick_frequency_weighted(self, candidates: list) -> tuple[int, int]:
        """Frequency-weighted: score = frequency × recency. Evict lowest."""
        now = time.monotonic()
        scored = []
        for layer_idx, seq_start, block in candidates:
            recency = max(now - block.last_access, 0.001)
            freq = block.access_count
            # Higher score = more valuable = keep longer
            score = freq / recency
            scored.append((layer_idx, seq_start, score))
        best = min(scored, key=lambda c: c[2])
        return (best[0], best[1])

    def _pick_belady(self, candidates: list) -> tuple[int, int]:
        """Belady-approximate: use learned features to predict next access.

        Features per block: [1/recency, freq/max_freq, avg_gap/max_gap]
        Higher eviction score = evict first (will be needed furthest in future).
        Falls back to frequency-weighted when no predictor is loaded.
        """
        if self._belady_weights is None:
            return self._pick_frequency_weighted(candidates)

        now = time.monotonic()
        max_freq = max((b.access_count for _, _, b in candidates), default=1) or 1
        max_gap = max(self._gap_ema.values(), default=1.0) or 1.0

        features = np.zeros((len(candidates), 3), dtype=np.float32)
        for i, (layer_idx, seq_start, block) in enumerate(candidates):
            recency = max(now - block.last_access, 0.001)
            key = (layer_idx, seq_start)
            features[i, 0] = 1.0 / recency
            features[i, 1] = block.access_count / max_freq
            features[i, 2] = self._gap_ema.get(key, 1.0) / max_gap

        # Forward pass through Belady predictor (3-layer FFN)
        w1, b1, w2, b2, w3, b3 = self._belady_weights
        z1 = features @ w1 + b1
        h1 = z1 / (1.0 + np.exp(-np.clip(z1, -20, 20)))  # SiLU
        z2 = h1 @ w2 + b2
        h2 = z2 / (1.0 + np.exp(-np.clip(z2, -20, 20)))
        scores = (h2 @ w3 + b3).squeeze(-1)

        # Highest eviction score → evict
        victim_idx = int(np.argmax(scores))
        return (candidates[victim_idx][0], candidates[victim_idx][1])

    def load_belady_weights(self, path: str):
        """Load trained Belady predictor weights from .npz file."""
        data = np.load(path)
        self._belady_weights = (
            data['w1'], data['b1'],
            data['w2'], data['b2'],
            data['w3'], data['b3'],
        )

    def set_belady_weights(self, weights: tuple):
        """Set Belady predictor weights directly (for testing)."""
        self._belady_weights = weights

    def pin_system_prompt(self, n_tokens: int):
        """Pin the first n_tokens across all layers (system prompt)."""
        for layer in self.layers:
            layer.pin_range(0, n_tokens)

    def unpin_system_prompt(self, n_tokens: int):
        """Unpin the first n_tokens across all layers."""
        for layer in self.layers:
            layer.unpin_range(0, n_tokens)

    def evict_before(self, seq_pos: int):
        """Evict all unpinned blocks before a sequence position.

        Useful for sliding window attention where old context is no longer needed.
        """
        freed = 0
        for layer in self.layers:
            to_evict = []
            for seq_start, block in layer.blocks.items():
                if block.seq_end <= seq_pos and not block.pinned:
                    to_evict.append(seq_start)
            for seq_start in to_evict:
                freed += layer.evict_block(seq_start)
                self.total_evictions += 1
        self.total_bytes_evicted += freed
        return freed

    def clear(self):
        """Clear all KV caches across all layers."""
        for layer in self.layers:
            layer.clear()
        self._access_counts.clear()
        self._gap_ema.clear()
        self._last_access_time.clear()
        self._access_history.clear()

    def stats(self) -> dict:
        """Global cache statistics."""
        return {
            "n_layers": self.n_layers,
            "total_seq_len": self.total_seq_len,
            "total_blocks": self.total_blocks,
            "total_ram_bytes": self.total_ram_bytes,
            "total_ram_mb": round(self.total_ram_bytes / 1024**2, 1),
            "budget_bytes": self.kv_budget_bytes,
            "budget_mb": round(self.kv_budget_bytes / 1024**2, 1),
            "budget_utilization": round(self.budget_utilization, 3),
            "total_evictions": self.total_evictions,
            "total_bytes_evicted": self.total_bytes_evicted,
            "eviction_policy": self.eviction_policy.value,
            "max_tokens_in_budget": self.max_tokens_in_budget(),
        }


# ═══════════════════════════════════════════════════════════════════════
# Part 4: KV-Expert Budget Coordinator
# ═══════════════════════════════════════════════════════════════════════

class KVExpertBudgetCoordinator:
    """Dynamically rebalances memory between KV cache and expert weight cache.

    The core tension in MoE inference on limited RAM:
      - More KV cache → longer context, but fewer cached experts → slower MoE
      - More expert cache → faster MoE (fewer NVMe reads), but shorter context

    This coordinator monitors both caches and adjusts budgets based on:
      1. Sequence length growth: as context grows, KV needs more RAM
      2. Expert cache hit rate: if hit rate is high, expert cache has slack
      3. Token generation phase: prefill needs more KV, decode more expert cache

    Budget rebalancing formula:
      - base_kv_budget: initial KV allocation (e.g. 10GB)
      - kv_growth = bytes_per_token × seq_len_growth
      - expert_slack = expert_budget × (1 - utilization)
      - rebalance: steal min(kv_growth, expert_slack) from expert budget

    Args:
        total_budget_bytes: Total RAM available for both KV + expert cache.
        initial_kv_fraction: Initial fraction allocated to KV (default 0.15).
        min_kv_fraction: Minimum fraction that KV can shrink to.
        max_kv_fraction: Maximum fraction that KV can grow to.
        expert_hit_rate_target: Target expert cache hit rate (0.0 to 1.0).
    """

    def __init__(self, total_budget_bytes: int = 70 * 1024**3,
                 initial_kv_fraction: float = 0.15,
                 min_kv_fraction: float = 0.05,
                 max_kv_fraction: float = 0.40,
                 expert_hit_rate_target: float = 0.85):
        self.total_budget_bytes = total_budget_bytes
        self.initial_kv_fraction = initial_kv_fraction
        self.min_kv_fraction = min_kv_fraction
        self.max_kv_fraction = max_kv_fraction
        self.expert_hit_rate_target = expert_hit_rate_target

        # Current allocation
        self.kv_budget_bytes = int(total_budget_bytes * initial_kv_fraction)
        self.expert_budget_bytes = total_budget_bytes - self.kv_budget_bytes

        # Tracking
        self._rebalance_count = 0
        self._history: list[dict] = []

    @property
    def kv_fraction(self) -> float:
        if self.total_budget_bytes <= 0:
            return 0.0
        return self.kv_budget_bytes / self.total_budget_bytes

    @property
    def expert_fraction(self) -> float:
        return 1.0 - self.kv_fraction

    def rebalance(self, kv_ram_used: int, expert_hit_rate: float,
                  seq_len: int, bytes_per_token: int) -> dict:
        """Rebalance budgets based on current usage patterns.

        Args:
            kv_ram_used: Current KV cache RAM usage in bytes.
            expert_hit_rate: Current expert cache hit rate (0.0 to 1.0).
            seq_len: Current sequence length.
            bytes_per_token: KV bytes consumed per token across all layers.

        Returns:
            Dict with new budgets and rebalance decision details.
        """
        self._rebalance_count += 1

        # Calculate pressure signals
        kv_pressure = kv_ram_used / max(self.kv_budget_bytes, 1)
        expert_pressure = 1.0 - (expert_hit_rate / max(self.expert_hit_rate_target, 0.01))

        # Project KV needs for next 256 tokens (one block)
        projected_kv_growth = bytes_per_token * 256

        # Decision: which direction to shift budget?
        # Positive shift = give more to KV, negative = give more to experts
        shift = 0

        if kv_pressure > 0.9 and expert_hit_rate > self.expert_hit_rate_target:
            # KV is full but expert cache is performing well — give KV more room
            shift = min(projected_kv_growth,
                        int(self.total_budget_bytes * 0.02))  # max 2% per rebalance
        elif expert_pressure > 0.1 and kv_pressure < 0.5:
            # Expert cache is underperforming and KV has slack — give experts more
            shift = -min(int(self.kv_budget_bytes * 0.1),
                         int(self.total_budget_bytes * 0.02))

        # Apply shift within bounds
        new_kv = self.kv_budget_bytes + shift
        min_kv = int(self.total_budget_bytes * self.min_kv_fraction)
        max_kv = int(self.total_budget_bytes * self.max_kv_fraction)
        new_kv = max(min_kv, min(max_kv, new_kv))

        old_kv = self.kv_budget_bytes
        self.kv_budget_bytes = new_kv
        self.expert_budget_bytes = self.total_budget_bytes - new_kv

        result = {
            "rebalance_id": self._rebalance_count,
            "old_kv_bytes": old_kv,
            "new_kv_bytes": new_kv,
            "shift_bytes": new_kv - old_kv,
            "kv_pressure": round(kv_pressure, 3),
            "expert_pressure": round(expert_pressure, 3),
            "expert_hit_rate": round(expert_hit_rate, 3),
            "expert_budget_bytes": self.expert_budget_bytes,
            "kv_fraction": round(self.kv_fraction, 3),
        }
        self._history.append(result)
        return result

    def expert_cache_slots(self, expert_bytes: int) -> int:
        """How many expert slots fit in the current expert budget."""
        if expert_bytes <= 0:
            return 0
        return self.expert_budget_bytes // expert_bytes

    def stats(self) -> dict:
        return {
            "total_budget_bytes": self.total_budget_bytes,
            "total_budget_gb": round(self.total_budget_bytes / 1024**3, 1),
            "kv_budget_bytes": self.kv_budget_bytes,
            "kv_budget_gb": round(self.kv_budget_bytes / 1024**3, 1),
            "expert_budget_bytes": self.expert_budget_bytes,
            "expert_budget_gb": round(self.expert_budget_bytes / 1024**3, 1),
            "kv_fraction": round(self.kv_fraction, 3),
            "expert_fraction": round(self.expert_fraction, 3),
            "rebalance_count": self._rebalance_count,
        }
