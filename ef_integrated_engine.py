#!/usr/bin/env python3
"""
ExpertFlow Integrated Inference Engine
=======================================

End-to-end MoE inference pipeline combining:
  - Expert weight cache (GGUFExpertCache) with NVMe mmap loading
  - Paged KV cache (MoEKVCacheManager) with block-level eviction
  - Dynamic budget coordinator (KVExpertBudgetCoordinator)
  - Async prefetch pipeline for expert weight preloading
  - Simulated MoE routing with Zipf-distributed expert popularity

This module provides:
  1. ExpertFlowEngine: the integrated engine that wires everything together
  2. SimulatedMoERouter: generates realistic expert routing decisions
  3. SimulatedAttention: produces KV tensors sized to match real models

No real model weights are needed — the engine simulates the compute
pipeline at the correct memory scale to benchmark the caching system.

Target architecture (DeepSeek V3 on M5 Max 128GB):
  - 61 layers, 58 MoE layers, 256 experts/layer, 8 active/token
  - Expert weights: ~24MB each (gate+up+down, Q2_K)
  - KV per token per layer: num_kv_heads × head_dim × 2 × 2 bytes
  - Memory: 30GB attention + 60GB expert cache + 10GB KV = 100GB
"""

import time
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field

from ef_kv_manager import (
    MoEKVCacheManager,
    KVExpertBudgetCoordinator,
    EvictionPolicy,
)


# ═══════════════════════════════════════════════════════════════════════
# Simulated MoE Router
# ═══════════════════════════════════════════════════════════════════════

class SimulatedMoERouter:
    """Generates realistic expert routing decisions for benchmarking.

    Models three properties of real MoE routing:
      1. Zipf popularity: some experts are much more popular than others
      2. Layer locality: different layers prefer different expert subsets
      3. Temporal locality: consecutive tokens tend to reuse recent experts

    Args:
        n_experts: Total experts per layer (e.g. 256 for DeepSeek V3).
        n_active: Experts activated per token (e.g. 8 for DeepSeek V3).
        n_moe_layers: Number of MoE layers.
        zipf_skew: Zipf exponent (higher = more skewed popularity).
        temporal_locality: Probability of reusing an expert from the last token.
    """

    def __init__(self, n_experts: int = 256, n_active: int = 8,
                 n_moe_layers: int = 58, zipf_skew: float = 1.1,
                 temporal_locality: float = 0.4, seed: int = 42):
        self.n_experts = n_experts
        self.n_active = n_active
        self.n_moe_layers = n_moe_layers
        self.temporal_locality = temporal_locality
        self.rng = np.random.RandomState(seed)

        # Zipf distribution for expert popularity (per layer)
        ranks = np.arange(1, n_experts + 1, dtype=np.float64)
        base_probs = 1.0 / np.power(ranks, zipf_skew)

        # Each layer has a shuffled version of the popularity distribution
        self._layer_probs = []
        for _ in range(n_moe_layers):
            perm = self.rng.permutation(n_experts)
            layer_p = base_probs[perm]
            layer_p /= layer_p.sum()
            self._layer_probs.append(layer_p)

        # Track last token's routing for temporal locality
        self._last_routing: dict[int, list[int]] = {}  # layer -> expert list

    def route(self, layer_idx: int) -> tuple[list[int], np.ndarray]:
        """Select active experts for one token at one MoE layer.

        Returns:
            (expert_indices, routing_weights) — selected experts and their scores.
        """
        probs = self._layer_probs[layer_idx % len(self._layer_probs)]

        # Temporal locality: boost probability of recently-used experts
        if layer_idx in self._last_routing and self.rng.random() < self.temporal_locality:
            boosted = probs.copy()
            for eidx in self._last_routing[layer_idx]:
                boosted[eidx] *= 3.0
            boosted /= boosted.sum()
            selected = self.rng.choice(
                self.n_experts, size=self.n_active, replace=False, p=boosted
            )
        else:
            selected = self.rng.choice(
                self.n_experts, size=self.n_active, replace=False, p=probs
            )

        # Generate softmax-like routing weights
        raw_scores = self.rng.exponential(1.0, size=self.n_active).astype(np.float32)
        weights = raw_scores / raw_scores.sum()

        self._last_routing[layer_idx] = selected.tolist()
        return selected.tolist(), weights

    def route_all_layers(self) -> list[tuple[list[int], np.ndarray]]:
        """Route one token through all MoE layers.

        Returns list of (expert_indices, weights) per layer.
        """
        return [self.route(i) for i in range(self.n_moe_layers)]


# ═══════════════════════════════════════════════════════════════════════
# Simulated Expert Weight Store
# ═══════════════════════════════════════════════════════════════════════

class SimulatedExpertStore:
    """Simulates loading expert weights from NVMe with realistic latencies.

    Does not hold actual weight tensors — generates random tensors on demand
    to simulate the mmap page-in cost without needing 200GB of GGUF files.

    Args:
        n_layers: Total transformer layers.
        first_moe_layer: First layer that is MoE.
        n_experts: Experts per MoE layer.
        expert_rows: Rows per expert projection (simulates hidden_dim).
        expert_cols: Cols per expert projection (simulates intermediate_dim).
        cold_load_ms: Simulated cold load latency per expert (ms).
        warm_load_ms: Simulated warm (page cache hit) latency per expert (ms).
    """

    def __init__(self, n_layers: int = 61, first_moe_layer: int = 3,
                 n_experts: int = 256, expert_rows: int = 256,
                 expert_cols: int = 128, cold_load_ms: float = 1.5,
                 warm_load_ms: float = 0.3):
        self.n_layers = n_layers
        self.first_moe_layer = first_moe_layer
        self.n_experts = n_experts
        self.expert_rows = expert_rows
        self.expert_cols = expert_cols
        self.cold_load_ms = cold_load_ms
        self.warm_load_ms = warm_load_ms

        # Track which experts have been loaded before (for warm vs cold)
        self._loaded_before: set[tuple[int, int]] = set()
        self.total_loads = 0
        self.total_cold = 0
        self.total_warm = 0
        self.total_bytes_loaded = 0

    @property
    def expert_bytes(self) -> int:
        """Bytes per expert (gate + up + down projections, float16)."""
        return self.expert_rows * self.expert_cols * 2 * 3  # 3 projections

    def load_expert(self, layer_idx: int, expert_idx: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate loading an expert from NVMe.

        Returns (gate, up, down) float16 arrays.
        Applies simulated latency based on cold/warm state.
        """
        key = (layer_idx, expert_idx)
        self.total_loads += 1

        if key in self._loaded_before:
            latency = self.warm_load_ms / 1000.0
            self.total_warm += 1
        else:
            latency = self.cold_load_ms / 1000.0
            self._loaded_before.add(key)
            self.total_cold += 1

        # Simulate NVMe read latency
        time.sleep(latency)

        # Generate dummy weights (small for simulation)
        shape = (self.expert_rows, self.expert_cols)
        gate = np.random.randn(*shape).astype(np.float16)
        up = np.random.randn(*shape).astype(np.float16)
        down = np.random.randn(*shape).astype(np.float16)

        self.total_bytes_loaded += self.expert_bytes
        return gate, up, down

    def stats(self) -> dict:
        return {
            "total_loads": self.total_loads,
            "cold_loads": self.total_cold,
            "warm_loads": self.total_warm,
            "total_bytes_loaded": self.total_bytes_loaded,
            "total_mb_loaded": round(self.total_bytes_loaded / 1024**2, 1),
            "expert_bytes": self.expert_bytes,
        }


# ═══════════════════════════════════════════════════════════════════════
# Expert Weight Cache (standalone, no GGUF dependency)
# ═══════════════════════════════════════════════════════════════════════

class ExpertWeightCache:
    """In-memory cache for dequantized expert weights.

    Stores (gate, up, down) tuples keyed by (layer_idx, expert_idx).
    Uses frequency-weighted scoring for eviction: experts accessed more
    often and more recently score higher and survive eviction longer.

    Compatible API with GGUFExpertCache from ef_deepseek_mmap.py.
    """

    def __init__(self, budget: int = 2000, decay: float = 0.95):
        self.cache: OrderedDict = OrderedDict()
        self.scores: dict = {}
        self.budget = budget
        self.decay = decay

        self.token_hits = 0
        self.token_misses = 0
        self.total_hits = 0
        self.total_misses = 0
        self.current_token = 0

    def get(self, key: tuple) -> tuple | None:
        if key in self.cache:
            self.token_hits += 1
            self.total_hits += 1
            self.scores[key] = self.scores.get(key, 0) + 1.0
            return self.cache[key]
        self.token_misses += 1
        self.total_misses += 1
        return None

    def put(self, key: tuple, value: tuple):
        self.cache[key] = value
        self.scores[key] = self.scores.get(key, 0) + 1.0

    def trim(self) -> int:
        """Evict lowest-scored entries down to budget, decay all scores."""
        evicted = 0
        while len(self.cache) > self.budget:
            min_key = min(self.cache.keys(), key=lambda k: self.scores.get(k, 0))
            del self.cache[min_key]
            self.scores.pop(min_key, None)
            evicted += 1
        for k in self.scores:
            self.scores[k] *= self.decay
        return evicted

    def end_token(self):
        self.current_token += 1

    def reset_token_stats(self):
        self.token_hits = 0
        self.token_misses = 0

    @property
    def token_hit_rate(self) -> float:
        t = self.token_hits + self.token_misses
        return self.token_hits / t * 100 if t > 0 else 0

    @property
    def total_hit_rate(self) -> float:
        t = self.total_hits + self.total_misses
        return self.total_hits / t * 100 if t > 0 else 0

    @property
    def size(self) -> int:
        return len(self.cache)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "budget": self.budget,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_hit_rate": round(self.total_hit_rate, 1),
            "token_hits": self.token_hits,
            "token_misses": self.token_misses,
            "token_hit_rate": round(self.token_hit_rate, 1),
        }


# ═══════════════════════════════════════════════════════════════════════
# Integrated Engine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuration for a simulated MoE model."""
    name: str = "DeepSeek-V3-671B"
    n_layers: int = 61
    first_moe_layer: int = 3
    n_experts: int = 256
    n_active_experts: int = 8
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 7168
    intermediate_dim: int = 2048
    expert_rows: int = 256
    expert_cols: int = 128

    # Memory layout (GB)
    total_ram_gb: int = 128
    attn_budget_gb: int = 30
    os_overhead_gb: int = 28

    # Simulated NVMe performance
    cold_load_ms: float = 1.5
    warm_load_ms: float = 0.3

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.first_moe_layer

    @property
    def available_budget_gb(self) -> int:
        """RAM available for KV cache + expert cache."""
        return self.total_ram_gb - self.attn_budget_gb - self.os_overhead_gb


# Preset configs for common models
DEEPSEEK_V3_CONFIG = ModelConfig(
    name="DeepSeek-V3-671B-Q2K",
    n_layers=61, first_moe_layer=3, n_experts=256, n_active_experts=8,
    num_kv_heads=8, head_dim=128,
    expert_rows=256, expert_cols=128,
)

MIXTRAL_8X7B_CONFIG = ModelConfig(
    name="Mixtral-8x7B-Q4K",
    n_layers=32, first_moe_layer=0, n_experts=8, n_active_experts=2,
    num_kv_heads=8, head_dim=128,
    expert_rows=128, expert_cols=64,
    cold_load_ms=0.5, warm_load_ms=0.1,
)

DEEPSEEK_V3_SMALL_SIM = ModelConfig(
    name="DeepSeek-V3-Sim-Small",
    n_layers=8, first_moe_layer=1, n_experts=32, n_active_experts=4,
    num_kv_heads=4, head_dim=32,
    expert_rows=64, expert_cols=32,
    total_ram_gb=16, attn_budget_gb=2, os_overhead_gb=2,
    cold_load_ms=0.1, warm_load_ms=0.01,
)


class ExpertFlowEngine:
    """End-to-end MoE inference engine with integrated caching.

    Wires together:
      - SimulatedExpertStore: provides expert weights (simulated NVMe)
      - ExpertWeightCache: caches hot expert weights in RAM
      - MoEKVCacheManager: paged KV cache with eviction
      - KVExpertBudgetCoordinator: dynamic memory rebalancing
      - SimulatedMoERouter: realistic expert routing

    Simulates the full token generation loop:
      1. For each token, iterate through layers
      2. Dense layers: just attention (KV cache append)
      3. MoE layers: route → load experts (cache or NVMe) → compute
      4. After each token: trim expert cache, rebalance budgets
    """

    def __init__(self, config: ModelConfig = DEEPSEEK_V3_CONFIG,
                 kv_eviction_policy: EvictionPolicy = EvictionPolicy.FREQUENCY_WEIGHTED,
                 initial_kv_fraction: float = 0.15,
                 zipf_skew: float = 1.1,
                 temporal_locality: float = 0.4,
                 seed: int = 42):
        self.config = config

        # Budget coordinator
        total_budget = config.available_budget_gb * 1024**3
        self.coordinator = KVExpertBudgetCoordinator(
            total_budget_bytes=total_budget,
            initial_kv_fraction=initial_kv_fraction,
            min_kv_fraction=0.05,
            max_kv_fraction=0.40,
            expert_hit_rate_target=0.85,
        )

        # KV cache manager
        self.kv_cache = MoEKVCacheManager(
            n_layers=config.n_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            block_size=256,
            kv_budget_bytes=self.coordinator.kv_budget_bytes,
            eviction_policy=kv_eviction_policy,
        )

        # Expert weight store (simulated NVMe)
        self.expert_store = SimulatedExpertStore(
            n_layers=config.n_layers,
            first_moe_layer=config.first_moe_layer,
            n_experts=config.n_experts,
            expert_rows=config.expert_rows,
            expert_cols=config.expert_cols,
            cold_load_ms=config.cold_load_ms,
            warm_load_ms=config.warm_load_ms,
        )

        # Expert weight cache
        expert_slots = self.coordinator.expert_cache_slots(self.expert_store.expert_bytes)
        self.expert_cache = ExpertWeightCache(budget=max(expert_slots, 10))

        # MoE router
        self.router = SimulatedMoERouter(
            n_experts=config.n_experts,
            n_active=config.n_active_experts,
            n_moe_layers=config.n_moe_layers,
            zipf_skew=zipf_skew,
            temporal_locality=temporal_locality,
            seed=seed,
        )

        # Token generation stats
        self.tokens_generated = 0
        self.total_compute_time = 0.0
        self.total_load_time = 0.0
        self.total_kv_time = 0.0
        self.token_times: list[float] = []
        self._rebalance_log: list[dict] = []

    def _simulate_attention(self, layer_idx: int, n_tokens: int = 1):
        """Simulate attention: generate KV and append to cache.

        In real inference, this would be the attention computation.
        Here we just create random KV tensors of the correct shape.
        """
        t0 = time.monotonic()
        keys = np.random.randn(
            self.config.num_kv_heads, n_tokens, self.config.head_dim
        ).astype(np.float16)
        values = np.random.randn(
            self.config.num_kv_heads, n_tokens, self.config.head_dim
        ).astype(np.float16)
        self.kv_cache.append_kv(layer_idx, keys, values)
        self.total_kv_time += time.monotonic() - t0

    def _simulate_moe(self, moe_layer_idx: int
                      ) -> tuple[list[int], float, float]:
        """Simulate MoE layer: route, load experts, compute.

        Returns: (expert_indices, load_time, compute_time)
        """
        expert_indices, weights = self.router.route(moe_layer_idx)
        load_time = 0.0
        compute_time = 0.0

        for eidx in expert_indices:
            layer_idx = moe_layer_idx + self.config.first_moe_layer
            key = (layer_idx, eidx)

            # Try cache first
            cached = self.expert_cache.get(key)
            if cached is None:
                # Cache miss — load from NVMe (simulated)
                t0 = time.monotonic()
                gate, up, down = self.expert_store.load_expert(layer_idx, eidx)
                load_time += time.monotonic() - t0
                self.expert_cache.put(key, (gate, up, down))
            else:
                gate, up, down = cached

            # Simulate compute (just a tiny matmul to represent the work)
            t0 = time.monotonic()
            x = np.random.randn(1, self.config.expert_cols).astype(np.float16)
            _ = x @ gate.T  # simulate gate projection
            compute_time += time.monotonic() - t0

        return expert_indices, load_time, compute_time

    def generate_token(self, is_prefill: bool = False,
                       prefill_tokens: int = 1) -> dict:
        """Generate a single token through the full pipeline.

        Args:
            is_prefill: If True, process prefill_tokens at once.
            prefill_tokens: Number of tokens in prefill batch.

        Returns:
            Per-token stats dict.
        """
        t_start = time.monotonic()
        n_tokens = prefill_tokens if is_prefill else 1
        total_load = 0.0
        total_compute = 0.0
        token_routing = []

        self.expert_cache.reset_token_stats()

        for layer_idx in range(self.config.n_layers):
            # Attention (all layers)
            self._simulate_attention(layer_idx, n_tokens)

            # MoE (only MoE layers)
            if layer_idx >= self.config.first_moe_layer:
                moe_idx = layer_idx - self.config.first_moe_layer
                experts, lt, ct = self._simulate_moe(moe_idx)
                total_load += lt
                total_compute += ct
                token_routing.append((layer_idx, experts))

        # Trim expert cache
        evicted = self.expert_cache.trim()
        self.expert_cache.end_token()

        # Track timing
        token_time = time.monotonic() - t_start
        self.total_compute_time += total_compute
        self.total_load_time += total_load
        self.tokens_generated += 1
        self.token_times.append(token_time)

        return {
            "token_idx": self.tokens_generated,
            "is_prefill": is_prefill,
            "n_tokens": n_tokens,
            "time_s": round(token_time, 4),
            "load_time_s": round(total_load, 4),
            "compute_time_s": round(total_compute, 4),
            "kv_time_s": round(self.total_kv_time, 4),
            "expert_hit_rate": round(self.expert_cache.token_hit_rate, 1),
            "expert_cache_size": self.expert_cache.size,
            "experts_evicted": evicted,
            "kv_ram_mb": round(self.kv_cache.total_ram_bytes / 1024**2, 1),
            "kv_blocks": self.kv_cache.total_blocks,
            "kv_seq_len": self.kv_cache.total_seq_len,
        }

    def rebalance_budgets(self) -> dict:
        """Rebalance memory between KV cache and expert cache.

        Call this periodically (e.g. every N tokens) to adapt to
        changing workload patterns.
        """
        result = self.coordinator.rebalance(
            kv_ram_used=self.kv_cache.total_ram_bytes,
            expert_hit_rate=self.expert_cache.total_hit_rate / 100.0,
            seq_len=self.kv_cache.total_seq_len,
            bytes_per_token=self.kv_cache.bytes_per_token_all_layers(),
        )

        # Update KV cache budget
        self.kv_cache.kv_budget_bytes = self.coordinator.kv_budget_bytes

        # Update expert cache budget
        new_expert_slots = self.coordinator.expert_cache_slots(
            self.expert_store.expert_bytes
        )
        self.expert_cache.budget = max(new_expert_slots, 10)

        self._rebalance_log.append(result)
        return result

    def generate_turn(self, prompt_tokens: int = 50,
                      response_tokens: int = 100,
                      rebalance_every: int = 32) -> dict:
        """Generate a full conversation turn: prefill + decode.

        Args:
            prompt_tokens: Number of input tokens to prefill.
            response_tokens: Number of tokens to generate.
            rebalance_every: Rebalance budgets every N decode tokens.

        Returns:
            Turn-level stats dict.
        """
        t_start = time.monotonic()
        token_stats = []

        # Prefill phase
        prefill_stat = self.generate_token(
            is_prefill=True, prefill_tokens=prompt_tokens
        )
        token_stats.append(prefill_stat)

        # Decode phase
        for i in range(response_tokens):
            stat = self.generate_token(is_prefill=False)
            token_stats.append(stat)

            if (i + 1) % rebalance_every == 0:
                self.rebalance_budgets()

        turn_time = time.monotonic() - t_start

        # Aggregate turn stats
        decode_stats = [s for s in token_stats if not s["is_prefill"]]
        decode_times = [s["time_s"] for s in decode_stats]
        avg_decode = np.mean(decode_times) if decode_times else 0
        decode_tok_s = 1.0 / avg_decode if avg_decode > 0 else 0

        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_time_s": round(turn_time, 3),
            "prefill_time_s": prefill_stat["time_s"],
            "avg_decode_s": round(float(avg_decode), 4),
            "decode_tok_s": round(float(decode_tok_s), 1),
            "expert_cache_hit_rate": round(self.expert_cache.total_hit_rate, 1),
            "expert_cache_size": self.expert_cache.size,
            "expert_cache_budget": self.expert_cache.budget,
            "kv_ram_mb": round(self.kv_cache.total_ram_bytes / 1024**2, 1),
            "kv_budget_mb": round(self.kv_cache.kv_budget_bytes / 1024**2, 1),
            "kv_seq_len": self.kv_cache.total_seq_len,
            "kv_evictions": self.kv_cache.total_evictions,
            "expert_loads_from_nvme": self.expert_store.total_loads,
            "budget_rebalances": len(self._rebalance_log),
            "token_details": token_stats,
        }

    def stats(self) -> dict:
        """Full engine statistics."""
        return {
            "config": {
                "model": self.config.name,
                "n_layers": self.config.n_layers,
                "n_moe_layers": self.config.n_moe_layers,
                "n_experts": self.config.n_experts,
                "n_active_experts": self.config.n_active_experts,
                "total_ram_gb": self.config.total_ram_gb,
            },
            "tokens_generated": self.tokens_generated,
            "total_time_s": round(sum(self.token_times), 3),
            "avg_token_time_s": round(
                float(np.mean(self.token_times)) if self.token_times else 0, 4
            ),
            "kv_cache": self.kv_cache.stats(),
            "expert_cache": self.expert_cache.stats(),
            "expert_store": self.expert_store.stats(),
            "budget_coordinator": self.coordinator.stats(),
            "rebalance_count": len(self._rebalance_log),
        }
