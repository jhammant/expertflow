"""Shared fixtures for ExpertFlow test suite."""

import numpy as np
import pytest

from ef_kv_manager import (
    KVBlock,
    PagedKVCache,
    MoEKVCacheManager,
    KVExpertBudgetCoordinator,
    EvictionPolicy,
)
from ef_integrated_engine import (
    SimulatedMoERouter,
    SimulatedExpertStore,
    ExpertWeightCache,
    ExpertFlowEngine,
    ModelConfig,
    DEEPSEEK_V3_CONFIG,
    DEEPSEEK_V3_SMALL_SIM,
    MIXTRAL_8X7B_CONFIG,
)


# A tiny model config with zero latency for fast tests
FAST_CONFIG = ModelConfig(
    name="Test-Tiny",
    n_layers=4,
    first_moe_layer=1,
    n_experts=8,
    n_active_experts=2,
    num_kv_heads=2,
    head_dim=16,
    expert_rows=16,
    expert_cols=8,
    total_ram_gb=1,
    attn_budget_gb=0,
    os_overhead_gb=0,
    cold_load_ms=0.0,
    warm_load_ms=0.0,
)


def make_kv(num_kv_heads=2, n_tokens=4, head_dim=16):
    """Create random K/V tensors of correct shape."""
    keys = np.random.randn(num_kv_heads, n_tokens, head_dim).astype(np.float16)
    values = np.random.randn(num_kv_heads, n_tokens, head_dim).astype(np.float16)
    return keys, values
