"""Tests for the benchmark runner and integrated engine.

Covers: ExpertFlowEngine generate_token/generate_turn, benchmark output
structure, model config presets, and budget rebalancing during generation.
"""

import numpy as np
import pytest

from ef_integrated_engine import (
    ExpertFlowEngine,
    ModelConfig,
    DEEPSEEK_V3_CONFIG,
    DEEPSEEK_V3_SMALL_SIM,
    MIXTRAL_8X7B_CONFIG,
)
from ef_kv_manager import EvictionPolicy

FAST_CONFIG = ModelConfig(
    name="Test-Tiny",
    n_layers=4, first_moe_layer=1, n_experts=8, n_active_experts=2,
    num_kv_heads=2, head_dim=16, expert_rows=16, expert_cols=8,
    total_ram_gb=1, attn_budget_gb=0, os_overhead_gb=0,
    cold_load_ms=0.0, warm_load_ms=0.0,
)


class TestExpertFlowEngineToken:
    """Tests for single-token generation."""

    def test_generate_token_returns_stats(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.generate_token()
        assert "token_idx" in result
        assert "expert_hit_rate" in result
        assert "kv_ram_mb" in result
        assert "kv_hit_rate" in result
        assert result["token_idx"] == 1

    def test_prefill_processes_multiple_tokens(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.generate_token(is_prefill=True, prefill_tokens=10)
        assert result["n_tokens"] == 10
        assert result["is_prefill"] is True

    def test_token_count_increments(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        for i in range(5):
            result = engine.generate_token()
            assert result["token_idx"] == i + 1
        assert engine.tokens_generated == 5

    def test_kv_cache_grows_with_tokens(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        engine.generate_token()
        ram_after_1 = engine.kv_cache.total_ram_bytes
        engine.generate_token()
        ram_after_2 = engine.kv_cache.total_ram_bytes
        assert ram_after_2 >= ram_after_1


class TestExpertFlowEngineTurn:
    """Tests for multi-token turn generation."""

    def test_generate_turn_returns_expected_keys(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.generate_turn(
            prompt_tokens=5, response_tokens=10, rebalance_every=5,
        )
        expected_keys = [
            "decode_tok_s", "avg_decode_s", "expert_cache_size",
            "expert_cache_budget", "kv_ram_mb", "kv_budget_mb",
            "kv_seq_len", "kv_evictions", "expert_loads_from_nvme",
            "budget_rebalances", "token_details",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_turn_generates_correct_token_count(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.generate_turn(
            prompt_tokens=5, response_tokens=10, rebalance_every=100,
        )
        # generate_token is called 1 (prefill) + 10 (decode) = 11 times
        # tokens_generated increments by 1 per call
        assert engine.tokens_generated == 11
        assert len(result["token_details"]) == 11

    def test_rebalance_fires(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        result = engine.generate_turn(
            prompt_tokens=5, response_tokens=20, rebalance_every=5,
        )
        # With 20 decode tokens and rebalance_every=5, should rebalance ~4 times
        assert result["budget_rebalances"] >= 3

    def test_multiple_turns_accumulate(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        engine.generate_turn(prompt_tokens=5, response_tokens=10)
        engine.generate_turn(prompt_tokens=5, response_tokens=10)
        # 2 turns × (1 prefill call + 10 decode calls) = 22
        assert engine.tokens_generated == 22


class TestModelConfigs:
    """Tests for model configuration presets."""

    def test_deepseek_v3_config(self):
        c = DEEPSEEK_V3_CONFIG
        assert c.n_layers == 61
        assert c.first_moe_layer == 3
        assert c.n_experts == 256
        assert c.n_active_experts == 8
        assert c.n_moe_layers == 58

    def test_mixtral_config(self):
        c = MIXTRAL_8X7B_CONFIG
        assert c.n_layers == 32
        assert c.first_moe_layer == 0
        assert c.n_experts == 8
        assert c.n_active_experts == 2
        assert c.n_moe_layers == 32

    def test_small_sim_config(self):
        c = DEEPSEEK_V3_SMALL_SIM
        assert c.total_ram_gb == 16
        assert c.n_layers == 8
        assert c.n_moe_layers == 7

    def test_available_budget_calculation(self):
        c = ModelConfig(total_ram_gb=128, attn_budget_gb=30, os_overhead_gb=28)
        assert c.available_budget_gb == 70

    def test_custom_config(self):
        c = ModelConfig(
            name="Custom",
            n_layers=10,
            first_moe_layer=2,
            n_experts=16,
            n_active_experts=4,
        )
        assert c.n_moe_layers == 8
        assert c.name == "Custom"


class TestBenchmarkRebalancing:
    """Tests for budget rebalancing during benchmark runs."""

    def test_rebalance_updates_budgets(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        engine.generate_turn(prompt_tokens=10, response_tokens=20)

        result = engine.rebalance_budgets()
        assert "kv_fraction" in result
        assert "expert_budget_bytes" in result

    def test_rebalance_log_tracks_history(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        engine.generate_turn(prompt_tokens=5, response_tokens=5, rebalance_every=2)
        assert len(engine._rebalance_log) > 0

    def test_engine_stats(self):
        engine = ExpertFlowEngine(config=FAST_CONFIG)
        engine.generate_turn(prompt_tokens=5, response_tokens=10)
        stats = engine.stats()
        assert "tokens_generated" in stats
        assert "expert_cache" in stats
        assert "kv_cache" in stats
        assert "cache_hit_rate" in stats["kv_cache"]
        assert stats["tokens_generated"] == 11

    def test_benchmark_reports_kv_cache_metrics(self):
        from benchmark_integrated import run_benchmark

        result = run_benchmark(
            config=FAST_CONFIG,
            n_turns=1,
            prompt_tokens=4,
            response_tokens=4,
            verbose=False,
        )

        aggregate = result["aggregate"]
        assert "final_kv_hit_rate" in aggregate
        assert "total_kv_page_ins" in aggregate
        assert "total_kv_page_outs" in aggregate


class TestEvictionPolicyComparison:
    """Tests running the engine with each eviction policy."""

    @pytest.mark.parametrize("policy", list(EvictionPolicy))
    def test_policy_runs_to_completion(self, policy):
        engine = ExpertFlowEngine(
            config=FAST_CONFIG,
            kv_eviction_policy=policy,
        )
        result = engine.generate_turn(
            prompt_tokens=5, response_tokens=10, rebalance_every=5,
        )
        assert len(result["token_details"]) > 0
        assert engine.tokens_generated == 11
