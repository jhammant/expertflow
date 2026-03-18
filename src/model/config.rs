//! MoE model configurations.

use serde::{Deserialize, Serialize};

/// MoE model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    pub name: String,
    pub num_layers: usize,
    pub num_experts: usize,
    pub num_active_experts: usize,
    pub expert_size_bytes: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl MoEConfig {
    /// DeepSeek V3 configuration (Q4 quantization)
    pub fn deepseek_v3_q4() -> Self {
        Self {
            name: "DeepSeek-V3-Q4".to_string(),
            num_layers: 61,
            num_experts: 256,
            num_active_experts: 8,
            expert_size_bytes: 1_400_000_000, // ~1.4GB per expert at Q4
            hidden_size: 7168,
            intermediate_size: 18432,
        }
    }

    /// DeepSeek V3 configuration (1-bit quantization)
    pub fn deepseek_v3_1bit() -> Self {
        Self {
            name: "DeepSeek-V3-1bit".to_string(),
            num_layers: 61,
            num_experts: 256,
            num_active_experts: 8,
            expert_size_bytes: 350_000_000, // ~350MB per expert at 1-bit
            hidden_size: 7168,
            intermediate_size: 18432,
        }
    }

    /// DeepSeek V3 configuration (Q2 quantization)
    pub fn deepseek_v3_q2() -> Self {
        Self {
            name: "DeepSeek-V3-Q2".to_string(),
            num_layers: 61,
            num_experts: 256,
            num_active_experts: 8,
            expert_size_bytes: 700_000_000, // ~700MB per expert at Q2
            hidden_size: 7168,
            intermediate_size: 18432,
        }
    }

    /// Qwen3 235B configuration
    pub fn qwen3_235b() -> Self {
        Self {
            name: "Qwen3-235B".to_string(),
            num_layers: 80,
            num_experts: 160,
            num_active_experts: 8,
            expert_size_bytes: 1_375_000_000, // ~1.375GB per expert
            hidden_size: 8192,
            intermediate_size: 29568,
        }
    }

    /// Get total model size in bytes
    pub fn total_size(&self) -> usize {
        self.num_experts * self.expert_size_bytes
    }

    /// Get active size per layer (memory needed for active experts)
    pub fn active_size_per_layer(&self) -> usize {
        self.num_active_experts * self.expert_size_bytes
    }

    /// Get total active size across all layers
    pub fn total_active_size(&self) -> usize {
        self.num_layers * self.active_size_per_layer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_v3_config() {
        let config = MoEConfig::deepseek_v3_q4();
        assert_eq!(config.num_layers, 61);
        assert_eq!(config.num_experts, 256);
        assert_eq!(config.num_active_experts, 8);

        let total_size_gb = config.total_size() / (1024 * 1024 * 1024);
        assert!(total_size_gb > 300); // ~350GB
    }

    #[test]
    fn test_qwen3_config() {
        let config = MoEConfig::qwen3_235b();
        assert_eq!(config.num_layers, 80);
        assert_eq!(config.num_experts, 160);
    }

    #[test]
    fn test_active_size() {
        let config = MoEConfig::deepseek_v3_1bit();
        let active_gb = config.active_size_per_layer() / (1024 * 1024 * 1024);
        assert!(active_gb >= 2); // 8 experts × 350MB ≈ 2.8GB
    }
}
