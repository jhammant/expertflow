//! Hybrid architecture support for MoE+SSM models.
//!
//! Support for models with mixed layer types:
//! - Nemotron-H: MoE + Dense layers
//! - Jamba: MoE + Mamba (SSM) layers
//! - GatedDeltaNet: MoE + SSM + Dense
//!
//! Only MoE layers need expert scheduling; Dense/SSM layers are always resident.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use crate::LayerIdx;

/// Layer type in a hybrid architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerKind {
    /// Mixture-of-Experts layer (requires expert scheduling)
    MoE,
    /// Dense transformer layer (always resident)
    Dense,
    /// State Space Model layer (e.g., Mamba)
    SSM,
    /// Mamba layer (specific SSM variant)
    Mamba,
}

impl LayerKind {
    /// Check if this layer type requires expert scheduling
    pub fn requires_expert_scheduling(&self) -> bool {
        matches!(self, LayerKind::MoE)
    }

    /// Check if this layer is always resident in memory
    pub fn is_always_resident(&self) -> bool {
        !self.requires_expert_scheduling()
    }
}

/// Configuration for a hybrid MoE+SSM model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    pub model_name: String,
    pub num_layers: usize,
    pub layer_kinds: Vec<LayerKind>,
    pub num_experts_per_moe_layer: usize,
    pub num_active_experts: usize,
    pub expert_size_bytes: usize,
}

impl HybridConfig {
    /// Create a new hybrid config with specified layer pattern
    ///
    /// # Parameters
    /// - `model_name`: Name of the model
    /// - `layer_kinds`: Vector mapping layer index to layer kind
    /// - `num_experts_per_moe_layer`: Number of experts in each MoE layer
    /// - `num_active_experts`: Number of experts activated per token
    /// - `expert_size_bytes`: Size of each expert in bytes
    pub fn new(
        model_name: String,
        layer_kinds: Vec<LayerKind>,
        num_experts_per_moe_layer: usize,
        num_active_experts: usize,
        expert_size_bytes: usize,
    ) -> Self {
        let num_layers = layer_kinds.len();
        info!(
            "Creating hybrid config for {} with {} layers ({} MoE, {} Dense, {} SSM)",
            model_name,
            num_layers,
            layer_kinds.iter().filter(|k| **k == LayerKind::MoE).count(),
            layer_kinds.iter().filter(|k| **k == LayerKind::Dense).count(),
            layer_kinds.iter().filter(|k| matches!(k, LayerKind::SSM | LayerKind::Mamba)).count(),
        );

        Self {
            model_name,
            num_layers,
            layer_kinds,
            num_experts_per_moe_layer,
            num_active_experts,
            expert_size_bytes,
        }
    }

    /// Get the layer kind at a specific index
    pub fn layer_kind(&self, layer_idx: LayerIdx) -> Option<LayerKind> {
        self.layer_kinds.get(layer_idx).copied()
    }

    /// Check if a layer is a MoE layer
    pub fn is_moe_layer(&self, layer_idx: LayerIdx) -> bool {
        self.layer_kind(layer_idx) == Some(LayerKind::MoE)
    }

    /// Get indices of all MoE layers
    pub fn moe_layers(&self) -> Vec<LayerIdx> {
        self.layer_kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == LayerKind::MoE)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get indices of all Dense layers
    pub fn dense_layers(&self) -> Vec<LayerIdx> {
        self.layer_kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == LayerKind::Dense)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get indices of all SSM layers
    pub fn ssm_layers(&self) -> Vec<LayerIdx> {
        self.layer_kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| matches!(kind, LayerKind::SSM | LayerKind::Mamba))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get layer kind statistics
    pub fn layer_stats(&self) -> LayerStats {
        let mut counts: HashMap<LayerKind, usize> = HashMap::new();
        for kind in &self.layer_kinds {
            *counts.entry(*kind).or_insert(0) += 1;
        }

        LayerStats {
            total_layers: self.num_layers,
            moe_count: *counts.get(&LayerKind::MoE).unwrap_or(&0),
            dense_count: *counts.get(&LayerKind::Dense).unwrap_or(&0),
            ssm_count: *counts.get(&LayerKind::SSM).unwrap_or(&0)
                + *counts.get(&LayerKind::Mamba).unwrap_or(&0),
        }
    }

    /// Get total memory required for all experts (MoE layers only)
    pub fn total_expert_memory(&self) -> usize {
        let num_moe_layers = self.moe_layers().len();
        num_moe_layers * self.num_experts_per_moe_layer * self.expert_size_bytes
    }

    /// Nemotron-H configuration (alternating MoE and Dense)
    ///
    /// Nemotron-H has 32 layers, alternating between MoE and Dense
    pub fn nemotron_h() -> Self {
        let layer_kinds = (0..32)
            .map(|i| if i % 2 == 0 { LayerKind::MoE } else { LayerKind::Dense })
            .collect();

        Self::new(
            "Nemotron-H".to_string(),
            layer_kinds,
            64,  // 64 experts per MoE layer
            8,   // 8 active experts
            400_000_000, // ~400MB per expert
        )
    }

    /// Jamba configuration (mixed MoE and Mamba)
    ///
    /// Jamba-1.5-Large: 64 layers with pattern [MoE, Mamba, Mamba, ...]
    pub fn jamba_large() -> Self {
        let layer_kinds = (0..64)
            .map(|i| {
                // Every 8th layer is MoE, rest are Mamba
                if i % 8 == 0 {
                    LayerKind::MoE
                } else {
                    LayerKind::Mamba
                }
            })
            .collect();

        Self::new(
            "Jamba-1.5-Large".to_string(),
            layer_kinds,
            16,  // 16 experts per MoE layer
            2,   // 2 active experts (Jamba uses top-2)
            800_000_000, // ~800MB per expert
        )
    }

    /// GatedDeltaNet configuration (MoE + SSM + Dense)
    pub fn gated_deltanet() -> Self {
        let layer_kinds = (0..48)
            .map(|i| match i % 3 {
                0 => LayerKind::MoE,
                1 => LayerKind::SSM,
                _ => LayerKind::Dense,
            })
            .collect();

        Self::new(
            "GatedDeltaNet".to_string(),
            layer_kinds,
            32,  // 32 experts per MoE layer
            4,   // 4 active experts
            500_000_000, // ~500MB per expert
        )
    }

    /// Create a pure MoE config (all layers are MoE)
    pub fn pure_moe(
        model_name: String,
        num_layers: usize,
        num_experts: usize,
        num_active: usize,
        expert_size: usize,
    ) -> Self {
        let layer_kinds = vec![LayerKind::MoE; num_layers];

        Self::new(
            model_name,
            layer_kinds,
            num_experts,
            num_active,
            expert_size,
        )
    }
}

/// Statistics about layer kinds in a hybrid model
#[derive(Debug, Clone)]
pub struct LayerStats {
    pub total_layers: usize,
    pub moe_count: usize,
    pub dense_count: usize,
    pub ssm_count: usize,
}

impl LayerStats {
    pub fn format(&self) -> String {
        format!(
            "{} layers total: {} MoE ({:.1}%), {} Dense ({:.1}%), {} SSM ({:.1}%)",
            self.total_layers,
            self.moe_count,
            (self.moe_count as f64 / self.total_layers as f64) * 100.0,
            self.dense_count,
            (self.dense_count as f64 / self.total_layers as f64) * 100.0,
            self.ssm_count,
            (self.ssm_count as f64 / self.total_layers as f64) * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kind_properties() {
        assert!(LayerKind::MoE.requires_expert_scheduling());
        assert!(!LayerKind::Dense.requires_expert_scheduling());
        assert!(!LayerKind::SSM.requires_expert_scheduling());
        assert!(!LayerKind::Mamba.requires_expert_scheduling());

        assert!(!LayerKind::MoE.is_always_resident());
        assert!(LayerKind::Dense.is_always_resident());
    }

    #[test]
    fn test_hybrid_config_creation() {
        let layer_kinds = vec![
            LayerKind::MoE,
            LayerKind::Dense,
            LayerKind::MoE,
            LayerKind::Mamba,
        ];

        let config = HybridConfig::new(
            "TestModel".to_string(),
            layer_kinds,
            16,
            2,
            1_000_000,
        );

        assert_eq!(config.num_layers, 4);
        assert_eq!(config.moe_layers(), vec![0, 2]);
        assert_eq!(config.dense_layers(), vec![1]);
        assert_eq!(config.ssm_layers(), vec![3]);
    }

    #[test]
    fn test_nemotron_h_config() {
        let config = HybridConfig::nemotron_h();

        assert_eq!(config.num_layers, 32);
        assert_eq!(config.moe_layers().len(), 16); // Half are MoE
        assert_eq!(config.dense_layers().len(), 16); // Half are Dense

        // Check alternating pattern
        assert!(config.is_moe_layer(0));
        assert!(!config.is_moe_layer(1));
        assert!(config.is_moe_layer(2));
    }

    #[test]
    fn test_jamba_config() {
        let config = HybridConfig::jamba_large();

        assert_eq!(config.num_layers, 64);
        assert_eq!(config.num_active_experts, 2); // Jamba uses top-2

        // Every 8th layer is MoE
        let moe_layers = config.moe_layers();
        assert_eq!(moe_layers.len(), 8); // 64 / 8 = 8 MoE layers
        assert_eq!(moe_layers[0], 0);
        assert_eq!(moe_layers[1], 8);
    }

    #[test]
    fn test_layer_stats() {
        let config = HybridConfig::nemotron_h();
        let stats = config.layer_stats();

        assert_eq!(stats.total_layers, 32);
        assert_eq!(stats.moe_count, 16);
        assert_eq!(stats.dense_count, 16);
        assert_eq!(stats.ssm_count, 0);
    }

    #[test]
    fn test_total_expert_memory() {
        let config = HybridConfig::jamba_large();

        // 8 MoE layers × 16 experts × 800MB = 102.4 GB
        let total = config.total_expert_memory();
        let expected = 8 * 16 * 800_000_000;
        assert_eq!(total, expected);
    }

    #[test]
    fn test_pure_moe_config() {
        let config = HybridConfig::pure_moe(
            "DeepSeek-V3".to_string(),
            61,
            256,
            8,
            1_400_000_000,
        );

        assert_eq!(config.num_layers, 61);
        assert_eq!(config.moe_layers().len(), 61); // All layers are MoE
        assert_eq!(config.dense_layers().len(), 0);
        assert_eq!(config.ssm_layers().len(), 0);
    }
}
