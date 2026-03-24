//! MoE router lookahead engine.
//!
//! Predicts which experts will be needed in upcoming layers to enable
//! prefetching before they're actually required.

use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

use crate::{ExpertId, LayerIdx};

/// Router lookahead predictor
pub struct RouterLookahead {
    num_layers: usize,
    num_experts: usize,
    num_active_experts: usize,
}

impl RouterLookahead {
    /// Create a new router lookahead
    pub fn new(num_layers: usize, num_experts: usize, num_active_experts: usize) -> Self {
        Self {
            num_layers,
            num_experts,
            num_active_experts,
        }
    }

    /// Predict which experts will be activated for a given layer.
    ///
    /// This is a mock implementation that returns random experts based on
    /// a distribution. In a real implementation, this would:
    /// 1. Run the router forward pass on the current hidden state
    /// 2. Use top-k selection to get the most likely experts
    /// 3. Optionally look ahead 1-2 tokens to predict future activations
    pub fn lookahead(&self, _layer: LayerIdx, _hidden_state: Option<&[f32]>) -> Vec<ExpertId> {
        let mut rng = rand::thread_rng();

        // Mock implementation: weighted random selection
        // In reality, this would run the actual router network

        // Create a distribution that favors lower expert IDs
        // (simulating that some experts are more commonly used)
        let weights: Vec<f64> = (0..self.num_experts)
            .map(|i| {
                let x = i as f64 / self.num_experts as f64;
                // Exponential decay: more weight to early experts
                (-2.0 * x).exp()
            })
            .collect();

        let dist = WeightedIndex::new(&weights).unwrap();

        let mut selected = Vec::new();
        for _ in 0..self.num_active_experts {
            let expert_id = dist.sample(&mut rng);
            if !selected.contains(&expert_id) {
                selected.push(expert_id);
            }
        }

        // Ensure we have exactly num_active_experts
        while selected.len() < self.num_active_experts {
            let expert_id = rng.gen_range(0..self.num_experts);
            if !selected.contains(&expert_id) {
                selected.push(expert_id);
            }
        }

        selected
    }

    /// Predict experts for multiple upcoming layers
    pub fn lookahead_batch(
        &self,
        start_layer: LayerIdx,
        num_layers: usize,
        _hidden_state: Option<&[f32]>,
    ) -> Vec<Vec<ExpertId>> {
        (start_layer..start_layer + num_layers)
            .map(|layer| self.lookahead(layer, None))
            .collect()
    }

    /// Get the number of experts
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the number of active experts per layer
    pub fn num_active_experts(&self) -> usize {
        self.num_active_experts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_lookahead() {
        let router = RouterLookahead::new(60, 256, 8);

        let experts = router.lookahead(0, None);
        assert_eq!(experts.len(), 8);

        // Check all experts are unique
        let mut sorted = experts.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 8);

        // Check all experts are in valid range
        for &expert_id in &experts {
            assert!(expert_id < 256);
        }
    }

    #[test]
    fn test_router_batch() {
        let router = RouterLookahead::new(60, 256, 8);

        let batch = router.lookahead_batch(0, 5, None);
        assert_eq!(batch.len(), 5);

        for experts in batch {
            assert_eq!(experts.len(), 8);
        }
    }
}
