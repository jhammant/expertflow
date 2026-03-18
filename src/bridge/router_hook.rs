//! Router prediction hook - taps into MoE router output for lookahead prefetching.
//!
//! The MoE router is a small network that predicts which experts to use for each token.
//! By intercepting the router's output BEFORE expert computation, we can:
//! 1. See which experts will be needed 1-2 layers ahead
//! 2. Start async SSD prefetch for those experts
//! 3. Improve cache hit rate by preloading before access

use crate::{ExpertId, LayerIdx};
use crate::core::scheduler::ExpertScheduler;
use std::sync::Arc;

/// Router prediction for a single token
#[derive(Debug, Clone)]
pub struct RouterPrediction {
    pub layer: LayerIdx,
    pub token_idx: usize,
    pub expert_ids: Vec<ExpertId>,
    pub weights: Vec<f32>,
}

/// Hook for intercepting MoE router predictions
pub struct RouterPredictionHook {
    scheduler: Arc<ExpertScheduler>,
}

impl RouterPredictionHook {
    /// Create a new router hook with the given scheduler
    pub fn new(scheduler: Arc<ExpertScheduler>) -> Self {
        Self { scheduler }
    }

    /// Process a batch of router predictions
    ///
    /// Called after the router computes expert selections but BEFORE
    /// the actual expert computation happens.
    pub fn on_router_output(&self, predictions: &[RouterPrediction]) {
        for pred in predictions {
            // Trigger prefetch for the predicted experts
            self.scheduler.on_router_prediction(pred.layer, &pred.expert_ids);
        }
    }

    /// Install hook into llama.cpp router computation
    ///
    /// In practice, this would patch llama.cpp to call our callback
    /// after the router but before expert FFN computation.
    pub fn install_router_hook(&self) {
        // TODO: In production, this would:
        // 1. Patch llama_decode() or the MoE layer forward pass
        // 2. After router softmax, call on_router_output()
        // 3. Then proceed with expert computation
        //
        // Implementation options:
        // - Modify llama.cpp directly (fork)
        // - Add callback hook to llama.cpp (upstream patch)
        // - Intercept Metal/GPU kernels if using GPU router

        tracing::info!("Router prediction hook installed (placeholder)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory::{MemoryManager, ExpertRegion};
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_router_hook_prefetch() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 1024 * 1024]).unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        for i in 0..16 {
            expert_map.insert(
                i,
                ExpertRegion {
                    offset: i * 64 * 1024,
                    length: 64 * 1024,
                },
            );
        }

        let memory = Arc::new(MemoryManager::new(file.path(), expert_map).unwrap());
        let scheduler = Arc::new(ExpertScheduler::new(memory, 64 * 1024 * 1024, 2));

        let hook = RouterPredictionHook::new(scheduler.clone());

        // Simulate router predictions for layer 0
        let predictions = vec![
            RouterPrediction {
                layer: 0,
                token_idx: 0,
                expert_ids: vec![0, 5, 12],
                weights: vec![0.5, 0.3, 0.2],
            },
        ];

        hook.on_router_output(&predictions);

        // Check that prefetch was triggered
        let stats = scheduler.stats();
        assert!(stats.prefetching > 0 || stats.loaded > 0);
    }
}
