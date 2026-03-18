//! Expert activation heatmap for visualizing routing patterns.

use crate::{ExpertId, LayerIdx};
use std::collections::HashMap;

/// Expert activation heatmap
pub struct ExpertHeatmap {
    activations: HashMap<(LayerIdx, ExpertId), usize>,
}

impl ExpertHeatmap {
    /// Create a new empty heatmap
    pub fn new() -> Self {
        Self {
            activations: HashMap::new(),
        }
    }

    /// Record an expert activation
    pub fn record(&mut self, layer: LayerIdx, expert: ExpertId) {
        *self.activations.entry((layer, expert)).or_insert(0) += 1;
    }

    /// Get activation count for an expert
    pub fn get(&self, layer: LayerIdx, expert: ExpertId) -> usize {
        self.activations.get(&(layer, expert)).copied().unwrap_or(0)
    }

    /// Get total number of activations
    pub fn total_activations(&self) -> usize {
        self.activations.values().sum()
    }
}

impl Default for ExpertHeatmap {
    fn default() -> Self {
        Self::new()
    }
}
