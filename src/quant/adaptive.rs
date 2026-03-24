//! Adaptive expert quantization.
//!
//! Hot experts stay at full precision (4-bit), cold experts get downgraded
//! to 2-bit for memory savings. Inspired by JANG-style quantization.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

use crate::ExpertId;

/// Quantization level for an expert
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantLevel {
    /// Full precision (unquantized)
    Full,
    /// 4-bit quantization (standard)
    Q4,
    /// 3-bit quantization (aggressive)
    Q3,
    /// 2-bit quantization (maximum compression)
    Q2,
}

impl QuantLevel {
    /// Get the memory multiplier for this quantization level
    /// (relative to full precision)
    pub fn memory_multiplier(&self) -> f64 {
        match self {
            QuantLevel::Full => 1.0,
            QuantLevel::Q4 => 0.25,
            QuantLevel::Q3 => 0.1875,
            QuantLevel::Q2 => 0.125,
        }
    }

    /// Get bytes per parameter for this quantization level
    pub fn bytes_per_param(&self) -> f64 {
        match self {
            QuantLevel::Full => 4.0,  // FP32
            QuantLevel::Q4 => 0.5,    // 4 bits = 0.5 bytes
            QuantLevel::Q3 => 0.375,  // 3 bits = 0.375 bytes
            QuantLevel::Q2 => 0.25,   // 2 bits = 0.25 bytes
        }
    }

    /// Get the next higher precision level (promotion)
    pub fn promote(&self) -> Option<QuantLevel> {
        match self {
            QuantLevel::Q2 => Some(QuantLevel::Q3),
            QuantLevel::Q3 => Some(QuantLevel::Q4),
            QuantLevel::Q4 => Some(QuantLevel::Full),
            QuantLevel::Full => None,
        }
    }

    /// Get the next lower precision level (demotion)
    pub fn demote(&self) -> Option<QuantLevel> {
        match self {
            QuantLevel::Full => Some(QuantLevel::Q4),
            QuantLevel::Q4 => Some(QuantLevel::Q3),
            QuantLevel::Q3 => Some(QuantLevel::Q2),
            QuantLevel::Q2 => None,
        }
    }
}

/// Expert quantization state
#[derive(Debug, Clone)]
struct ExpertQuantState {
    current_level: QuantLevel,
    original_size_bytes: usize,
}

impl ExpertQuantState {
    fn new(size_bytes: usize, initial_level: QuantLevel) -> Self {
        Self {
            current_level: initial_level,
            original_size_bytes: size_bytes,
        }
    }

    fn current_size_bytes(&self) -> usize {
        (self.original_size_bytes as f64 * self.current_level.memory_multiplier()) as usize
    }
}

/// Adaptive quantizer for MoE experts
pub struct AdaptiveQuantizer {
    states: Arc<Mutex<HashMap<ExpertId, ExpertQuantState>>>,
    default_level: QuantLevel,
    hot_threshold: f64,
    cold_threshold: f64,
}

impl AdaptiveQuantizer {
    /// Create a new adaptive quantizer
    ///
    /// # Parameters
    /// - `default_level`: Default quantization level for new experts
    /// - `hot_threshold`: Temperature threshold above which experts are "hot"
    /// - `cold_threshold`: Temperature threshold below which experts are "cold"
    pub fn new(default_level: QuantLevel, hot_threshold: f64, cold_threshold: f64) -> Self {
        Self {
            states: Arc::new(Mutex::new(HashMap::new())),
            default_level,
            hot_threshold,
            cold_threshold,
        }
    }

    /// Create with default parameters (Q4 default, hot > 5.0, cold < 1.0)
    pub fn with_defaults() -> Self {
        Self::new(QuantLevel::Q4, 5.0, 1.0)
    }

    /// Register an expert with its original size
    pub fn register_expert(&self, expert_id: ExpertId, size_bytes: usize) {
        let mut states = self.states.lock().unwrap();
        states.insert(
            expert_id,
            ExpertQuantState::new(size_bytes, self.default_level),
        );
        debug!(
            "Registered expert {} with size {} bytes at {:?}",
            expert_id, size_bytes, self.default_level
        );
    }

    /// Promote an expert to higher precision (when it gets hot)
    ///
    /// Returns the new quantization level, or None if already at max precision
    pub fn promote(&self, expert_id: ExpertId) -> Option<QuantLevel> {
        let mut states = self.states.lock().unwrap();

        if let Some(state) = states.get_mut(&expert_id) {
            if let Some(new_level) = state.current_level.promote() {
                let old_level = state.current_level;
                state.current_level = new_level;
                info!(
                    "Promoted expert {} from {:?} to {:?}",
                    expert_id, old_level, new_level
                );
                return Some(new_level);
            } else {
                debug!("Expert {} already at max precision", expert_id);
            }
        }

        None
    }

    /// Demote an expert to lower precision (when it gets cold)
    ///
    /// Returns the new quantization level, or None if already at min precision
    pub fn demote(&self, expert_id: ExpertId) -> Option<QuantLevel> {
        let mut states = self.states.lock().unwrap();

        if let Some(state) = states.get_mut(&expert_id) {
            if let Some(new_level) = state.current_level.demote() {
                let old_level = state.current_level;
                state.current_level = new_level;
                info!(
                    "Demoted expert {} from {:?} to {:?}",
                    expert_id, old_level, new_level
                );
                return Some(new_level);
            } else {
                debug!("Expert {} already at min precision", expert_id);
            }
        }

        None
    }

    /// Update quantization level based on expert temperature
    ///
    /// Returns true if the level changed
    pub fn update_for_temperature(&self, expert_id: ExpertId, temperature: f64) -> bool {
        if temperature > self.hot_threshold {
            self.promote(expert_id).is_some()
        } else if temperature < self.cold_threshold {
            self.demote(expert_id).is_some()
        } else {
            false
        }
    }

    /// Get the current quantization level for an expert
    pub fn get_level(&self, expert_id: ExpertId) -> Option<QuantLevel> {
        let states = self.states.lock().unwrap();
        states.get(&expert_id).map(|s| s.current_level)
    }

    /// Get the current size in bytes for an expert
    pub fn get_current_size(&self, expert_id: ExpertId) -> Option<usize> {
        let states = self.states.lock().unwrap();
        states.get(&expert_id).map(|s| s.current_size_bytes())
    }

    /// Calculate total memory savings from adaptive quantization
    ///
    /// Returns (current_total_bytes, savings_bytes, savings_percentage)
    pub fn memory_savings(&self) -> (usize, usize, f64) {
        let states = self.states.lock().unwrap();

        let mut total_original = 0usize;
        let mut total_current = 0usize;

        for state in states.values() {
            total_original += state.original_size_bytes;
            total_current += state.current_size_bytes();
        }

        let savings = total_original.saturating_sub(total_current);
        let savings_pct = if total_original > 0 {
            (savings as f64 / total_original as f64) * 100.0
        } else {
            0.0
        };

        (total_current, savings, savings_pct)
    }

    /// Get statistics about quantization levels
    pub fn stats(&self) -> QuantizerStats {
        let states = self.states.lock().unwrap();

        let mut level_counts: HashMap<QuantLevel, usize> = HashMap::new();
        for state in states.values() {
            *level_counts.entry(state.current_level).or_insert(0) += 1;
        }

        drop(states); // Release lock
        let (total_bytes, savings_bytes, savings_pct) = self.memory_savings();

        QuantizerStats {
            num_experts: self.states.lock().unwrap().len(),
            full_count: *level_counts.get(&QuantLevel::Full).unwrap_or(&0),
            q4_count: *level_counts.get(&QuantLevel::Q4).unwrap_or(&0),
            q3_count: *level_counts.get(&QuantLevel::Q3).unwrap_or(&0),
            q2_count: *level_counts.get(&QuantLevel::Q2).unwrap_or(&0),
            total_bytes,
            savings_bytes,
            savings_pct,
        }
    }

    /// Reset all experts to default quantization level
    pub fn reset_all(&self) {
        let mut states = self.states.lock().unwrap();
        for state in states.values_mut() {
            state.current_level = self.default_level;
        }
        info!("Reset all experts to {:?}", self.default_level);
    }
}

/// Statistics about the adaptive quantizer
#[derive(Debug, Clone)]
pub struct QuantizerStats {
    pub num_experts: usize,
    pub full_count: usize,
    pub q4_count: usize,
    pub q3_count: usize,
    pub q2_count: usize,
    pub total_bytes: usize,
    pub savings_bytes: usize,
    pub savings_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_level_multipliers() {
        assert_eq!(QuantLevel::Full.memory_multiplier(), 1.0);
        assert_eq!(QuantLevel::Q4.memory_multiplier(), 0.25);
        assert_eq!(QuantLevel::Q2.memory_multiplier(), 0.125);
    }

    #[test]
    fn test_quant_level_promotion() {
        assert_eq!(QuantLevel::Q2.promote(), Some(QuantLevel::Q3));
        assert_eq!(QuantLevel::Q3.promote(), Some(QuantLevel::Q4));
        assert_eq!(QuantLevel::Q4.promote(), Some(QuantLevel::Full));
        assert_eq!(QuantLevel::Full.promote(), None);
    }

    #[test]
    fn test_quant_level_demotion() {
        assert_eq!(QuantLevel::Full.demote(), Some(QuantLevel::Q4));
        assert_eq!(QuantLevel::Q4.demote(), Some(QuantLevel::Q3));
        assert_eq!(QuantLevel::Q3.demote(), Some(QuantLevel::Q2));
        assert_eq!(QuantLevel::Q2.demote(), None);
    }

    #[test]
    fn test_adaptive_quantizer() {
        let quantizer = AdaptiveQuantizer::with_defaults();

        // Register experts
        quantizer.register_expert(0, 1_000_000);
        quantizer.register_expert(1, 1_000_000);

        // Initially at Q4
        assert_eq!(quantizer.get_level(0), Some(QuantLevel::Q4));

        // Promote hot expert
        quantizer.promote(0);
        assert_eq!(quantizer.get_level(0), Some(QuantLevel::Full));

        // Demote cold expert
        quantizer.demote(1);
        assert_eq!(quantizer.get_level(1), Some(QuantLevel::Q3));
    }

    #[test]
    fn test_temperature_based_update() {
        let quantizer = AdaptiveQuantizer::with_defaults();

        quantizer.register_expert(0, 1_000_000);
        quantizer.register_expert(1, 1_000_000);

        // Hot expert gets promoted
        assert!(quantizer.update_for_temperature(0, 10.0));
        assert_eq!(quantizer.get_level(0), Some(QuantLevel::Full));

        // Cold expert gets demoted
        assert!(quantizer.update_for_temperature(1, 0.5));
        assert_eq!(quantizer.get_level(1), Some(QuantLevel::Q3));
    }

    #[test]
    fn test_memory_savings() {
        let quantizer = AdaptiveQuantizer::with_defaults();

        // 1MB expert at Q4 = 250KB
        quantizer.register_expert(0, 1_000_000);

        // Demote to Q2 = 125KB
        quantizer.demote(0);
        quantizer.demote(0);

        let (current, savings, pct) = quantizer.memory_savings();
        assert_eq!(current, 125_000); // Q2 = 12.5%
        assert_eq!(savings, 875_000);
        assert!((pct - 87.5).abs() < 0.1);
    }
}
