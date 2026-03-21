//! KV cache-aware memory budget.
//!
//! Dynamic expert memory budget that adjusts based on KV cache pressure.
//! Long conversations = big KV cache = less room for experts.

use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

/// Memory budget tracker with KV cache awareness
#[derive(Debug)]
pub struct MemoryBudget {
    inner: Arc<Mutex<MemoryBudgetInner>>,
}

#[derive(Debug)]
struct MemoryBudgetInner {
    total_ram: usize,
    kv_cache_size: usize,
    system_reserve: usize,
    min_expert_budget: usize,
}

impl MemoryBudget {
    /// Create a new memory budget tracker
    ///
    /// # Parameters
    /// - `total_ram`: Total RAM available for the system (bytes)
    /// - `system_reserve`: RAM to reserve for OS and other processes (bytes)
    /// - `min_expert_budget`: Minimum RAM to guarantee for experts (bytes)
    pub fn new(total_ram: usize, system_reserve: usize, min_expert_budget: usize) -> Self {
        info!(
            "Initializing memory budget: total={:.2}GB, reserve={:.2}GB, min_expert={:.2}GB",
            total_ram as f64 / 1e9,
            system_reserve as f64 / 1e9,
            min_expert_budget as f64 / 1e9
        );

        Self {
            inner: Arc::new(Mutex::new(MemoryBudgetInner {
                total_ram,
                kv_cache_size: 0,
                system_reserve,
                min_expert_budget,
            })),
        }
    }

    /// Create with reasonable defaults for the given total RAM
    ///
    /// Reserves 20% for system, guarantees 10GB minimum for experts
    pub fn with_defaults(total_ram: usize) -> Self {
        let system_reserve = (total_ram as f64 * 0.2) as usize;
        let min_expert_budget = 10 * 1024 * 1024 * 1024; // 10 GB
        Self::new(total_ram, system_reserve, min_expert_budget)
    }

    /// Reserve memory for KV cache
    ///
    /// Called by the inference engine when allocating KV cache memory
    pub fn reserve_kv(&self, bytes: usize) {
        let mut inner = self.inner.lock().unwrap();
        let old_size = inner.kv_cache_size;
        inner.kv_cache_size += bytes;

        debug!(
            "Reserved {:.2}MB for KV cache, total KV cache: {:.2}MB",
            bytes as f64 / 1e6,
            inner.kv_cache_size as f64 / 1e6
        );

        // Warn if KV cache is getting large
        let kv_ratio = inner.kv_cache_size as f64 / inner.total_ram as f64;
        if kv_ratio > 0.5 {
            warn!(
                "KV cache consuming {:.1}% of total RAM ({:.2}GB / {:.2}GB)",
                kv_ratio * 100.0,
                inner.kv_cache_size as f64 / 1e9,
                inner.total_ram as f64 / 1e9
            );
        }
    }

    /// Release memory from KV cache
    ///
    /// Called when KV cache is freed (e.g., conversation cleared)
    pub fn release_kv(&self, bytes: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.kv_cache_size = inner.kv_cache_size.saturating_sub(bytes);

        debug!(
            "Released {:.2}MB from KV cache, total KV cache: {:.2}MB",
            bytes as f64 / 1e6,
            inner.kv_cache_size as f64 / 1e6
        );
    }

    /// Set KV cache size directly (for external KV cache implementations)
    pub fn set_kv_size(&self, bytes: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.kv_cache_size = bytes;

        debug!(
            "Set KV cache size to {:.2}MB",
            inner.kv_cache_size as f64 / 1e6
        );
    }

    /// Get the current KV cache size
    pub fn kv_cache_size(&self) -> usize {
        self.inner.lock().unwrap().kv_cache_size
    }

    /// Get the expert memory budget (dynamic, based on KV cache pressure)
    ///
    /// Returns: `total_ram - kv_cache_size - system_reserve`, clamped to min_expert_budget
    pub fn expert_budget(&self) -> usize {
        let inner = self.inner.lock().unwrap();

        let available = inner
            .total_ram
            .saturating_sub(inner.kv_cache_size)
            .saturating_sub(inner.system_reserve);

        let budget = available.max(inner.min_expert_budget);

        debug!(
            "Expert budget: {:.2}GB (total={:.2}GB, KV={:.2}GB, reserve={:.2}GB)",
            budget as f64 / 1e9,
            inner.total_ram as f64 / 1e9,
            inner.kv_cache_size as f64 / 1e9,
            inner.system_reserve as f64 / 1e9
        );

        budget
    }

    /// Get the ratio of memory consumed by KV cache (0.0 - 1.0)
    pub fn kv_pressure(&self) -> f64 {
        let inner = self.inner.lock().unwrap();
        if inner.total_ram == 0 {
            return 0.0;
        }
        inner.kv_cache_size as f64 / inner.total_ram as f64
    }

    /// Get the ratio of memory available for experts (0.0 - 1.0)
    pub fn expert_budget_ratio(&self) -> f64 {
        let inner = self.inner.lock().unwrap();
        if inner.total_ram == 0 {
            return 0.0;
        }
        self.expert_budget() as f64 / inner.total_ram as f64
    }

    /// Check if expert budget is below minimum threshold
    pub fn is_expert_budget_critical(&self) -> bool {
        let budget = self.expert_budget();
        let inner = self.inner.lock().unwrap();
        budget <= inner.min_expert_budget
    }

    /// Get statistics about memory budget
    pub fn stats(&self) -> MemoryBudgetStats {
        let inner = self.inner.lock().unwrap();
        let expert_budget = drop(inner); // Release lock
        let expert_budget = self.expert_budget();

        let inner = self.inner.lock().unwrap();

        MemoryBudgetStats {
            total_ram: inner.total_ram,
            kv_cache_size: inner.kv_cache_size,
            system_reserve: inner.system_reserve,
            expert_budget,
            kv_pressure: self.kv_pressure(),
            expert_budget_ratio: self.expert_budget_ratio(),
        }
    }

    /// Reset KV cache tracking (e.g., on conversation restart)
    pub fn reset_kv(&self) {
        let mut inner = self.inner.lock().unwrap();
        info!("Resetting KV cache (was {:.2}MB)", inner.kv_cache_size as f64 / 1e6);
        inner.kv_cache_size = 0;
    }
}

impl Clone for MemoryBudget {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Statistics about memory budget
#[derive(Debug, Clone)]
pub struct MemoryBudgetStats {
    pub total_ram: usize,
    pub kv_cache_size: usize,
    pub system_reserve: usize,
    pub expert_budget: usize,
    pub kv_pressure: f64,
    pub expert_budget_ratio: f64,
}

impl MemoryBudgetStats {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Memory Budget: total={:.2}GB, KV={:.2}GB ({:.1}%), expert={:.2}GB ({:.1}%), reserve={:.2}GB",
            self.total_ram as f64 / 1e9,
            self.kv_cache_size as f64 / 1e9,
            self.kv_pressure * 100.0,
            self.expert_budget as f64 / 1e9,
            self.expert_budget_ratio * 100.0,
            self.system_reserve as f64 / 1e9,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_creation() {
        let total = 128 * 1024 * 1024 * 1024; // 128 GB
        let reserve = 20 * 1024 * 1024 * 1024; // 20 GB
        let min_expert = 10 * 1024 * 1024 * 1024; // 10 GB

        let budget = MemoryBudget::new(total, reserve, min_expert);

        // Initially, no KV cache, so expert budget = total - reserve
        assert_eq!(budget.expert_budget(), total - reserve);
        assert_eq!(budget.kv_cache_size(), 0);
    }

    #[test]
    fn test_kv_cache_reservation() {
        let total = 128 * 1024 * 1024 * 1024; // 128 GB
        let budget = MemoryBudget::with_defaults(total);

        // Reserve 10GB for KV cache
        let kv_size = 10 * 1024 * 1024 * 1024;
        budget.reserve_kv(kv_size);

        assert_eq!(budget.kv_cache_size(), kv_size);

        // Expert budget should be reduced
        let expert = budget.expert_budget();
        assert!(expert < total);
    }

    #[test]
    fn test_kv_cache_release() {
        let total = 128 * 1024 * 1024 * 1024; // 128 GB
        let budget = MemoryBudget::with_defaults(total);

        let kv_size = 10 * 1024 * 1024 * 1024;
        budget.reserve_kv(kv_size);
        budget.release_kv(kv_size / 2);

        assert_eq!(budget.kv_cache_size(), kv_size / 2);
    }

    #[test]
    fn test_dynamic_expert_budget() {
        let total = 100 * 1024 * 1024 * 1024; // 100 GB
        let reserve = 20 * 1024 * 1024 * 1024; // 20 GB
        let min_expert = 10 * 1024 * 1024 * 1024; // 10 GB

        let budget = MemoryBudget::new(total, reserve, min_expert);

        // Initially: 100GB - 20GB = 80GB for experts
        assert_eq!(budget.expert_budget(), 80 * 1024 * 1024 * 1024);

        // Add 30GB KV cache: 100GB - 30GB - 20GB = 50GB for experts
        budget.reserve_kv(30 * 1024 * 1024 * 1024);
        assert_eq!(budget.expert_budget(), 50 * 1024 * 1024 * 1024);

        // Add 50GB more KV cache: would be 0GB, but clamped to min_expert
        budget.reserve_kv(50 * 1024 * 1024 * 1024);
        assert_eq!(budget.expert_budget(), min_expert);
    }

    #[test]
    fn test_kv_pressure() {
        let total = 100 * 1024 * 1024 * 1024; // 100 GB
        let budget = MemoryBudget::with_defaults(total);

        assert_eq!(budget.kv_pressure(), 0.0);

        budget.reserve_kv(50 * 1024 * 1024 * 1024); // 50GB
        assert!((budget.kv_pressure() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_critical_budget() {
        let total = 100 * 1024 * 1024 * 1024; // 100 GB
        let reserve = 20 * 1024 * 1024 * 1024; // 20 GB
        let min_expert = 30 * 1024 * 1024 * 1024; // 30 GB

        let budget = MemoryBudget::new(total, reserve, min_expert);

        assert!(!budget.is_expert_budget_critical());

        // Fill up KV cache to force expert budget to minimum
        budget.reserve_kv(70 * 1024 * 1024 * 1024);
        assert!(budget.is_expert_budget_critical());
    }
}
