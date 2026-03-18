//! Dynamic expert scheduler.
//!
//! Orchestrates expert loading, prefetching, and eviction based on router
//! predictions and memory pressure.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::core::evictor::TemperatureEvictor;
use crate::core::memory::{MemoryManager, MemoryError};
use crate::core::prefetcher::AsyncPrefetcher;
use crate::{ExpertId, LayerIdx};

/// Expert state in the scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertState {
    /// Expert is not loaded in memory
    Evicted,
    /// Expert is being prefetched
    Prefetching,
    /// Expert is loaded and ready
    Loaded,
}

/// Expert scheduler
pub struct ExpertScheduler {
    memory: Arc<MemoryManager>,
    prefetcher: AsyncPrefetcher,
    evictor: TemperatureEvictor,
    states: Arc<Mutex<HashMap<ExpertId, ExpertState>>>,
    ram_budget: usize,
    prefetch_lookahead: usize,
}

impl ExpertScheduler {
    /// Create a new expert scheduler
    ///
    /// # Parameters
    /// - `memory`: Memory manager
    /// - `ram_budget`: Maximum RAM to use for experts (bytes)
    /// - `prefetch_lookahead`: Number of layers to prefetch ahead
    pub fn new(memory: Arc<MemoryManager>, ram_budget: usize, prefetch_lookahead: usize) -> Self {
        let prefetcher = AsyncPrefetcher::new(Arc::clone(&memory));
        let evictor = TemperatureEvictor::with_defaults(Arc::clone(&memory));

        Self {
            memory,
            prefetcher,
            evictor,
            states: Arc::new(Mutex::new(HashMap::new())),
            ram_budget,
            prefetch_lookahead,
        }
    }

    /// Handle router prediction for upcoming layer.
    ///
    /// This is the main entry point for the scheduler. When the router predicts
    /// which experts will be needed, call this to trigger prefetching.
    pub fn on_router_prediction(&self, layer: LayerIdx, expert_ids: &[ExpertId]) {
        debug!("Router prediction for layer {}: {:?}", layer, expert_ids);

        // Check memory pressure and evict if needed
        let pressure = self.memory.memory_pressure();
        if pressure > 0.8 {
            warn!("High memory pressure: {:.2}", pressure);
            if let Err(e) = self.evictor.evict_to_pressure(0.6, 4) {
                warn!("Failed to evict experts: {}", e);
            }
        }

        // Trigger prefetch for predicted experts
        let mut handles = Vec::new();
        for &expert_id in expert_ids {
            let state = self.get_state(expert_id);

            match state {
                ExpertState::Evicted => {
                    // Start prefetch
                    debug!("Prefetching expert {}", expert_id);
                    self.set_state(expert_id, ExpertState::Prefetching);
                    let handle = self.prefetcher.prefetch(expert_id);
                    handles.push(handle);
                }
                ExpertState::Prefetching => {
                    debug!("Expert {} already prefetching", expert_id);
                }
                ExpertState::Loaded => {
                    debug!("Expert {} already loaded", expert_id);
                    // Record access for temperature tracking
                    self.evictor.record_access(expert_id);
                }
            }
        }

        // Don't wait for prefetches - they happen in background
        // The get_expert() call will block if needed
    }

    /// Get expert data, blocking if necessary.
    ///
    /// This is called when the expert is actually needed for computation.
    /// If the expert is still being prefetched, this will block until ready.
    pub fn get_expert(&self, expert_id: ExpertId) -> Result<&[u8], MemoryError> {
        let state = self.get_state(expert_id);

        match state {
            ExpertState::Evicted => {
                warn!("Expert {} not prefetched, blocking load", expert_id);
                // Trigger immediate load
                let _ = self.memory.pin_expert(expert_id);
                self.set_state(expert_id, ExpertState::Loaded);
            }
            ExpertState::Prefetching => {
                debug!("Expert {} prefetching, may block", expert_id);
                // Prefetch is in progress, this may block briefly
                self.set_state(expert_id, ExpertState::Loaded);
            }
            ExpertState::Loaded => {
                // Already loaded, no-op
            }
        }

        // Record access for temperature tracking
        self.evictor.record_access(expert_id);

        // Return the expert data
        self.memory.get_expert(expert_id)
    }

    /// Get the current state of an expert
    pub fn get_state(&self, expert_id: ExpertId) -> ExpertState {
        self.states
            .lock()
            .unwrap()
            .get(&expert_id)
            .copied()
            .unwrap_or(ExpertState::Evicted)
    }

    /// Set the state of an expert
    fn set_state(&self, expert_id: ExpertId, state: ExpertState) {
        self.states.lock().unwrap().insert(expert_id, state);
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> SchedulerStats {
        let states = self.states.lock().unwrap();

        let mut loaded = 0;
        let mut prefetching = 0;
        let mut evicted = 0;

        for state in states.values() {
            match state {
                ExpertState::Loaded => loaded += 1,
                ExpertState::Prefetching => prefetching += 1,
                ExpertState::Evicted => evicted += 1,
            }
        }

        let evictor_stats = self.evictor.stats();

        SchedulerStats {
            loaded,
            prefetching,
            evicted,
            memory_pressure: self.memory.memory_pressure(),
            avg_temperature: evictor_stats.avg_temperature,
        }
    }

    /// Run temperature decay (should be called periodically)
    pub fn tick(&self) {
        self.evictor.decay_all();
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub loaded: usize,
    pub prefetching: usize,
    pub evicted: usize,
    pub memory_pressure: f64,
    pub avg_temperature: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory::ExpertRegion;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_scheduler_creation() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 1024 * 1024]).unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        for i in 0..8 {
            expert_map.insert(
                i,
                ExpertRegion {
                    offset: i * 128 * 1024,
                    length: 128 * 1024,
                },
            );
        }

        let memory = Arc::new(MemoryManager::new(file.path(), expert_map).unwrap());
        let scheduler = ExpertScheduler::new(memory, 64 * 1024 * 1024, 2);

        assert_eq!(scheduler.get_state(0), ExpertState::Evicted);
    }

    #[test]
    fn test_scheduler_prediction() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 1024 * 1024]).unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        for i in 0..8 {
            expert_map.insert(
                i,
                ExpertRegion {
                    offset: i * 128 * 1024,
                    length: 128 * 1024,
                },
            );
        }

        let memory = Arc::new(MemoryManager::new(file.path(), expert_map).unwrap());
        let scheduler = ExpertScheduler::new(memory, 64 * 1024 * 1024, 2);

        scheduler.on_router_prediction(0, &[0, 1, 2]);

        // Give prefetch a moment
        std::thread::sleep(std::time::Duration::from_millis(10));

        let stats = scheduler.stats();
        assert!(stats.prefetching > 0 || stats.loaded > 0);
    }
}
