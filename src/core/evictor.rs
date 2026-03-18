//! Temperature-based expert eviction.
//!
//! Tracks expert "temperature" (recency × frequency) and evicts cold experts
//! when memory pressure is high.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info};

use crate::core::memory::MemoryManager;
use crate::ExpertId;

/// Expert state with temperature tracking
#[derive(Debug, Clone)]
struct ExpertState {
    temperature: f64,
    last_access: Instant,
    access_count: u64,
}

impl ExpertState {
    fn new() -> Self {
        Self {
            temperature: 0.0,
            last_access: Instant::now(),
            access_count: 0,
        }
    }

    /// Bump temperature on access
    fn bump(&mut self, boost: f64) {
        self.temperature += boost;
        self.last_access = Instant::now();
        self.access_count += 1;
    }

    /// Decay temperature over time
    fn decay(&mut self, rate: f64, dt: Duration) {
        let decay_factor = (-rate * dt.as_secs_f64()).exp();
        self.temperature *= decay_factor;
    }
}

/// Temperature-based expert evictor
pub struct TemperatureEvictor {
    memory: Arc<MemoryManager>,
    states: Arc<Mutex<HashMap<ExpertId, ExpertState>>>,
    decay_rate: f64,
    min_resident_time: Duration,
    access_boost: f64,
}

impl TemperatureEvictor {
    /// Create a new temperature evictor
    ///
    /// # Parameters
    /// - `memory`: Memory manager for madvise operations
    /// - `decay_rate`: Temperature decay rate per second (e.g., 0.1 = 10% decay/sec)
    /// - `min_resident_time`: Minimum time an expert must stay resident before eviction
    /// - `access_boost`: Temperature boost on each access
    pub fn new(
        memory: Arc<MemoryManager>,
        decay_rate: f64,
        min_resident_time: Duration,
        access_boost: f64,
    ) -> Self {
        Self {
            memory,
            states: Arc::new(Mutex::new(HashMap::new())),
            decay_rate,
            min_resident_time,
            access_boost,
        }
    }

    /// Create with default parameters
    pub fn with_defaults(memory: Arc<MemoryManager>) -> Self {
        Self::new(
            memory,
            0.1,                        // 10% decay per second
            Duration::from_secs(5),     // 5 second minimum residency
            1.0,                        // +1.0 temp per access
        )
    }

    /// Record an expert access
    pub fn record_access(&self, expert_id: ExpertId) {
        let mut states = self.states.lock().unwrap();
        let state = states.entry(expert_id).or_insert_with(ExpertState::new);
        state.bump(self.access_boost);
        debug!(
            "Expert {} accessed, temp={:.2}, count={}",
            expert_id, state.temperature, state.access_count
        );
    }

    /// Decay all expert temperatures
    pub fn decay_all(&self) {
        let mut states = self.states.lock().unwrap();
        let now = Instant::now();

        for (id, state) in states.iter_mut() {
            let dt = now.duration_since(state.last_access);
            state.decay(self.decay_rate, dt);
        }
    }

    /// Get the coldest N experts eligible for eviction
    fn get_coldest(&self, n: usize) -> Vec<ExpertId> {
        let states = self.states.lock().unwrap();
        let now = Instant::now();

        let mut eligible: Vec<_> = states
            .iter()
            .filter(|(_, state)| {
                // Must have been resident for at least min_resident_time
                now.duration_since(state.last_access) >= self.min_resident_time
            })
            .map(|(&id, state)| (id, state.temperature))
            .collect();

        // Sort by temperature (coldest first)
        eligible.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        eligible.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Evict the N coldest experts
    pub fn evict_coldest(&self, n: usize) -> Result<Vec<ExpertId>, String> {
        let coldest = self.get_coldest(n);

        info!("Evicting {} coldest experts: {:?}", coldest.len(), coldest);

        let mut evicted = Vec::new();
        for expert_id in coldest {
            match self.memory.evict_expert(expert_id) {
                Ok(_) => {
                    debug!("Evicted expert {}", expert_id);
                    evicted.push(expert_id);

                    // Remove from state tracking
                    self.states.lock().unwrap().remove(&expert_id);
                }
                Err(e) => {
                    return Err(format!("Failed to evict expert {}: {}", expert_id, e));
                }
            }
        }

        Ok(evicted)
    }

    /// Evict experts until memory pressure is below threshold
    pub fn evict_to_pressure(&self, target_pressure: f64, batch_size: usize) -> Result<usize, String> {
        let mut total_evicted = 0;

        while self.memory.memory_pressure() > target_pressure {
            let evicted = self.evict_coldest(batch_size)?;
            if evicted.is_empty() {
                break; // No more experts to evict
            }
            total_evicted += evicted.len();
        }

        info!("Evicted {} experts to reach target pressure", total_evicted);
        Ok(total_evicted)
    }

    /// Get current temperature for an expert
    pub fn get_temperature(&self, expert_id: ExpertId) -> Option<f64> {
        self.states.lock().unwrap().get(&expert_id).map(|s| s.temperature)
    }

    /// Get statistics about tracked experts
    pub fn stats(&self) -> EvictorStats {
        let states = self.states.lock().unwrap();
        let temps: Vec<f64> = states.values().map(|s| s.temperature).collect();

        EvictorStats {
            num_tracked: states.len(),
            avg_temperature: if temps.is_empty() {
                0.0
            } else {
                temps.iter().sum::<f64>() / temps.len() as f64
            },
            max_temperature: temps.iter().cloned().fold(0.0, f64::max),
            min_temperature: temps.iter().cloned().fold(f64::INFINITY, f64::min),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvictorStats {
    pub num_tracked: usize,
    pub avg_temperature: f64,
    pub max_temperature: f64,
    pub min_temperature: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory::ExpertRegion;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_temperature_tracking() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 1024]).unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        expert_map.insert(0, ExpertRegion { offset: 0, length: 512 });
        expert_map.insert(1, ExpertRegion { offset: 512, length: 512 });

        let memory = Arc::new(MemoryManager::new(file.path(), expert_map).unwrap());
        let evictor = TemperatureEvictor::with_defaults(memory);

        evictor.record_access(0);
        evictor.record_access(0);
        evictor.record_access(1);

        let temp0 = evictor.get_temperature(0).unwrap();
        let temp1 = evictor.get_temperature(1).unwrap();

        assert!(temp0 > temp1); // Expert 0 accessed more

        let stats = evictor.stats();
        assert_eq!(stats.num_tracked, 2);
    }

    #[test]
    fn test_temperature_decay() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 1024]).unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        expert_map.insert(0, ExpertRegion { offset: 0, length: 512 });

        let memory = Arc::new(MemoryManager::new(file.path(), expert_map).unwrap());
        let evictor = TemperatureEvictor::with_defaults(memory);

        evictor.record_access(0);
        let temp_before = evictor.get_temperature(0).unwrap();

        std::thread::sleep(Duration::from_millis(100));
        evictor.decay_all();

        let temp_after = evictor.get_temperature(0).unwrap();
        assert!(temp_after < temp_before);
    }
}
