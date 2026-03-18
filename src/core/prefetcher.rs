//! Async prefetcher for expert weights.
//!
//! Uses madvise(MADV_WILLNEED) to hint to the kernel that memory will be
//! accessed soon, triggering async page-in from disk.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::core::memory::MemoryManager;
use crate::ExpertId;

/// Async prefetcher for expert weights
pub struct AsyncPrefetcher {
    memory: Arc<MemoryManager>,
    pending: Arc<Mutex<HashSet<ExpertId>>>,
}

impl AsyncPrefetcher {
    /// Create a new async prefetcher
    pub fn new(memory: Arc<MemoryManager>) -> Self {
        Self {
            memory,
            pending: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Prefetch a single expert asynchronously.
    ///
    /// Returns a JoinHandle that completes when the prefetch hint is issued.
    /// The actual page-in happens asynchronously in the kernel.
    pub fn prefetch(&self, expert_id: ExpertId) -> JoinHandle<()> {
        let memory = Arc::clone(&self.memory);
        let pending = Arc::clone(&self.pending);

        // Check if already pending
        {
            let mut pending_set = pending.lock().unwrap();
            if pending_set.contains(&expert_id) {
                debug!("Expert {} already pending prefetch", expert_id);
                return tokio::spawn(async {});
            }
            pending_set.insert(expert_id);
        }

        tokio::spawn(async move {
            debug!("Prefetching expert {}", expert_id);

            match memory.pin_expert(expert_id) {
                Ok(_) => debug!("Prefetch initiated for expert {}", expert_id),
                Err(e) => warn!("Failed to prefetch expert {}: {}", expert_id, e),
            }

            // Remove from pending set
            pending.lock().unwrap().remove(&expert_id);
        })
    }

    /// Prefetch multiple experts in parallel.
    ///
    /// Returns a Vec of JoinHandles that can be awaited.
    pub fn prefetch_batch(&self, expert_ids: &[ExpertId]) -> Vec<JoinHandle<()>> {
        expert_ids.iter().map(|&id| self.prefetch(id)).collect()
    }

    /// Wait for all pending prefetches to complete
    pub async fn wait_all(&self, handles: Vec<JoinHandle<()>>) {
        for handle in handles {
            let _ = handle.await;
        }
    }

    /// Get number of pending prefetches
    pub fn num_pending(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    /// Check if an expert is currently being prefetched
    pub fn is_pending(&self, expert_id: ExpertId) -> bool {
        self.pending.lock().unwrap().contains(&expert_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory::ExpertRegion;
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_prefetcher_single() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 1024 * 1024]).unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        expert_map.insert(0, ExpertRegion { offset: 0, length: 512 * 1024 });
        expert_map.insert(1, ExpertRegion { offset: 512 * 1024, length: 512 * 1024 });

        let memory = Arc::new(MemoryManager::new(file.path(), expert_map).unwrap());
        let prefetcher = AsyncPrefetcher::new(memory);

        let handle = prefetcher.prefetch(0);
        handle.await.unwrap();

        assert_eq!(prefetcher.num_pending(), 0);
    }

    #[tokio::test]
    async fn test_prefetcher_batch() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 1024 * 1024]).unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        for i in 0..4 {
            expert_map.insert(
                i,
                ExpertRegion {
                    offset: i * 256 * 1024,
                    length: 256 * 1024,
                },
            );
        }

        let memory = Arc::new(MemoryManager::new(file.path(), expert_map).unwrap());
        let prefetcher = AsyncPrefetcher::new(memory);

        let handles = prefetcher.prefetch_batch(&[0, 1, 2, 3]);
        prefetcher.wait_all(handles).await;

        assert_eq!(prefetcher.num_pending(), 0);
    }
}
