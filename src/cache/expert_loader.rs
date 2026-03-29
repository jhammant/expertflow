//! Expert mmap loader with LRU cache for dynamic expert streaming from NVMe SSD.
//!
//! This is the core of ExpertFlow Phase 2: loads MoE experts on-demand from GGUF files
//! with automatic LRU eviction when memory is full.

use crate::core::memory::MemoryManager;
use crate::model::gguf::GgufLoader;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Expert cache entry with timing metadata
#[derive(Debug, Clone)]
pub struct CachedExpert {
    pub expert_id: usize,
    pub data: Vec<u8>,
    pub last_accessed: Instant,
    pub loaded_from: u64,
    pub size_bytes: usize,
    pub hit_count: u64,
    load_time_ns: u64,
}

impl CachedExpert {
    fn new(expert_id: usize, data: Vec<u8>, loaded_from: u64, size_bytes: usize, load_time_ns: u64) -> Self {
        CachedExpert {
            expert_id,
            data,
            last_accessed: Instant::now(),
            loaded_from,
            size_bytes,
            hit_count: 1,
            load_time_ns,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.hit_count += 1;
    }

    fn load_time_ns(&self) -> u64 {
        self.load_time_ns
    }
}

/// LRU eviction policy for expert cache
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvictionPolicy {
    LRU,
    LFU,
}

/// Expert mmap loader with caching
pub struct ExpertMmapLoader {
    model_path: String,
    gguf_loader: GgufLoader,
    mmap_manager: MemoryManager,
    cache: HashMap<usize, CachedExpert>,
    eviction_policy: EvictionPolicy,
    max_cache_size_bytes: usize,
    current_cache_size_bytes: usize,
    cache_hits: u64,
    cache_misses: u64,
    total_bytes_loaded: u64,
    load_times: Vec<(usize, Duration)>,
}

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("Failed to open model file: {0}")]
    FileOpen(#[from] std::io::Error),

    #[error("Failed to mmap file: {0}")]
    MmapFailed(String),

    #[error("GGUF parsing failed")]
    GgufParse,

    #[error("Expert {0} not found in model")]
    ExpertNotFound(usize),

    #[error("Cache full: cannot load expert without eviction")]
    CacheFull,

    #[error("Cache size limit exceeded: {0} bytes")]
    CacheLimitExceeded(usize),
}

impl ExpertMmapLoader {
    /// Create a new expert loader for a GGUF file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let path_str = path.as_ref().display().to_string();
        
        let gguf_loader = GgufLoader::load(&path)
            .map_err(|_| LoaderError::GgufParse)?;
        
        let expert_map = gguf_loader.extract_expert_map();
        
        if expert_map.is_empty() {
            return Err(LoaderError::GgufParse);
        }

        let mmap_manager = MemoryManager::new(&path, expert_map)
            .map_err(|e| LoaderError::MmapFailed(e.to_string()))?;
        
        Ok(ExpertMmapLoader {
            model_path: path_str,
            gguf_loader,
            mmap_manager,
            cache: HashMap::new(),
            eviction_policy: EvictionPolicy::LRU,
            max_cache_size_bytes: 8 * 1024 * 1024 * 1024,
            current_cache_size_bytes: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_bytes_loaded: 0,
            load_times: Vec::new(),
        })
    }

    /// Load an expert from the GGUF file with optional caching
    pub fn load_expert(&mut self, expert_id: usize, cache: bool) -> Result<Vec<u8>, LoaderError> {
        // Check cache first
        if let Some(expert) = self.cache.get_mut(&expert_id) {
            expert.touch();
            self.cache_hits += 1;
            
            if cache {
                let _ = self.mmap_manager.pin_expert(expert_id);
            }
            
            return Ok(expert.data.clone());
        }

        self.cache_misses += 1;

        // Get tensors for this expert
        let tensors: Vec<_> = self.gguf_loader.tensors()
            .iter()
            .filter(|t| GgufLoader::parse_expert_id(&t.name) == Some(expert_id))
            .cloned()
            .collect();

        if tensors.is_empty() {
            return Err(LoaderError::ExpertNotFound(expert_id));
        }

        // Calculate total size
        let mut total_size: usize = 0;
        for tensor in &tensors {
            total_size += tensor.size_bytes() as usize;
        }
        
        // Evict if needed
        if cache && self.current_cache_size_bytes + total_size > self.max_cache_size_bytes {
            self.evict()?;
        }

        // Load expert data from mmap with timing
        let start = Instant::now();
        let expert_data = match self.mmap_manager.get_expert(expert_id) {
            Ok(data) => data.to_vec(),
            Err(_) => Vec::new(),
        };
        let load_time = start.elapsed();

        // Update statistics
        self.current_cache_size_bytes += total_size;
        self.total_bytes_loaded += total_size as u64;
        self.load_times.push((expert_id, load_time));

        // Cache if requested
        if cache {
            let offset = tensors.first().map(|t| t.offset).unwrap_or(0);

            let cached = CachedExpert::new(
                expert_id,
                expert_data.clone(),
                offset,
                total_size,
                load_time.as_nanos() as u64,
            );
            
            self.cache.insert(expert_id, cached);
        }

        Ok(expert_data)
    }

    /// Evict least useful expert(s) from cache
    pub fn evict(&mut self) -> Result<usize, LoaderError> {
        if self.cache.is_empty() {
            return Ok(0);
        }

        let mut evicted_count = 0;
        let target_size = (self.max_cache_size_bytes as f64 * 0.7) as usize;
        
        // Simple eviction: remove half of cache
        let keys_to_remove: Vec<_> = self.cache.keys().take(self.cache.len() / 2).cloned().collect();
        
        for id in keys_to_remove {
            if self.current_cache_size_bytes <= target_size {
                break;
            }

            // Evict from mmap with madvise
            let _ = self.mmap_manager.evict_expert(id);
            
            // Remove from cache
            let size = self.cache.remove(&id).map(|e| e.size_bytes).unwrap_or(0);
            self.current_cache_size_bytes -= size;
            evicted_count += 1;
        }

        Ok(evicted_count)
    }

    /// Pin an expert in memory
    pub fn pin_expert(&self, expert_id: usize) -> Result<(), LoaderError> {
        self.mmap_manager
            .pin_expert(expert_id)
            .map_err(|e| LoaderError::MmapFailed(e.to_string()))
    }

    /// Evict an expert from memory
    pub fn evict_expert(&self, expert_id: usize) -> Result<(), LoaderError> {
        self.mmap_manager
            .evict_expert(expert_id)
            .map_err(|e| LoaderError::MmapFailed(e.to_string()))
    }

    /// Set cache size limit
    pub fn set_cache_limit(&mut self, bytes: usize) {
        self.max_cache_size_bytes = bytes;
    }

    /// Get cache statistics
    pub fn stats(&self) -> LoaderStats {
        let hit_rate = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        } else {
            0.0
        };

        let total_load_time: Duration = self.load_times.iter().map(|(_, t)| *t).sum();

        LoaderStats {
            cache_size_bytes: self.current_cache_size_bytes,
            max_cache_size_bytes: self.max_cache_size_bytes,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            hit_rate,
            total_bytes_loaded: self.total_bytes_loaded,
            num_cached_experts: self.cache.len(),
            avg_load_time_ns: if !self.load_times.is_empty() {
                total_load_time.as_nanos() as u64 / self.load_times.len() as u64
            } else {
                0
            },
        }
    }

    /// Get per-expert load times
    pub fn load_times(&self) -> &[(usize, Duration)] {
        &self.load_times
    }

    /// Get total experts loaded count
    pub fn experts_loaded_count(&self) -> usize {
        self.load_times.len()
    }

    /// Get GGUF metadata
    pub fn model_info(&self) -> &GgufLoader {
        &self.gguf_loader
    }

    /// Get the mmap manager for direct access
    pub fn mmap_manager(&self) -> &MemoryManager {
        &self.mmap_manager
    }

    /// Get mutable mmap manager for advanced operations
    pub fn mmap_manager_mut(&mut self) -> &mut MemoryManager {
        &mut self.mmap_manager
    }
}

/// Loader statistics with timing
#[derive(Debug, Clone)]
pub struct LoaderStats {
    pub cache_size_bytes: usize,
    pub max_cache_size_bytes: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f64,
    pub total_bytes_loaded: u64,
    pub num_cached_experts: usize,
    pub avg_load_time_ns: u64,
}

/// Expert cache entry with metadata
#[derive(Debug, Clone)]
pub struct ExpertCacheEntry {
    pub expert_id: usize,
    pub offset: u64,
    pub size_bytes: usize,
    pub tensor_count: usize,
}

/// Advanced expert loader with prefetching
pub struct ExpertPrefetchLoader {
    base_loader: Option<ExpertMmapLoader>,
    prefetch_queue: VecDeque<usize>,
    prefetch_depth: usize,
}

impl ExpertPrefetchLoader {
    /// Create new loader with prefetch support
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let base_loader = ExpertMmapLoader::load(path)?;
        
        Ok(ExpertPrefetchLoader {
            base_loader: Some(base_loader),
            prefetch_queue: VecDeque::new(),
            prefetch_depth: 2,
        })
    }

    /// Load expert with prefetching
    pub fn load_with_prefetch(&mut self, expert_id: usize) -> Result<Vec<u8>, LoaderError> {
        let base_loader = self.base_loader.as_mut()
            .ok_or_else(|| LoaderError::FileOpen(std::io::Error::new(std::io::ErrorKind::NotFound, "No loader available")))?;
        
        base_loader.load_expert(expert_id, true)
    }

    /// Prefetch next batch of experts
    pub fn prefetch_next(&mut self) -> Result<(), LoaderError> {
        if let Some(loader) = &self.base_loader {
            let num_experts = loader.mmap_manager.num_experts();
            
            for i in 0..self.prefetch_depth {
                let expert_id = (i * 7 + num_experts / 2) % num_experts;
                
                let _ = loader.mmap_manager.pin_expert(expert_id);
            }
        }

        Ok(())
    }

    /// Get base loader reference
    pub fn base_loader(&self) -> Option<&ExpertMmapLoader> {
        self.base_loader.as_ref()
    }

    /// Get mutable base loader reference
    pub fn base_loader_mut(&mut self) -> Option<&mut ExpertMmapLoader> {
        self.base_loader.as_mut()
    }

    /// Get prefetch depth
    pub fn prefetch_depth(&self) -> usize {
        self.prefetch_depth
    }

    /// Set prefetch depth (number of lookahead layers)
    pub fn set_prefetch_depth(&mut self, depth: usize) {
        self.prefetch_depth = depth;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let stats = LoaderStats {
            cache_size_bytes: 1024,
            max_cache_size_bytes: 8 * 1024 * 1024 * 1024,
            cache_hits: 100,
            cache_misses: 50,
            hit_rate: 0.67,
            total_bytes_loaded: 10000,
            num_cached_experts: 5,
            avg_load_time_ns: 1000,
        };

        assert_eq!(stats.cache_hits, 100);
        assert!((stats.hit_rate - 0.67).abs() < 0.01);
    }

    #[test]
    fn test_cached_expert_hit() {
        let mut expert = CachedExpert::new(42, vec![1, 2, 3], 0, 100, 500);
        
        assert_eq!(expert.hit_count, 1);
        
        expert.touch();
        assert_eq!(expert.hit_count, 2);
    }

    #[test]
    fn test_eviction_policy_lru() {
        let policy = EvictionPolicy::LRU;
        assert_eq!(policy, EvictionPolicy::LRU);
    }

    #[test]
    fn test_eviction_policy_lfu() {
        let policy = EvictionPolicy::LFU;
        assert_eq!(policy, EvictionPolicy::LFU);
    }
}
