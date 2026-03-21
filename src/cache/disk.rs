//! Disk-backed expert cache for persistent temperature tracking.
//!
//! Saves expert activation patterns to disk on shutdown, loads on startup
//! to warm-start the scheduler with hot experts from previous session.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::ExpertId;

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Failed to read cache file: {0}")]
    ReadFailed(#[from] std::io::Error),

    #[error("Failed to parse cache JSON: {0}")]
    ParseFailed(#[from] serde_json::Error),

    #[error("Cache directory does not exist and could not be created")]
    DirectoryCreateFailed,

    #[error("Cache file not found")]
    NotFound,
}

/// Cache entry for a single expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub expert_id: ExpertId,
    pub temperature: f64,
    pub access_count: u64,
    pub last_session_time: u64, // Unix timestamp
}

/// Expert activation cache stored on disk
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheData {
    version: u32,
    session_timestamp: u64,
    entries: Vec<CacheEntry>,
}

impl Default for CacheData {
    fn default() -> Self {
        Self {
            version: 1,
            session_timestamp: 0,
            entries: Vec::new(),
        }
    }
}

/// Disk-backed expert cache
pub struct DiskCache {
    cache_path: PathBuf,
    entries: HashMap<ExpertId, CacheEntry>,
}

impl DiskCache {
    /// Create a new disk cache at the default location (~/.expertflow/cache.json)
    pub fn new() -> Result<Self, CacheError> {
        let cache_path = Self::default_cache_path()?;
        Self::with_path(cache_path)
    }

    /// Create a new disk cache at a specific path
    pub fn with_path<P: AsRef<Path>>(cache_path: P) -> Result<Self, CacheError> {
        let cache_path = cache_path.as_ref().to_path_buf();

        // Create parent directory if it doesn't exist
        if let Some(parent) = cache_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .map_err(|_| CacheError::DirectoryCreateFailed)?;
                info!("Created cache directory: {}", parent.display());
            }
        }

        let mut cache = Self {
            cache_path,
            entries: HashMap::new(),
        };

        // Try to load existing cache
        match cache.load() {
            Ok(count) => {
                info!("Loaded {} cached expert entries from disk", count);
            }
            Err(CacheError::NotFound) => {
                debug!("No existing cache file found, starting fresh");
            }
            Err(e) => {
                warn!("Failed to load cache, starting fresh: {}", e);
            }
        }

        Ok(cache)
    }

    /// Get the default cache path (~/.expertflow/cache.json)
    fn default_cache_path() -> Result<PathBuf, CacheError> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| CacheError::DirectoryCreateFailed)?;

        Ok(PathBuf::from(home)
            .join(".expertflow")
            .join("cache.json"))
    }

    /// Load cache from disk
    pub fn load(&mut self) -> Result<usize, CacheError> {
        if !self.cache_path.exists() {
            return Err(CacheError::NotFound);
        }

        let file = File::open(&self.cache_path)?;
        let reader = BufReader::new(file);
        let data: CacheData = serde_json::from_reader(reader)?;

        debug!(
            "Loading cache version {}, session timestamp {}",
            data.version, data.session_timestamp
        );

        self.entries.clear();
        for entry in data.entries {
            self.entries.insert(entry.expert_id, entry);
        }

        Ok(self.entries.len())
    }

    /// Save cache to disk
    pub fn save(&self) -> Result<(), CacheError> {
        let entries: Vec<CacheEntry> = self.entries.values().cloned().collect();

        let data = CacheData {
            version: 1,
            session_timestamp: Self::current_timestamp(),
            entries,
        };

        let file = File::create(&self.cache_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &data)?;

        info!(
            "Saved {} expert entries to cache: {}",
            self.entries.len(),
            self.cache_path.display()
        );

        Ok(())
    }

    /// Update or insert a cache entry
    pub fn update_entry(&mut self, entry: CacheEntry) {
        self.entries.insert(entry.expert_id, entry);
    }

    /// Get a cache entry
    pub fn get_entry(&self, expert_id: ExpertId) -> Option<&CacheEntry> {
        self.entries.get(&expert_id)
    }

    /// Get all entries sorted by temperature (hottest first)
    pub fn get_hot_experts(&self, limit: usize) -> Vec<CacheEntry> {
        let mut entries: Vec<CacheEntry> = self.entries.values().cloned().collect();
        entries.sort_by(|a, b| b.temperature.partial_cmp(&a.temperature).unwrap());
        entries.into_iter().take(limit).collect()
    }

    /// Get the number of cached entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get current Unix timestamp
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

impl Default for DiskCache {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            warn!("Failed to create default cache, using in-memory only");
            Self {
                cache_path: PathBuf::from("/tmp/expertflow_cache.json"),
                entries: HashMap::new(),
            }
        })
    }
}

impl Drop for DiskCache {
    fn drop(&mut self) {
        if let Err(e) = self.save() {
            warn!("Failed to save cache on drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cache_save_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache_path = temp_file.path().to_path_buf();

        // Create and populate cache
        let mut cache = DiskCache::with_path(&cache_path).unwrap();

        cache.update_entry(CacheEntry {
            expert_id: 0,
            temperature: 5.0,
            access_count: 10,
            last_session_time: 1234567890,
        });

        cache.update_entry(CacheEntry {
            expert_id: 1,
            temperature: 3.0,
            access_count: 5,
            last_session_time: 1234567890,
        });

        // Save
        cache.save().unwrap();

        // Load into new cache
        let mut cache2 = DiskCache::with_path(&cache_path).unwrap();
        assert_eq!(cache2.len(), 2);

        let entry0 = cache2.get_entry(0).unwrap();
        assert_eq!(entry0.temperature, 5.0);
        assert_eq!(entry0.access_count, 10);
    }

    #[test]
    fn test_hot_experts() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache_path = temp_file.path().to_path_buf();
        let mut cache = DiskCache::with_path(&cache_path).unwrap();

        // Add experts with different temperatures
        for i in 0..10 {
            cache.update_entry(CacheEntry {
                expert_id: i,
                temperature: i as f64,
                access_count: i as u64,
                last_session_time: 0,
            });
        }

        let hot = cache.get_hot_experts(3);
        assert_eq!(hot.len(), 3);
        assert_eq!(hot[0].expert_id, 9); // Hottest
        assert_eq!(hot[1].expert_id, 8);
        assert_eq!(hot[2].expert_id, 7);
    }
}
