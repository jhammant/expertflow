//! Memory management using mmap and madvise.
//!
//! Zero-copy memory management for expert weights. Uses mmap to map model
//! files directly into memory, and madvise for prefetch/eviction hints.

use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

use crate::ExpertId;

#[cfg(target_os = "macos")]
const MADV_WILLNEED: libc::c_int = 3;
#[cfg(target_os = "macos")]
const MADV_DONTNEED: libc::c_int = 4;

#[cfg(target_os = "linux")]
const MADV_WILLNEED: libc::c_int = 3;
#[cfg(target_os = "linux")]
const MADV_DONTNEED: libc::c_int = 4;

/// Expert memory region descriptor
#[derive(Debug, Clone)]
pub struct ExpertRegion {
    pub offset: usize,
    pub length: usize,
}

/// Memory manager for expert weights
pub struct MemoryManager {
    mmap: Mmap,
    expert_map: HashMap<ExpertId, ExpertRegion>,
}

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Failed to open model file: {0}")]
    FileOpen(#[from] std::io::Error),

    #[error("Failed to mmap file: {0}")]
    MmapFailed(String),

    #[error("Expert {0} not found in memory map")]
    ExpertNotFound(ExpertId),

    #[error("madvise failed: {0}")]
    MadviseFailed(String),
}

impl MemoryManager {
    /// Create a new memory manager for a model file.
    ///
    /// The file is mmap'd into memory, but pages are not loaded until accessed.
    pub fn new<P: AsRef<Path>>(
        path: P,
        expert_map: HashMap<ExpertId, ExpertRegion>,
    ) -> Result<Self, MemoryError> {
        let file = File::open(path)?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| MemoryError::MmapFailed(e.to_string()))?
        };

        Ok(Self { mmap, expert_map })
    }

    /// Create a memory manager for testing with a pre-mapped mmap
    pub fn from_mmap(mmap: Mmap, expert_map: HashMap<ExpertId, ExpertRegion>) -> Self {
        Self { mmap, expert_map }
    }

    /// Get the memory slice for an expert.
    ///
    /// This does NOT trigger loading - just returns a reference to the mmap region.
    pub fn get_expert(&self, expert_id: ExpertId) -> Result<&[u8], MemoryError> {
        let region = self.expert_map
            .get(&expert_id)
            .ok_or(MemoryError::ExpertNotFound(expert_id))?;

        let end = region.offset + region.length;
        if end > self.mmap.len() {
            return Err(MemoryError::ExpertNotFound(expert_id));
        }

        Ok(&self.mmap[region.offset..end])
    }

    /// Pin an expert in memory (madvise WILLNEED).
    ///
    /// Hints to the kernel that this memory will be accessed soon.
    /// Triggers async page-in from disk.
    pub fn pin_expert(&self, expert_id: ExpertId) -> Result<(), MemoryError> {
        let region = self.expert_map
            .get(&expert_id)
            .ok_or(MemoryError::ExpertNotFound(expert_id))?;

        let ptr = unsafe { self.mmap.as_ptr().add(region.offset) };

        let result = unsafe {
            libc::madvise(
                ptr as *mut libc::c_void,
                region.length,
                MADV_WILLNEED,
            )
        };

        if result != 0 {
            let err = std::io::Error::last_os_error();
            return Err(MemoryError::MadviseFailed(err.to_string()));
        }

        Ok(())
    }

    /// Evict an expert from memory (madvise DONTNEED).
    ///
    /// Hints to the kernel that this memory is no longer needed.
    /// Pages may be reclaimed by the OS.
    pub fn evict_expert(&self, expert_id: ExpertId) -> Result<(), MemoryError> {
        let region = self.expert_map
            .get(&expert_id)
            .ok_or(MemoryError::ExpertNotFound(expert_id))?;

        let ptr = unsafe { self.mmap.as_ptr().add(region.offset) };

        let result = unsafe {
            libc::madvise(
                ptr as *mut libc::c_void,
                region.length,
                MADV_DONTNEED,
            )
        };

        if result != 0 {
            let err = std::io::Error::last_os_error();
            return Err(MemoryError::MadviseFailed(err.to_string()));
        }

        Ok(())
    }

    /// Get current memory pressure (0.0 = no pressure, 1.0 = critical).
    ///
    /// On macOS, reads from vm_stat. On Linux, reads from /proc/meminfo.
    pub fn memory_pressure(&self) -> f64 {
        #[cfg(target_os = "macos")]
        {
            // Parse vm_stat output
            // For now, return a mock value
            // TODO: Implement proper vm_stat parsing
            0.5
        }

        #[cfg(target_os = "linux")]
        {
            // Parse /proc/meminfo
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total = 0u64;
                let mut available = 0u64;

                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        total = line.split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                    } else if line.starts_with("MemAvailable:") {
                        available = line.split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                    }
                }

                if total > 0 {
                    return 1.0 - (available as f64 / total as f64);
                }
            }
            0.5
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            0.5
        }
    }

    /// Get the total size of the mmap
    pub fn total_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get number of experts in the map
    pub fn num_experts(&self) -> usize {
        self.expert_map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_memory_manager_creation() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test data for expert weights").unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        expert_map.insert(0, ExpertRegion { offset: 0, length: 10 });
        expert_map.insert(1, ExpertRegion { offset: 10, length: 18 });

        let manager = MemoryManager::new(file.path(), expert_map).unwrap();
        assert_eq!(manager.num_experts(), 2);
    }

    #[test]
    fn test_get_expert() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"AAAAAAAAAA").unwrap();
        file.write_all(b"BBBBBBBBBB").unwrap();
        file.flush().unwrap();

        let mut expert_map = HashMap::new();
        expert_map.insert(0, ExpertRegion { offset: 0, length: 10 });
        expert_map.insert(1, ExpertRegion { offset: 10, length: 10 });

        let manager = MemoryManager::new(file.path(), expert_map).unwrap();

        let expert0 = manager.get_expert(0).unwrap();
        assert_eq!(expert0, b"AAAAAAAAAA");

        let expert1 = manager.get_expert(1).unwrap();
        assert_eq!(expert1, b"BBBBBBBBBB");
    }
}
