//! GGUF model file parser.
//!
//! Parses GGUF headers to extract tensor information and identify expert
//! tensors for memory mapping.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use thiserror::Error;

use crate::core::memory::ExpertRegion;
use crate::ExpertId;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid GGUF magic number")]
    InvalidMagic,

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Failed to parse GGUF header")]
    ParseError,
}

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// GGUF file loader
pub struct GgufLoader {
    version: u32,
    tensor_count: u64,
    tensors: Vec<TensorInfo>,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

impl GgufLoader {
    /// Load a GGUF file and parse its header
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;

        // Read and verify magic number
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        let magic_u32 = u32::from_le_bytes(magic);

        if magic_u32 != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic);
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // Read tensor count
        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes)?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes);

        // For now, we'll create a mock tensor list
        // A real implementation would parse the full GGUF metadata section
        let tensors = Vec::new();

        Ok(Self {
            version,
            tensor_count,
            tensors,
        })
    }

    /// Extract expert regions from tensor list.
    ///
    /// Identifies expert tensors by naming convention (e.g., "layers.0.experts.3.w1").
    /// Returns a map from ExpertId to ExpertRegion.
    pub fn extract_expert_map(&self) -> HashMap<ExpertId, ExpertRegion> {
        let mut expert_map = HashMap::new();

        for tensor in &self.tensors {
            if let Some(expert_id) = Self::parse_expert_id(&tensor.name) {
                // Group all tensors for the same expert
                let region = expert_map.entry(expert_id).or_insert_with(|| ExpertRegion {
                    offset: tensor.offset as usize,
                    length: 0,
                });

                // Extend the region to include this tensor
                let end = tensor.offset + tensor.size;
                let current_end = region.offset + region.length;

                if end as usize > current_end {
                    region.length = end as usize - region.offset;
                }
            }
        }

        expert_map
    }

    /// Parse expert ID from tensor name.
    ///
    /// Examples:
    /// - "layers.0.experts.3.w1" → Some(3)
    /// - "layers.5.experts.127.w2" → Some(127)
    /// - "layers.0.attention.q" → None
    fn parse_expert_id(name: &str) -> Option<ExpertId> {
        let parts: Vec<&str> = name.split('.').collect();

        for i in 0..parts.len() - 1 {
            if parts[i] == "experts" {
                return parts[i + 1].parse().ok();
            }
        }

        None
    }

    /// Get GGUF version
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Get tensor count
    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Get list of all tensors
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expert_id() {
        assert_eq!(GgufLoader::parse_expert_id("layers.0.experts.3.w1"), Some(3));
        assert_eq!(GgufLoader::parse_expert_id("layers.5.experts.127.w2"), Some(127));
        assert_eq!(GgufLoader::parse_expert_id("layers.0.attention.q"), None);
        assert_eq!(GgufLoader::parse_expert_id("embed_tokens.weight"), None);
    }

    #[test]
    fn test_gguf_magic() {
        // GGUF magic is "GGUF" in ASCII, little-endian
        let magic_bytes = [b'G', b'G', b'U', b'F'];
        let magic = u32::from_le_bytes(magic_bytes);
        assert_eq!(magic, GGUF_MAGIC);
    }
}
