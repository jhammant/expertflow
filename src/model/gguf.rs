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
    size_bytes: u64,
}

impl TensorInfo {
    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Set tensor size
    pub fn set_size(&mut self, bytes: u64) {
        self.size_bytes = bytes;
    }
}

/// GGML tensor types for size calculations
#[derive(Debug, Clone, Copy)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    BF16 = 30,
}

impl GgmlType {
    /// Get block size in bytes
    pub fn block_size(&self) -> u64 {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 2 + 32 * 4 / 16,
            GgmlType::Q8_0 => 2 + 32,
            GgmlType::BF16 => 2,
        }
    }

    /// Get elements per block
    pub fn elements_per_block(&self) -> u64 {
        match self {
            GgmlType::F32 => 1,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 32,
            GgmlType::Q8_0 => 32,
            GgmlType::BF16 => 1,
        }
    }

    /// Calculate size from dimensions
    pub fn element_count(&self, dims: &[u64]) -> u64 {
        let mut size: u64 = 1;
        for &dim in dims {
            size *= dim;
        }
        let elements_per_block = self.elements_per_block();
        if elements_per_block > 0 {
            (size + elements_per_block - 1) / elements_per_block * self.block_size()
        } else {
            size
        }
    }
}

/// Metadata value types for GGUF parsing
#[derive(Debug, Clone, Copy)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GgufValueType::Uint8),
            1 => Ok(GgufValueType::Int8),
            2 => Ok(GgufValueType::Uint16),
            3 => Ok(GgufValueType::Int16),
            4 => Ok(GgufValueType::Uint32),
            5 => Ok(GgufValueType::Int32),
            6 => Ok(GgufValueType::Float32),
            7 => Ok(GgufValueType::Bool),
            8 => Ok(GgufValueType::String),
            9 => Ok(GgufValueType::Array),
            10 => Ok(GgufValueType::Uint64),
            11 => Ok(GgufValueType::Int64),
            12 => Ok(GgufValueType::Float64),
            _ => Err(()),
        }
    }
}

/// Metadata key-value pair
#[derive(Debug, Clone)]
pub struct MetadataKV {
    pub key: String,
    pub value_type: GgufValueType,
    pub value: MetadataValue,
}

/// Metadata values
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
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

        // Parse tensors
        let mut tensors = Vec::new();
        
        for _ in 0..tensor_count {
            // Read tensor name length
            let mut name_len_bytes = [0u8; 8];
            file.read_exact(&mut name_len_bytes)?;
            let name_len = u64::from_le_bytes(name_len_bytes) as usize;

            // Read tensor name
            let mut name = vec![0u8; name_len];
            file.read_exact(&mut name)?;
            let name = String::from_utf8_lossy(&name).to_string();

            // Read n_dimensions
            let mut n_dims_bytes = [0u8; 4];
            file.read_exact(&mut n_dims_bytes)?;
            let n_dims = u32::from_le_bytes(n_dims_bytes);

            // Read dimensions
            let mut dims = vec![0u64; n_dims as usize];
            for i in 0..n_dims as usize {
                let mut dim_bytes = [0u8; 8];
                file.read_exact(&mut dim_bytes)?;
                dims[i] = u64::from_le_bytes(dim_bytes);
            }

            // Read tensor type
            let mut tensor_type_bytes = [0u8; 4];
            file.read_exact(&mut tensor_type_bytes)?;
            let tensor_type_val = u32::from_le_bytes(tensor_type_bytes);
            
            // Map type to GgmlType
            let ggml_type = match tensor_type_val {
                0 => GgmlType::F32,
                1 => GgmlType::F16,
                2 => GgmlType::Q4_0,
                8 => GgmlType::Q8_0,
                30 => GgmlType::BF16,
                _ => GgmlType::F32, // Default
            };

            // Calculate tensor size
            let size = ggml_type.element_count(&dims);

            // Read tensor offset (align to 32 bytes)
            let mut offset_bytes = [0u8; 8];
            file.read_exact(&mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);

            // Skip tensor data (we'll mmap later)
            // In a real implementation, you'd seek past the data

            let mut tensor = TensorInfo {
                name,
                offset,
                size_bytes: 0, // Will calculate later
            };
            tensor.set_size(size);
            
            tensors.push(tensor);
        }

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
                let end = (tensor.offset + tensor.size_bytes()) as usize;
                let current_end = region.offset + region.length;

                if end > current_end {
                    region.length = end - region.offset;
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
    pub fn parse_expert_id(name: &str) -> Option<ExpertId> {
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

    #[test]
    fn test_ggml_type_block_size() {
        assert_eq!(GgmlType::F32.block_size(), 4);
        assert_eq!(GgmlType::Q4_0.block_size(), 10);
        assert_eq!(GgmlType::Q8_0.block_size(), 34);
    }

    #[test]
    fn test_ggml_type_elements_per_block() {
        assert_eq!(GgmlType::F32.elements_per_block(), 1);
        assert_eq!(GgmlType::Q4_0.elements_per_block(), 32);
        assert_eq!(GgmlType::Q8_0.elements_per_block(), 32);
    }
}
