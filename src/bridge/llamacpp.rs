//! FFI bindings to llama.cpp for model loading and inference.
//!
//! This module provides safe Rust wrappers around the unsafe C FFI
//! to llama.cpp's core functionality.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::path::Path;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlamaError {
    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),

    #[error("Failed to create context")]
    ContextCreateFailed,

    #[error("Failed to decode tokens")]
    DecodeFailed,

    #[error("Invalid path: {0}")]
    InvalidPath(String),
}

pub type Result<T> = std::result::Result<T, LlamaError>;

// Opaque types from llama.cpp
#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context_params {
    pub seed: u32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
}

#[repr(C)]
pub struct llama_model_params {
    pub n_gpu_layers: c_int,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

// FFI declarations
extern "C" {
    fn llama_backend_init();
    fn llama_backend_free();

    fn llama_model_default_params() -> llama_model_params;
    fn llama_context_default_params() -> llama_context_params;

    fn llama_load_model_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;

    fn llama_free_model(model: *mut llama_model);

    fn llama_new_context_with_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;

    fn llama_free(ctx: *mut llama_context);

    fn llama_decode(
        ctx: *mut llama_context,
        batch: *const c_void,
    ) -> c_int;

    fn llama_token_to_piece(
        model: *const llama_model,
        token: c_int,
        buf: *mut c_char,
        length: c_int,
    ) -> c_int;
}

/// Safe wrapper around llama_model
pub struct LlamaModel {
    ptr: *mut llama_model,
}

impl LlamaModel {
    /// Load a model from a GGUF file
    pub fn load<P: AsRef<Path>>(path: P, n_gpu_layers: i32) -> Result<Self> {
        // Initialize backend (idempotent)
        unsafe { llama_backend_init() };

        let path_str = path.as_ref()
            .to_str()
            .ok_or_else(|| LlamaError::InvalidPath(format!("{:?}", path.as_ref())))?;

        let c_path = CString::new(path_str)
            .map_err(|_| LlamaError::InvalidPath(path_str.to_string()))?;

        let mut params = unsafe { llama_model_default_params() };
        params.n_gpu_layers = n_gpu_layers;
        params.use_mmap = true;
        params.use_mlock = false;

        let ptr = unsafe {
            llama_load_model_from_file(c_path.as_ptr(), params)
        };

        if ptr.is_null() {
            return Err(LlamaError::ModelLoadFailed(path_str.to_string()));
        }

        Ok(LlamaModel { ptr })
    }

    /// Get the raw pointer (for passing to context creation)
    pub fn as_ptr(&self) -> *mut llama_model {
        self.ptr
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe {
            llama_free_model(self.ptr);
        }
    }
}

unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

/// Safe wrapper around llama_context
pub struct LlamaContext {
    ptr: *mut llama_context,
}

impl LlamaContext {
    /// Create a new context for the given model
    pub fn new(model: &LlamaModel, n_ctx: u32, n_threads: u32) -> Result<Self> {
        let mut params = unsafe { llama_context_default_params() };
        params.n_ctx = n_ctx;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.n_threads = n_threads;
        params.n_threads_batch = n_threads;
        params.seed = 1234;

        let ptr = unsafe {
            llama_new_context_with_model(model.as_ptr(), params)
        };

        if ptr.is_null() {
            return Err(LlamaError::ContextCreateFailed);
        }

        Ok(LlamaContext { ptr })
    }

    /// Decode a batch of tokens
    pub fn decode(&mut self, batch: *const c_void) -> Result<()> {
        let ret = unsafe { llama_decode(self.ptr, batch) };

        if ret != 0 {
            return Err(LlamaError::DecodeFailed);
        }

        Ok(())
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut llama_context {
        self.ptr
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.ptr);
        }
    }
}

unsafe impl Send for LlamaContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires actual GGUF model file
    fn test_model_load() {
        let result = LlamaModel::load("/path/to/model.gguf", 0);
        assert!(result.is_ok());
    }
}
