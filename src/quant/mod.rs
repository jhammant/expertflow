//! Adaptive quantization for MoE experts.
//!
//! Different quantization levels for hot vs cold experts to balance
//! memory usage and quality.

pub mod adaptive;

pub use adaptive::{AdaptiveQuantizer, QuantLevel};
