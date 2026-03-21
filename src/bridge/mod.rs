//! FFI bridge to llama.cpp for MoE expert loading and routing.

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "llamacpp")]
pub mod expert_hook;

#[cfg(feature = "llamacpp")]
pub mod router_hook;

#[cfg(feature = "llamacpp")]
pub use llamacpp::{LlamaModel, LlamaContext};

#[cfg(feature = "llamacpp")]
pub use expert_hook::ExpertLoadHook;

#[cfg(feature = "llamacpp")]
pub use router_hook::RouterPredictionHook;

// MLX backend (Apple Silicon native)
#[cfg(feature = "mlx")]
pub mod mlx;

#[cfg(feature = "mlx")]
pub use mlx::{MlxModel, MlxContext};
