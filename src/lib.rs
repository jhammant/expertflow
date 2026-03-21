//! # ExpertFlow
//!
//! Dynamic MoE expert streaming for Apple Silicon.
//!
//! ExpertFlow orchestrates expert computation across GPU, ANE, and CPU,
//! streaming cold experts from NVMe on demand while keeping hot experts
//! pinned in unified memory.

pub mod cache;
pub mod core;
pub mod compute;
pub mod model;
pub mod profiler;
pub mod quant;

#[cfg(feature = "llamacpp")]
pub mod bridge;

pub use cache::disk::{DiskCache, CacheEntry};

pub use core::{
    scheduler::ExpertScheduler,
    prefetcher::AsyncPrefetcher,
    evictor::TemperatureEvictor,
    memory::MemoryManager,
    budget::{MemoryBudget, MemoryBudgetStats},
};

pub use compute::router::RouterLookahead;

pub use model::{
    config::MoEConfig,
    gguf::GgufLoader,
    hybrid::{HybridConfig, LayerKind},
};

pub use profiler::{
    heatmap::ExpertHeatmap,
    bench::BenchmarkSuite,
};

pub use quant::adaptive::{AdaptiveQuantizer, QuantLevel};

/// Expert ID type
pub type ExpertId = usize;

/// Layer index type
pub type LayerIdx = usize;
