//! Persistent expert cache and dynamic loading.
//!
//! Serializes expert activation patterns to disk so hot experts from previous
//! sessions can be pre-loaded on startup.

pub mod disk;
pub mod expert_loader;

pub use disk::{DiskCache, CacheEntry};
pub use expert_loader::{ExpertMmapLoader, ExpertPrefetchLoader, CachedExpert, LoaderStats};
