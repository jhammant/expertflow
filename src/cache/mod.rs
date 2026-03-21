//! Persistent expert cache for warm restarts.
//!
//! Serializes expert activation patterns to disk so hot experts from previous
//! sessions can be pre-loaded on startup.

pub mod disk;

pub use disk::{DiskCache, CacheEntry};
