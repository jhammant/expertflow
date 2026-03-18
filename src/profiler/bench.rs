//! Benchmarking suite for FlashMoE.

use std::time::{Duration, Instant};

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration: Duration,
    pub tokens_per_second: f64,
}

/// Benchmark suite
pub struct BenchmarkSuite {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Run a benchmark
    pub fn bench<F>(&mut self, name: &str, num_tokens: usize, f: F)
    where
        F: FnOnce(),
    {
        let start = Instant::now();
        f();
        let duration = start.elapsed();

        let tokens_per_second = if duration.as_secs_f64() > 0.0 {
            num_tokens as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        self.results.push(BenchmarkResult {
            name: name.to_string(),
            duration,
            tokens_per_second,
        });
    }

    /// Get all results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}
