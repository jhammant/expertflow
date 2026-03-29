//! Benchmarking suite for ExpertFlow.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::cache::expert_loader::{ExpertMmapLoader, LoaderError};
use crate::model::gguf::GgufLoader;

/// Benchmark results for expert loading
#[derive(Debug, Clone)]
pub struct ExpertBenchmarkResult {
    pub name: String,
    pub expert_id: usize,
    pub load_time: Duration,
    pub bytes_loaded: usize,
}

/// Expert loader benchmark suite
pub struct ExpertBenchmarkSuite {
    results: Vec<ExpertBenchmarkResult>,
}

impl ExpertBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Benchmark loading a single expert from an mmap loader
    pub fn bench_expert_load(
        &mut self,
        name: &str,
        loader: &mut ExpertMmapLoader,
        expert_id: usize,
    ) -> Result<(), LoaderError> {
        // Warmup
        let _ = loader.load_expert(expert_id, false);

        // Actual benchmark
        let start = Instant::now();
        let expert_data = loader.load_expert(expert_id, false)?;
        let duration = start.elapsed();

        let bytes_loaded = expert_data.len();

        self.results.push(ExpertBenchmarkResult {
            name: name.to_string(),
            expert_id,
            load_time: duration,
            bytes_loaded,
        });

        Ok(())
    }

    /// Benchmark loading multiple experts sequentially
    pub fn bench_multiple_experts(
        &mut self,
        name: &str,
        loader: &mut ExpertMmapLoader,
        expert_ids: &[usize],
    ) -> Result<(), LoaderError> {
        for &expert_id in expert_ids {
            self.bench_expert_load(&format!("{}_{}", name, expert_id), loader, expert_id)?;
        }
        Ok(())
    }

    /// Benchmark cache hit performance
    pub fn bench_cache_hits(
        &mut self,
        name: &str,
        loader: &mut ExpertMmapLoader,
        expert_id: usize,
        iterations: usize,
    ) -> Result<(), LoaderError> {
        // Pre-load to ensure cache hit
        let _ = loader.load_expert(expert_id, true);

        for i in 0..iterations {
            let start = Instant::now();
            let expert_data = loader.load_expert(expert_id, false)?;
            let duration = start.elapsed();

            self.results.push(ExpertBenchmarkResult {
                name: format!("{}_{}", name, i),
                expert_id,
                load_time: duration,
                bytes_loaded: expert_data.len(),
            });
        }

        Ok(())
    }

    /// Get all results
    pub fn results(&self) -> &[ExpertBenchmarkResult] {
        &self.results
    }

    /// Get statistics summary
    pub fn summary(&self) -> BenchmarkSummary {
        if self.results.is_empty() {
            return BenchmarkSummary {
                total_loads: 0,
                total_bytes: 0,
                min_time_ns: 0,
                max_time_ns: 0,
                avg_time_ns: 0,
                total_time_ns: 0,
            };
        }

        let total_loads = self.results.len();
        let total_bytes: usize = self.results.iter().map(|r| r.bytes_loaded).sum();
        let times_ns: Vec<u64> = self.results.iter().map(|r| r.load_time.as_nanos() as u64).collect();
        let total_time_ns: u64 = times_ns.iter().sum();
        let avg_time_ns = total_time_ns / total_loads as u64;
        let min_time_ns = *times_ns.iter().min().unwrap_or(&0);
        let max_time_ns = *times_ns.iter().max().unwrap_or(&0);

        BenchmarkSummary {
            total_loads,
            total_bytes,
            min_time_ns,
            max_time_ns,
            avg_time_ns,
            total_time_ns,
        }
    }
}

impl Default for ExpertBenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark summary statistics
#[derive(Debug, Clone)]
#[derive(serde::Serialize)]
pub struct BenchmarkSummary {
    pub total_loads: usize,
    pub total_bytes: usize,
    pub min_time_ns: u64,
    pub max_time_ns: u64,
    pub avg_time_ns: u64,
    pub total_time_ns: u64,
}

/// Model file benchmark
pub struct ModelBenchmark {
    pub loader: Option<ExpertMmapLoader>,
}

impl ModelBenchmark {
    /// Create a new model benchmark
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let loader = ExpertMmapLoader::load(path)?;
        
        Ok(Self { loader: Some(loader) })
    }

    /// Run benchmark suite
    pub fn run_benchmark(
        &mut self,
        expert_ids: &[usize],
    ) -> Result<ExpertBenchmarkSuite, LoaderError> {
        let mut suite = ExpertBenchmarkSuite::new();
        
        if !expert_ids.is_empty() {
            let loader = self.loader.as_mut().unwrap();
            suite.bench_multiple_experts("benchmark", loader, expert_ids)?;
        }

        Ok(suite)
    }

    /// Get model info
    pub fn model_info(&self) -> Option<&GgufLoader> {
        self.loader.as_ref().map(|l| l.model_info())
    }

    /// Get stats
    pub fn stats(&self) -> Option<crate::cache::expert_loader::LoaderStats> {
        self.loader.as_ref().map(|l| l.stats())
    }
}
