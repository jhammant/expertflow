//! Simulation binary for FlashMoE without llama.cpp dependency.
//!
//! This creates a fake model file and simulates MoE inference to demonstrate
//! the scheduling algorithm in isolation. Useful for testing and benchmarking
//! the scheduler without needing actual model weights.

use clap::Parser;
use flashmoe::core::{
    evictor::TemperatureEvictor,
    memory::{ExpertRegion, MemoryManager},
    prefetcher::AsyncPrefetcher,
    scheduler::ExpertScheduler,
};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(name = "flashmoe-simulate")]
#[command(about = "Simulate FlashMoE scheduling without llama.cpp", long_about = None)]
struct Args {
    /// Number of MoE layers
    #[arg(long, default_value = "28")]
    layers: usize,

    /// Number of experts per layer
    #[arg(long, default_value = "256")]
    experts_per_layer: usize,

    /// Number of active experts per token (top-K routing)
    #[arg(long, default_value = "8")]
    active_per_token: usize,

    /// Number of inference steps (token generations)
    #[arg(long, default_value = "100")]
    steps: usize,

    /// Size of each expert in MB
    #[arg(long, default_value = "64")]
    expert_size_mb: usize,

    /// RAM budget in GB (for pinned experts)
    #[arg(long, default_value = "32")]
    ram_budget_gb: usize,

    /// Enable detailed logging
    #[arg(long, default_value = "false")]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Setup logging
    let level = if args.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
    tracing::subscriber::set_global_default(subscriber)?;

    println!("🚀 FlashMoE Simulation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Configuration:");
    println!("  Layers: {}", args.layers);
    println!("  Experts per layer: {}", args.experts_per_layer);
    println!("  Active per token: {}", args.active_per_token);
    println!("  Expert size: {} MB", args.expert_size_mb);
    println!("  Total experts: {}", args.layers * args.experts_per_layer);
    println!(
        "  Total model size: {:.2} GB",
        (args.layers * args.experts_per_layer * args.expert_size_mb) as f64 / 1024.0
    );
    println!("  RAM budget: {} GB", args.ram_budget_gb);
    println!("  Inference steps: {}", args.steps);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create fake model file
    info!("Creating temporary model file...");
    let total_size = args.layers * args.experts_per_layer * args.expert_size_mb * 1024 * 1024;
    let mut file = create_fake_model(total_size)?;

    // Build expert map
    info!("Building expert map...");
    let expert_map = build_expert_map(&args);

    // Create memory manager
    info!("Initializing memory manager...");
    let memory = Arc::new(MemoryManager::new(file.path(), expert_map)?);

    // Create scheduler
    info!("Initializing scheduler...");
    let ram_budget = args.ram_budget_gb * 1024 * 1024 * 1024;
    let scheduler = ExpertScheduler::new(memory, ram_budget, 2);

    // Run simulation
    info!("Starting inference simulation...");
    let stats = run_simulation(&scheduler, &args)?;

    // Print results
    print_results(&args, &stats);

    Ok(())
}

/// Create a fake model file with random data
fn create_fake_model(size: usize) -> anyhow::Result<NamedTempFile> {
    let mut file = NamedTempFile::new()?;

    // Write in chunks to avoid memory issues
    const CHUNK_SIZE: usize = 64 * 1024 * 1024; // 64MB chunks
    let chunk = vec![0u8; CHUNK_SIZE.min(size)];

    let mut written = 0;
    while written < size {
        let to_write = (size - written).min(CHUNK_SIZE);
        file.write_all(&chunk[..to_write])?;
        written += to_write;
    }

    file.flush()?;
    Ok(file)
}

/// Build expert memory map
fn build_expert_map(args: &Args) -> HashMap<usize, ExpertRegion> {
    let mut map = HashMap::new();
    let expert_size = args.expert_size_mb * 1024 * 1024;

    for layer in 0..args.layers {
        for expert in 0..args.experts_per_layer {
            let expert_id = layer * args.experts_per_layer + expert;
            let offset = expert_id * expert_size;

            map.insert(
                expert_id,
                ExpertRegion {
                    offset,
                    length: expert_size,
                },
            );
        }
    }

    map
}

#[derive(Debug, Default)]
struct SimulationStats {
    total_accesses: usize,
    cache_hits: usize,
    cache_misses: usize,
    prefetch_hits: usize,
    total_latency_ms: f64,
    layer_times: Vec<Duration>,
}

/// Run the inference simulation
fn run_simulation(scheduler: &ExpertScheduler, args: &Args) -> anyhow::Result<SimulationStats> {
    let mut rng = rand::thread_rng();
    let mut stats = SimulationStats::default();

    let pb = ProgressBar::new(args.steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} steps ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    for step in 0..args.steps {
        let step_start = Instant::now();

        // Process each layer
        for layer in 0..args.layers {
            let layer_start = Instant::now();

            // Simulate router prediction: select K random experts
            let selected_experts: Vec<usize> = (0..args.active_per_token)
                .map(|_| rng.gen_range(0..args.experts_per_layer))
                .collect();

            // Convert to global expert IDs
            let expert_ids: Vec<usize> = selected_experts
                .iter()
                .map(|&e| layer * args.experts_per_layer + e)
                .collect();

            // Router prediction phase - prefetch
            scheduler.on_router_prediction(layer, &expert_ids);

            // Expert computation phase - access
            for &expert_id in &expert_ids {
                let access_start = Instant::now();
                let state = scheduler.get_state(expert_id);

                match state {
                    flashmoe::core::scheduler::ExpertState::Loaded => {
                        stats.cache_hits += 1;
                    }
                    flashmoe::core::scheduler::ExpertState::Prefetching => {
                        stats.prefetch_hits += 1;
                        // Simulate blocking wait for prefetch
                        std::thread::sleep(Duration::from_micros(100));
                    }
                    flashmoe::core::scheduler::ExpertState::Evicted => {
                        stats.cache_misses += 1;
                        // Simulate blocking load from SSD
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }

                // Actually get the expert (triggers load if needed)
                let _ = scheduler.get_expert(expert_id);

                stats.total_accesses += 1;
                let access_time = access_start.elapsed();
                stats.total_latency_ms += access_time.as_secs_f64() * 1000.0;
            }

            let layer_time = layer_start.elapsed();
            stats.layer_times.push(layer_time);
        }

        // Run temperature decay every 10 steps
        if step % 10 == 0 {
            scheduler.tick();
        }

        pb.inc(1);
    }

    pb.finish_with_message("Simulation complete");

    Ok(stats)
}

/// Print simulation results
fn print_results(args: &Args, stats: &SimulationStats) {
    println!("\n📊 Simulation Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let cache_hit_rate = if stats.total_accesses > 0 {
        (stats.cache_hits as f64 / stats.total_accesses as f64) * 100.0
    } else {
        0.0
    };

    let prefetch_hit_rate = if stats.total_accesses > 0 {
        (stats.prefetch_hits as f64 / stats.total_accesses as f64) * 100.0
    } else {
        0.0
    };

    let cache_miss_rate = if stats.total_accesses > 0 {
        (stats.cache_misses as f64 / stats.total_accesses as f64) * 100.0
    } else {
        0.0
    };

    println!("Expert Access Statistics:");
    println!("  Total accesses: {}", stats.total_accesses);
    println!(
        "  Cache hits: {} ({:.2}%)",
        stats.cache_hits, cache_hit_rate
    );
    println!(
        "  Prefetch hits: {} ({:.2}%)",
        stats.prefetch_hits, prefetch_hit_rate
    );
    println!(
        "  Cache misses: {} ({:.2}%)",
        stats.cache_misses, cache_miss_rate
    );

    let avg_latency = if stats.total_accesses > 0 {
        stats.total_latency_ms / stats.total_accesses as f64
    } else {
        0.0
    };

    println!("\nPerformance:");
    println!("  Avg access latency: {:.3} ms", avg_latency);

    if !stats.layer_times.is_empty() {
        let avg_layer_time = stats.layer_times.iter().sum::<Duration>() / stats.layer_times.len() as u32;
        println!("  Avg layer time: {:.3} ms", avg_layer_time.as_secs_f64() * 1000.0);
    }

    let tokens_per_sec = if stats.total_latency_ms > 0.0 {
        (args.steps as f64) / (stats.total_latency_ms / 1000.0)
    } else {
        0.0
    };

    println!("  Throughput: {:.2} tokens/sec", tokens_per_sec);

    println!("\nScheduler Effectiveness:");
    let effective_cache_rate = cache_hit_rate + prefetch_hit_rate;
    println!("  Effective cache rate: {:.2}%", effective_cache_rate);

    if effective_cache_rate > 90.0 {
        println!("  Status: ✅ EXCELLENT - Scheduler is working very well");
    } else if effective_cache_rate > 70.0 {
        println!("  Status: ✅ GOOD - Scheduler is effective");
    } else if effective_cache_rate > 50.0 {
        println!("  Status: ⚠️  FAIR - Consider tuning parameters");
    } else {
        println!("  Status: ❌ POOR - Scheduler needs improvement");
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}
