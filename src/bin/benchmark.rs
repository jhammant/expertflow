//! Benchmark expert loading times from GGUF models.

use clap::Parser;
use indicatif::ProgressBar;

fn main() {
    let args = Args::parse();

    // Find model
    let model_path = find_model(args.model);
    
    if !model_path.exists() {
        eprintln!("Model file not found: {}", model_path.display());
        std::process::exit(1);
    }

    println!("Benchmarking expert loading from: {}", model_path.display());
    
    // Load the model
    let mut benchmark = match expertflow::profiler::bench::ModelBenchmark::load(&model_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    // Parse expert IDs
    let expert_ids: Vec<usize> = args.experts
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    if expert_ids.is_empty() {
        eprintln!("No valid expert IDs provided");
        std::process::exit(1);
    }

    println!("Benchmarking experts: {:?}", expert_ids);

    // Run benchmark
    let pb = ProgressBar::new((expert_ids.len() * args.iterations) as u64);
    pb.set_message("Loading experts...");
    
    let mut suite = expertflow::profiler::bench::ExpertBenchmarkSuite::new();
    
    // Warmup
    if expert_ids.len() > 0 {
        let _ = benchmark.loader.as_mut().unwrap().load_expert(expert_ids[0], false);
    }
    
    for &expert_id in &expert_ids {
        if let Some(loader) = benchmark.loader.as_mut() {
            if let Err(e) = suite.bench_cache_hits(&format!("expert_{}", expert_id), 
                                                      loader, 
                                                      expert_id, 
                                                      args.iterations) {
                eprintln!("Failed to benchmark expert {}: {}", expert_id, e);
            }
        } else {
            eprintln!("No loader available");
        }
        pb.inc(1);
    }
    
    pb.finish_with_message("Done!");

    // Print results
    print_results(&suite, &args.output);
}

#[derive(Parser)]
#[command(name = "benchmark")]
#[command(about = "Benchmark expert loading performance from GGUF models")]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: Option<std::path::PathBuf>,

    /// List of experts to benchmark (comma-separated)
    #[arg(long, default_value = "0,1,2")]
    experts: String,

    /// Number of iterations per expert
    #[arg(long, default_value = "3")]
    iterations: usize,

    /// Output format (text or json)
    #[arg(short, long, default_value = "text")]
    output: String,
}

fn find_model(cli_path: Option<std::path::PathBuf>) -> std::path::PathBuf {
    if let Some(path) = cli_path {
        return path;
    }

    // Check common locations
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let lmstudio_path = format!("{}/.lmstudio/models", home);
    
    if std::path::Path::new(&lmstudio_path).exists() {
        // Find first GGUF file with experts
        if let Ok(dirs) = std::fs::read_dir(&lmstudio_path) {
            for entry in dirs.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(&path) {
                        for entry in entries.flatten() {
                            let file_path = entry.path();
                            if let Some(ext) = file_path.extension() {
                                if ext == "gguf" && is_moe_model(&file_path) {
                                    return file_path;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::path::PathBuf::from("model.gguf")
}

fn is_moe_model(path: &std::path::Path) -> bool {
    // Try to load and check for experts
    if let Ok(loader) = expertflow::cache::expert_loader::ExpertMmapLoader::load(path) {
        let stats = loader.stats();
        stats.num_cached_experts > 0
    } else {
        false
    }
}

fn print_results(suite: &expertflow::profiler::bench::ExpertBenchmarkSuite, output_format: &str) {
    let summary = suite.summary();

    if output_format == "json" {
        println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    } else {
        println!("\n=== Expert Loading Benchmark Results ===\n");
        println!("Total experts loaded: {}", summary.total_loads);
        println!("Total bytes loaded: {} MB", summary.total_bytes / (1024 * 1024));
        println!("\nTiming Statistics:");
        println!("  Min load time: {} µs", summary.min_time_ns as f64 / 1000.0);
        println!("  Max load time: {} µs", summary.max_time_ns as f64 / 1000.0);
        println!("  Avg load time: {} µs", summary.avg_time_ns as f64 / 1000.0);
        println!("  Total time:    {} µs", summary.total_time_ns as f64 / 1000.0);
        
        println!("\nLoad times per expert:");
        for result in suite.results() {
            let time_us = result.load_time.as_nanos() as f64 / 1000.0;
            let mb = result.bytes_loaded as f64 / (1024.0 * 1024.0);
            println!("  Expert {}: {} µs ({} MB)", 
                     result.expert_id, 
                     format!("{:.2}", time_us),
                     format!("{:.3}", mb));
        }
    }
}
