//! FlashMoE CLI - Dynamic MoE expert streaming for Apple Silicon

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "flashmoe")]
#[command(about = "Dynamic MoE expert streaming for Apple Silicon", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference with dynamic expert streaming
    #[cfg(feature = "llamacpp")]
    Run {
        /// Path to the GGUF model file
        #[arg(long)]
        model: PathBuf,

        /// RAM budget in GB
        #[arg(long, default_value = "32")]
        ram_budget: usize,

        /// Number of threads
        #[arg(long, default_value = "8")]
        threads: usize,

        /// Number of GPU layers
        #[arg(long, default_value = "0")]
        n_gpu_layers: i32,

        /// Context size
        #[arg(long, default_value = "4096")]
        n_ctx: u32,

        /// Prompt to generate from
        #[arg(long)]
        prompt: Option<String>,
    },

    /// Profile expert activation patterns
    #[cfg(feature = "llamacpp")]
    Profile {
        /// Path to the GGUF model file
        #[arg(long)]
        model: PathBuf,

        /// Number of samples to profile
        #[arg(long, default_value = "100")]
        samples: usize,

        /// Output file for heatmap
        #[arg(long, default_value = "expert_heatmap.json")]
        output: PathBuf,
    },

    /// Benchmark FlashMoE vs vanilla mmap
    #[cfg(feature = "llamacpp")]
    Bench {
        /// Path to the GGUF model file
        #[arg(long)]
        model: PathBuf,

        /// Number of tokens to generate
        #[arg(long, default_value = "100")]
        tokens: usize,
    },

    /// Run simulation without llama.cpp (use simulate binary instead)
    Simulate {
        /// Number of MoE layers
        #[arg(long, default_value = "28")]
        layers: usize,

        /// Number of experts per layer
        #[arg(long, default_value = "256")]
        experts: usize,

        /// Number of active experts per token
        #[arg(long, default_value = "8")]
        active: usize,

        /// Number of inference steps
        #[arg(long, default_value = "100")]
        steps: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        #[cfg(feature = "llamacpp")]
        Commands::Run {
            model,
            ram_budget,
            threads,
            n_gpu_layers,
            n_ctx,
            prompt,
        } => {
            println!("🚀 FlashMoE - Running inference");
            println!("Model: {:?}", model);
            println!("RAM budget: {} GB", ram_budget);
            println!("Threads: {}", threads);
            println!("GPU layers: {}", n_gpu_layers);
            println!("Context size: {}", n_ctx);

            if let Some(p) = prompt {
                println!("Prompt: {}", p);
            }

            // TODO: Implement actual inference with llama.cpp
            eprintln!("❌ Run command not yet implemented");
            eprintln!("   This requires the llamacpp feature and full integration");
            std::process::exit(1);
        }

        #[cfg(feature = "llamacpp")]
        Commands::Profile {
            model,
            samples,
            output,
        } => {
            println!("📊 FlashMoE - Profiling expert activation patterns");
            println!("Model: {:?}", model);
            println!("Samples: {}", samples);
            println!("Output: {:?}", output);

            // TODO: Implement profiling
            eprintln!("❌ Profile command not yet implemented");
            std::process::exit(1);
        }

        #[cfg(feature = "llamacpp")]
        Commands::Bench { model, tokens } => {
            println!("⚡ FlashMoE - Benchmarking");
            println!("Model: {:?}", model);
            println!("Tokens: {}", tokens);

            // TODO: Implement benchmarking
            eprintln!("❌ Bench command not yet implemented");
            std::process::exit(1);
        }

        Commands::Simulate {
            layers,
            experts,
            active,
            steps,
        } => {
            println!("🎮 FlashMoE - Simulation mode");
            println!();
            println!("For simulation, please use the dedicated binary:");
            println!();
            println!("  cargo run --release --bin simulate -- \\");
            println!("    --layers {} \\", layers);
            println!("    --experts-per-layer {} \\", experts);
            println!("    --active-per-token {} \\", active);
            println!("    --steps {}", steps);
            println!();
            Ok(())
        }

        #[allow(unreachable_patterns)]
        _ => {
            eprintln!("❌ This command requires the 'llamacpp' feature");
            eprintln!("   Rebuild with: cargo build --release --features llamacpp");
            std::process::exit(1);
        }
    }
}
