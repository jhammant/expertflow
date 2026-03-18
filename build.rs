// Build script for compiling llama.cpp as a static library

fn main() {
    // Only compile llama.cpp if the llamacpp feature is enabled
    #[cfg(feature = "llamacpp")]
    build_llamacpp();
}

#[cfg(feature = "llamacpp")]
fn build_llamacpp() {
    use std::env;
    use std::path::PathBuf;

    let llamacpp_dir = PathBuf::from("vendor/llama.cpp");

    // Check if llama.cpp submodule exists
    if !llamacpp_dir.exists() {
        panic!("llama.cpp submodule not found. Run: git submodule update --init --recursive");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    println!("cargo:rerun-if-changed=vendor/llama.cpp");

    // Core GGML files
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++11")
        .include(llamacpp_dir.join("include"))
        .include(llamacpp_dir.join("ggml/include"))
        .include(llamacpp_dir.join("ggml/src"))
        .file(llamacpp_dir.join("ggml/src/ggml.c"))
        .file(llamacpp_dir.join("ggml/src/ggml-alloc.c"))
        .file(llamacpp_dir.join("ggml/src/ggml-backend.c"))
        .file(llamacpp_dir.join("ggml/src/ggml-quants.c"))
        .file(llamacpp_dir.join("src/llama.cpp"))
        .file(llamacpp_dir.join("src/llama-vocab.cpp"))
        .file(llamacpp_dir.join("src/llama-grammar.cpp"))
        .file(llamacpp_dir.join("src/llama-sampling.cpp"))
        .file(llamacpp_dir.join("common/common.cpp"))
        .warnings(false); // Suppress warnings from third-party code

    // Platform-specific configuration
    match target_os.as_str() {
        "macos" => {
            // Apple Silicon - use Metal and Accelerate
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Accelerate");

            build.define("GGML_USE_ACCELERATE", None);
            build.define("GGML_USE_METAL", None);

            // Add Metal backend files
            build.file(llamacpp_dir.join("ggml/src/ggml-metal.m"));
        }
        "linux" => {
            // Linux - use OpenBLAS or fall back to CPU
            if target_arch == "aarch64" {
                // ARM64 Linux - just use CPU with NEON
                build.define("GGML_USE_CPU", None);
            } else {
                // x86_64 Linux - try OpenBLAS
                if pkg_config::probe_library("openblas").is_ok() {
                    build.define("GGML_USE_OPENBLAS", None);
                    println!("cargo:rustc-link-lib=openblas");
                }
            }
        }
        _ => {
            // Default to CPU only
            build.define("GGML_USE_CPU", None);
        }
    }

    // Compile the library
    build.compile("llama");

    // Tell cargo to link against the standard C++ library
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Generate the output directory for the library
    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir);
}
