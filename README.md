# infercraft

[![Crates.io](https://img.shields.io/crates/v/infercraft.svg)](https://crates.io/crates/infercraft)
[![Docs.rs](https://docs.rs/infercraft/badge.svg)](https://docs.rs/infercraft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance local LLM inference engine with zero-copy mmap, layer-by-layer weight streaming, and hardware calibration for Rust.

Built and maintained by **[Kreive Tech](https://kreive.com.br)**.

---

## Features

- **Layer-by-Layer Weight Streaming**: Radically minimizes RAM footprint by keeping only base embeddings, normalization, and a single active transformer block in memory during forward passes via zero-copy `mmap`.
- **Hardware-Aware Auto-Calibration**: Introspects dedicated GPUs (NVIDIA VRAM) and integrated GPUs with unified shared memory (Intel Iris, AMD Radeon, Apple Silicon UMA) to calculate optimal GPU offload layers (`gpu_layers`).
- **Multi-Architecture Support**: Native support for quantized GGUF models across Llama (3/3.1/3.2), Qwen (2/2.5), Gemma (2/3), and Phi (3/4).
- **Token Streaming & Interruption**: Non-blocking streaming generation with atomic cancellation tokens and sliding context windows.
- **Resource Lifecycle Watchdog**: Automatic model memory unloading based on configurable idle timeouts to return 100% of RAM/VRAM to the operating system.

---

## Installation

Add `infercraft` to your `Cargo.toml`:

```toml
[dependencies]
infercraft = "0.1.0"
```

### Hardware Acceleration (Optional Features)

```toml
# NVIDIA CUDA acceleration
infercraft = { version = "0.1.0", features = ["cuda"] }

# Apple Metal acceleration
infercraft = { version = "0.1.0", features = ["metal"] }
```

---

## Quick Examples

### 1. Streaming Local LLM Inference

```rust
use infercraft::{load_model, run_inference_streaming};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load GGUF model with 4 threads
    load_model("smollm2-135m-q4", 4).await?;

    let prompt = "Explain quantum computing in one short sentence:";

    // Stream tokens in real time
    let (complete_text, token_count) = run_inference_streaming(
        prompt,
        64,     // max_tokens
        0.7,    // temperature
        None,   // top_p
        1.1,    // repeat_penalty
        2048,   // max_context
        None,   // grammar
        |token| print!("{}", token),
    ).await?;

    println!("\nGenerated {} tokens.", token_count);
    Ok(())
}
```

---

### 2. Hardware Profiling & UMA Detection

```rust
use infercraft::HardwareDetector;

fn main() {
    let detector = HardwareDetector::new();
    if let Ok(info) = detector.detect_system() {
        println!("CPU Cores: {}", info.cpu_cores);
        println!("RAM: {:.1} GB", info.ram_total_mb as f64 / 1024.0);
        println!("GPU Detected: {}", info.gpu.detected);
        println!("Integrated / UMA: {}", info.gpu.is_integrated);
        println!("Recommended GPU Layers: {}", info.gpu.gpu_layers);
        println!("System Diagnosis: {}", info.recommendation);
    }
}
```

---

### 3. Custom Model Directory & Idle Auto-Unload

```rust
use infercraft::{load_model, set_models_dir, set_unload_timeout_minutes, unload_model};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Custom storage directory for .gguf files
    set_models_dir(PathBuf::from("/opt/ai/models"));

    // Automatically unload weights from RAM after 10 minutes of inactivity
    set_unload_timeout_minutes(10);

    load_model("qwen-2.5-1.5b-q8", 4).await?;

    // ... run inference ...

    // Manually unload when finished
    unload_model().await;

    Ok(())
}
```

---

## Sponsor & Support

Developed by **[Kreive Tech](https://kreive.com.br)**.

For questions, enterprise agent support, or contributions, reach out to `contact@kreive.com.br`.

---

## License

MIT © [Kreive Tech](https://kreive.com.br)
