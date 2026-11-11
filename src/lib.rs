//! # infercraft
//!
//! High-performance local LLM inference engine with zero-copy mmap,
//! layer-by-layer weight streaming, and hardware calibration for Rust.
//!
//! Built and maintained by [Kreive Tech](https://kreive.com.br).

pub mod chat_template;
pub mod device;
pub mod engine;
pub mod error;
pub mod grammar;
pub mod hardware;
pub mod loader;
pub mod response_payload;
pub mod streaming_llama;
pub mod types;

pub use error::{InferError, UnevenError};
pub use hardware::{GpuInfo, GpuKind, HardwareDetector, HardwareInfo};
pub use loader::{AnyModel, LlamaState, ModelBackend};
pub use types::GenerateParams;

use once_cell::sync::{Lazy, OnceCell};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;

static LAST_ACTIVITY: Lazy<std::sync::Mutex<Option<Instant>>> =
    Lazy::new(|| std::sync::Mutex::new(None));

static UNLOAD_TASK_SPAWNED: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new(false));

static UNLOAD_TIMEOUT_SECS: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(300));

static DEFAULT_UNLOAD_TIMEOUT_MINS: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(5));

static MODELS_DIR: Lazy<std::sync::RwLock<Option<PathBuf>>> =
    Lazy::new(|| std::sync::RwLock::new(None));

/// Configures the directory where GGUF model files are stored.
pub fn set_models_dir(dir: PathBuf) {
    if let Ok(mut g) = MODELS_DIR.write() {
        *g = Some(dir);
    }
}

/// Retrieves the configured or default models directory.
pub fn get_models_dir() -> PathBuf {
    if let Ok(g) = MODELS_DIR.read() {
        if let Some(ref d) = *g {
            return d.clone();
        }
    }
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let uneven_models = PathBuf::from(&home).join(".uneven").join("bin").join("models");
        if uneven_models.exists() {
            return uneven_models;
        }
        PathBuf::from(home).join(".cache").join("infercraft").join("models")
    } else {
        PathBuf::from("models")
    }
}

/// Configures the idle unload timeout in minutes.
pub fn set_unload_timeout_minutes(mins: u64) {
    DEFAULT_UNLOAD_TIMEOUT_MINS.store(mins, Ordering::Release);
    UNLOAD_TIMEOUT_SECS.store(mins * 60, Ordering::Release);
}

pub static INFER_POOL: Lazy<std::sync::RwLock<Option<rayon::ThreadPool>>> =
    Lazy::new(|| std::sync::RwLock::new(None));

/// Executes a closure inside the dedicated inference Rayon thread pool.
pub fn run_in_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let guard = INFER_POOL.read().unwrap();
    if let Some(ref pool) = *guard {
        pool.install(f)
    } else {
        f()
    }
}

static RAYON_POOL_INIT: OnceCell<()> = OnceCell::new();

fn init_thread_pool(threads: u32) {
    RAYON_POOL_INIT.get_or_init(|| {
        let detector = hardware::HardwareDetector::new();
        let mut n = threads.max(1) as usize;
        if let Ok(info) = detector.detect_system() {
            if info.is_weak_system {
                n = n.min(3);
                tracing::info!(
                    "Inference: low-resource system detected. Throttling Rayon threads to {}",
                    n
                );
            } else if info.gpu.is_integrated
                && let Ok(cores) = std::thread::available_parallelism()
            {
                let c = cores.get();
                if c > 2 {
                    n = n.min(c - 1);
                }
            }
        }
        match rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            Ok(_) => tracing::info!("Inference: Rayon thread pool limited to {} threads", n),
            Err(e) => tracing::warn!("Inference: could not configure Rayon thread pool: {e}"),
        }

        if let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(n).build()
            && let Ok(mut guard) = INFER_POOL.write()
            && guard.is_none()
        {
            *guard = Some(pool);
        }
    });
}

pub static LLAMA: Lazy<tokio::sync::Mutex<Option<LlamaState<loader::AnyModel>>>> =
    Lazy::new(|| tokio::sync::Mutex::new(None));

pub static LOADED_MODEL: Lazy<RwLock<Option<String>>> = Lazy::new(|| RwLock::new(None));

pub static INTERRUPT: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new(false));

pub static MODEL_LOADED: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new(false));

pub async fn load_model(model_name: &str, threads: u32) -> Result<(), InferError> {
    init_thread_pool(threads);
    {
        let current = LOADED_MODEL.read().await;
        if current.as_deref() == Some(model_name) {
            return Ok(());
        }
    }

    let mut guard = LLAMA.lock().await;

    {
        let current = LOADED_MODEL.read().await;
        if current.as_deref() == Some(model_name) {
            return Ok(());
        }
    }

    let model_name_owned = model_name.to_string();
    let state = tokio::task::block_in_place(|| {
        run_in_pool(|| loader::try_load(&model_name_owned))
    })
    .map_err(|e| InferError::LlmError(format!("Failed to load model: {e}")))?;

    let device_name = state.device_name.clone();
    *guard = Some(state);
    drop(guard);

    let mut current = LOADED_MODEL.write().await;
    *current = Some(model_name.to_string());
    MODEL_LOADED.store(true, Ordering::Release);
    tracing::info!("Inference: model '{model_name}' ready on {device_name}");

    let timeout_mins = DEFAULT_UNLOAD_TIMEOUT_MINS.load(Ordering::Acquire);
    let timeout_secs = timeout_mins * 60;
    UNLOAD_TIMEOUT_SECS.store(timeout_secs, Ordering::Release);

    {
        if let Ok(mut act) = LAST_ACTIVITY.lock() {
            *act = Some(Instant::now());
        }
    }

    if timeout_secs > 0 && !UNLOAD_TASK_SPAWNED.swap(true, Ordering::SeqCst) {
        tokio::spawn(async {
            loop {
                sleep(Duration::from_secs(1)).await;

                if !MODEL_LOADED.load(Ordering::Acquire) {
                    UNLOAD_TASK_SPAWNED.store(false, Ordering::Release);
                    break;
                }

                let last_act = {
                    match LAST_ACTIVITY.lock() {
                        Ok(act) => *act,
                        Err(_) => None,
                    }
                };

                if let Some(last) = last_act {
                    let timeout_secs = UNLOAD_TIMEOUT_SECS.load(Ordering::Acquire);
                    if timeout_secs > 0 && last.elapsed() >= Duration::from_secs(timeout_secs) {
                        tracing::info!(
                            "Inference: model auto-unload triggered due to {} seconds of inactivity",
                            timeout_secs
                        );
                        unload_model().await;
                        UNLOAD_TASK_SPAWNED.store(false, Ordering::Release);
                        break;
                    }
                }
            }
        });
    }

    Ok(())
}

async fn cpu_fallback<F>(run: F) -> Result<(String, usize), InferError>
where
    F: FnOnce(&mut LlamaState<loader::AnyModel>) -> anyhow::Result<(String, usize)> + Send,
{
    let model_name = LOADED_MODEL.read().await.clone().unwrap_or_default();
    tracing::warn!("Inference: OOM on GPU, falling back to CPU for '{model_name}'");

    let mut cpu_state = tokio::task::block_in_place(|| {
        run_in_pool(|| {
            loader::try_load_on_device(&model_name, candle_core::Device::Cpu)
        })
    })
    .map_err(|e| InferError::OutOfMemory(format!("CPU fallback load failed: {e}")))?;

    let retry = tokio::task::block_in_place(|| {
        run_in_pool(|| run(&mut cpu_state))
    });

    let mut guard = LLAMA.lock().await;
    *guard = Some(cpu_state);
    retry.map_err(|e| InferError::LlmError(format!("CPU fallback failed: {e}")))
}

pub async fn run_inference(
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: Option<f64>,
    repeat_penalty: f32,
    max_context: usize,
    grammar: Option<String>,
) -> Result<(String, usize), InferError> {
    INTERRUPT.store(false, Ordering::SeqCst);
    let params = GenerateParams {
        max_tokens,
        temperature,
        top_p,
        repeat_penalty,
        max_context,
        grammar,
    };

    let mut guard = LLAMA.lock().await;
    let Some(state) = guard.as_mut() else {
        return Err(InferError::ModelNotFound("Model not loaded".into()));
    };

    let result = tokio::task::block_in_place(|| {
        run_in_pool(|| engine::generate(state, prompt, &params))
    });
    if let Ok(mut act) = LAST_ACTIVITY.lock() {
        *act = Some(Instant::now());
    }

    match result {
        Ok(r) => Ok(r),
        Err(e) if device::is_oom(&e) => {
            *guard = None;
            drop(guard);
            cpu_fallback(|s| engine::generate(s, prompt, &params)).await
        }
        Err(e) => Err(InferError::LlmError(format!("Generation failed: {e}"))),
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run_inference_streaming(
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: Option<f64>,
    repeat_penalty: f32,
    max_context: usize,
    grammar: Option<String>,
    on_token: impl Fn(String) + Send + Sync + 'static,
) -> Result<(String, usize), InferError> {
    INTERRUPT.store(false, Ordering::SeqCst);
    let params = GenerateParams {
        max_tokens,
        temperature,
        top_p,
        repeat_penalty,
        max_context,
        grammar,
    };

    let mut guard = LLAMA.lock().await;
    let Some(state) = guard.as_mut() else {
        return Err(InferError::ModelNotFound("Model not loaded".into()));
    };

    let result = tokio::task::block_in_place(|| {
        run_in_pool(|| {
            engine::generate_streaming(state, prompt, &params, &on_token, &INTERRUPT)
        })
    });
    if let Ok(mut act) = LAST_ACTIVITY.lock() {
        *act = Some(Instant::now());
    }

    match result {
        Ok(r) => Ok(r),
        Err(e) if device::is_oom(&e) => {
            *guard = None;
            drop(guard);
            cpu_fallback(|s| {
                engine::generate_streaming(s, prompt, &params, &on_token, &INTERRUPT)
            })
            .await
        }
        Err(_e) if INTERRUPT.load(Ordering::SeqCst) => Err(InferError::Interrupted),
        Err(e) => Err(InferError::LlmError(format!("Streaming failed: {e}"))),
    }
}

pub fn is_model_loaded() -> bool {
    MODEL_LOADED.load(Ordering::Acquire)
}

pub fn loaded_model_name() -> Option<String> {
    LOADED_MODEL.try_read().ok().and_then(|g| g.clone())
}

pub async fn unload_model() {
    let mut guard = LLAMA.lock().await;
    *guard = None;
    let mut current = LOADED_MODEL.write().await;
    *current = None;
    MODEL_LOADED.store(false, Ordering::Release);
}

pub fn interrupt() {
    INTERRUPT.store(true, Ordering::SeqCst);
}

pub fn reset_interrupt() {
    INTERRUPT.store(false, Ordering::SeqCst);
}

pub fn active_device_name() -> String {
    if let Ok(guard) = LLAMA.try_lock()
        && let Some(state) = guard.as_ref()
    {
        return state.device_name.clone();
    }
    if is_model_loaded() {
        "CPU".to_string()
    } else {
        "none".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_llama_mutex_availability() {
        let mut guard = LLAMA.lock().await;
        *guard = None;
        assert!(guard.is_none());
        drop(guard);
        let _guard2 = LLAMA.lock().await;
    }

    #[tokio::test]
    async fn test_auto_unload_trigger() {
        MODEL_LOADED.store(true, Ordering::Release);
        {
            let mut current = LOADED_MODEL.write().await;
            *current = Some("test-model".to_string());
        }
        UNLOAD_TIMEOUT_SECS.store(1, Ordering::Release);
        {
            if let Ok(mut act) = LAST_ACTIVITY.lock() {
                *act = Some(Instant::now());
            }
        }

        if !UNLOAD_TASK_SPAWNED.swap(true, Ordering::SeqCst) {
            tokio::spawn(async {
                loop {
                    sleep(Duration::from_secs(1)).await;

                    if !MODEL_LOADED.load(Ordering::Acquire) {
                        UNLOAD_TASK_SPAWNED.store(false, Ordering::Release);
                        break;
                    }

                    let last_act = {
                        match LAST_ACTIVITY.lock() {
                            Ok(act) => *act,
                            Err(_) => None,
                        }
                    };

                    if let Some(last) = last_act {
                        let timeout_secs = UNLOAD_TIMEOUT_SECS.load(Ordering::Acquire);
                        if last.elapsed() >= Duration::from_secs(timeout_secs) {
                            unload_model().await;
                            UNLOAD_TASK_SPAWNED.store(false, Ordering::Release);
                            break;
                        }
                    }
                }
            });
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
        assert!(!is_model_loaded());
        assert!(loaded_model_name().is_none());
    }
}
