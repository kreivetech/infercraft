use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama;
use candle_transformers::models::quantized_qwen2;
use candle_transformers::models::quantized_gemma3;
use candle_transformers::models::quantized_phi3;
use tokenizers::Tokenizer;
use super::device::{select_device, device_label};

pub trait ModelBackend: Send {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor>;
}

pub enum AnyModel {
    Llama(quantized_llama::ModelWeights),
    StreamingLlama(super::streaming_llama::StreamingLlamaWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Gemma3(quantized_gemma3::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
}

impl ModelBackend for AnyModel {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m)          => m.forward(x, index_pos),
            Self::StreamingLlama(m) => m.forward(x, index_pos),
            Self::Qwen2(m)          => m.forward(x, index_pos),
            Self::Gemma3(m)         => m.forward(x, index_pos),
            Self::Phi3(m)           => m.forward(x, index_pos),
        }
    }
}

pub struct LlamaState<M: ModelBackend = AnyModel> {
    pub model:       M,
    pub tokenizer:   Tokenizer,
    pub device:      Device,
    pub device_name: String,
    pub eos_token:   u32,
}

pub fn try_load_on_device(model_name: &str, device: Device) -> anyhow::Result<LlamaState<AnyModel>> {
    let device_name = device_label(&device);
    let models_dir = crate::get_models_dir();
    let model_path = models_dir.join(format!("{model_name}.gguf"));
    tracing::info!("Inference: loading GGUF from {} on {device_name}", model_path.display());

    let mut file = std::fs::File::open(&model_path)
        .map_err(|e| anyhow::anyhow!("Cannot open {}: {e}", model_path.display()))?;

    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("GGUF read failed: {e}"))?;

    let eos_from_gguf = content.metadata.get("tokenizer.ggml.eos_token_id")
        .and_then(|v| match v {
            gguf_file::Value::U32(n) => Some(*n),
            gguf_file::Value::I32(n) => Some(*n as u32),
            _ => None,
        });

    let arch = content.metadata.get("general.architecture")
        .and_then(|v| match v {
            gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "llama".to_string());

    tracing::info!("Inference: GGUF architecture = '{arch}'");

    let model = match arch.as_str() {
        "qwen2" => {
            let m = quantized_qwen2::ModelWeights::from_gguf(content, &mut file, &device)
                .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf failed: {e}"))?;
            AnyModel::Qwen2(m)
        }
        "gemma3" | "gemma2" | "gemma" => {
            let m = quantized_gemma3::ModelWeights::from_gguf(content, &mut file, &device)
                .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf failed: {e}"))?;
            AnyModel::Gemma3(m)
        }
        "phi3" | "phi4" => {
            let m = quantized_phi3::ModelWeights::from_gguf(false, content, &mut file, &device)
                .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf failed: {e}"))?;
            AnyModel::Phi3(m)
        }
        _ => {
            // "llama" and any unknown architecture use zero-copy mmap + layer streaming by default
            // to radically minimize RAM consumption across all platforms (Linux, Windows, macOS).
            let disable_streaming = std::env::var("UNEVEN_STREAMING_LAYERS")
                .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
                .unwrap_or(false);

            if !disable_streaming {
                tracing::info!("Inference: loading with zero-copy mmap and layer-by-layer streaming (default)");
                let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
                    .map_err(|e| anyhow::anyhow!("Failed to mmap {}: {e}", model_path.display()))?;
                let m = super::streaming_llama::StreamingLlamaWeights::from_mmap(
                    std::sync::Arc::new(mmap),
                    content,
                    &device,
                )
                .map_err(|e| anyhow::anyhow!("StreamingLlamaWeights::from_mmap failed: {e}"))?;
                AnyModel::StreamingLlama(m)
            } else {
                tracing::info!("Inference: streaming explicitly disabled, loading full model in memory");
                let m = quantized_llama::ModelWeights::from_gguf(content, &mut file, &device)
                    .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf failed: {e}"))?;
                AnyModel::Llama(m)
            }
        }
    };

    let tokenizer_path = models_dir.join(format!("{model_name}-tokenizer.json"));
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!(
            "Tokenizer not found at {}: {e}\n\
             Run `uneven init` to download it.",
             tokenizer_path.display()
        ))?;

    let eos_token = eos_from_gguf
        .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .or_else(|| tokenizer.token_to_id("<|end|>"))
        .or_else(|| tokenizer.token_to_id("<eos>"))
        .or_else(|| tokenizer.token_to_id("</s>"))
        .unwrap_or(128009);

    Ok(LlamaState { model, tokenizer, device, device_name, eos_token })
}

pub fn try_load(model_name: &str) -> anyhow::Result<LlamaState<AnyModel>> {
    let device = select_device();
    let is_cpu = matches!(device, Device::Cpu);

    match try_load_on_device(model_name, device) {
        Ok(state) => Ok(state),
        Err(e) if !is_cpu => {
            tracing::warn!("Inference: failed to load on GPU ({e}), falling back to CPU");
            try_load_on_device(model_name, Device::Cpu)
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use tempfile::tempdir;

    fn write_mock_gguf(path: &std::path::Path, arch: &str) {
        let mut f = std::fs::File::create(path).unwrap();
        // Magic
        f.write_all(b"GGUF").unwrap();
        // Version 3 (u32)
        f.write_all(&3u32.to_le_bytes()).unwrap();
        // Tensor count 0 (u64)
        f.write_all(&0u64.to_le_bytes()).unwrap();
        // KV count 1 (u64)
        f.write_all(&1u64.to_le_bytes()).unwrap();
        
        // Key: "general.architecture"
        let key = "general.architecture";
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        f.write_all(key.as_bytes()).unwrap();
        
        // Type: String (8)
        f.write_all(&8u32.to_le_bytes()).unwrap();
        
        // Value: arch
        f.write_all(&(arch.len() as u64).to_le_bytes()).unwrap();
        f.write_all(arch.as_bytes()).unwrap();
    }

    #[test]
    fn test_gguf_architecture_routing() {
        let dir = tempdir().unwrap();
        let gguf_path = dir.path().join("test.gguf");
        write_mock_gguf(&gguf_path, "qwen2");
        
        // We can't easily call try_load_on_device because it has hardcoded paths to ./.uneven/models
        // But we can test the architecture matching logic if we refactor it slightly or mock the file open.
        // For now, let's just verify the GGUF header we wrote is valid for candle.
        
        let mut file = std::fs::File::open(&gguf_path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();
        
        let arch = content.metadata.get("general.architecture")
            .and_then(|v| match v {
                candle_core::quantized::gguf_file::Value::String(s) => Some(s.clone()),
                _ => None,
            });
            
        assert_eq!(arch.unwrap(), "qwen2");
    }
}
