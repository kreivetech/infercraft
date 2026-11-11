use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use std::sync::atomic::Ordering;
use super::loader::{LlamaState, ModelBackend};
use super::types::GenerateParams;
use super::grammar::GrammarConstraint;
use super::INTERRUPT;

struct GenerationSetup {
    tokens: Vec<u32>,
    generated: Vec<u32>,
    logits_processor: LogitsProcessor,
    grammar: Option<GrammarConstraint>,
}

fn setup_generation<M: ModelBackend>(
    state: &LlamaState<M>,
    prompt: &str,
    p: &GenerateParams,
) -> anyhow::Result<GenerationSetup> {
    // add_special_tokens=false: chat templates explicitly include all special tokens
    // (BOS, role markers, EOS anchors). Passing true would double-add BOS on Llama 3.
    let encoding = state.tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenize: {e}"))?;

    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

    let max_prompt_tokens = p.max_context.saturating_sub(p.max_tokens);
    if max_prompt_tokens == 0 {
        tracing::error!(
            "Inference misconfiguration: max_context={} must exceed max_tokens={}",
            p.max_context, p.max_tokens
        );
        anyhow::bail!("Inference unavailable: context window is too small for the requested output length");
    }
    if tokens.len() > max_prompt_tokens {
        let excess = tokens.len() - max_prompt_tokens;
        tokens.drain(..excess);
        tracing::warn!("Inference: prompt truncated by {} tokens", excess);
    }
    if tokens.is_empty() {
        anyhow::bail!("Inference unavailable: prompt could not be processed");
    }

    tokens.reserve(p.max_tokens);
    let generated = Vec::with_capacity(p.max_tokens);

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let temp_opt = if p.temperature < 1e-7 { None } else { Some(p.temperature) };
    let logits_processor = LogitsProcessor::new(seed, temp_opt, p.top_p);

    let grammar = p.grammar.as_ref().map(|g| {
        if g == "analyze" {
            GrammarConstraint::new(vec!["VISUAL:".to_string(), "EXPLANATION:".to_string()])
        } else {
            GrammarConstraint::new(vec![])
        }
    });

    Ok(GenerationSetup { tokens, generated, logits_processor, grammar })
}

fn next_token<M: ModelBackend>(
    state: &mut LlamaState<M>,
    tokens: &[u32],
    generated: &[u32],
    lp: &mut LogitsProcessor,
    grammar: &Option<GrammarConstraint>,
    p: &GenerateParams,
    step: usize,
) -> anyhow::Result<u32> {
    let logits = if step == 0 {
        let threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4);
        let hw_info = crate::hardware::HardwareDetector::new()
            .detect_system()
            .ok();
        let is_weak = hw_info.as_ref().map(|info| info.is_weak_system).unwrap_or(true);
        let is_integrated = hw_info.as_ref().map(|info| info.gpu.is_integrated).unwrap_or(false);

        let max_chunk = if is_integrated {
            64
        } else if is_weak {
            32
        } else {
            128
        };
        let alignment = (threads * 4).max(8);
        let n = tokens.len();

        let chunk_size = if n <= max_chunk {
            let aligned = n.div_ceil(alignment) * alignment;
            aligned.min(max_chunk)
        } else {
            let aligned = (max_chunk / alignment) * alignment;
            if aligned == 0 { max_chunk } else { aligned }
        };

        let sleep_ms = if is_integrated {
            4
        } else if is_weak {
            10
        } else {
            2
        };
        tracing::info!(
            "Inference prefill: total_tokens={}, chunk_size={}, threads={}, sleep_ms={}",
            n, chunk_size, threads, sleep_ms
        );

        if n <= chunk_size {
            let input = Tensor::new(tokens, &state.device)?.unsqueeze(0)?;
            state.model.forward(&input, 0)?
        } else {
            let mut last_logits = None;
            let mut start = 0;
            while start < n {
                let end = (start + chunk_size).min(n);
                let chunk = &tokens[start..end];
                let input = Tensor::new(chunk, &state.device)?.unsqueeze(0)?;
                let lg = state.model.forward(&input, start)?;
                if end == n {
                    last_logits = Some(lg);
                } else {
                    std::thread::yield_now();
                    std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
                }
                start += chunk_size;
            }
            last_logits.ok_or_else(|| anyhow::anyhow!("Chunked prefill failed to produce logits"))?
        }
    } else {
        let context_start = tokens.len() - 1;
        let input = Tensor::new(&tokens[context_start..], &state.device)?.unsqueeze(0)?;
        state.model.forward(&input, context_start)?
    };

    let last = {
        let l = logits.squeeze(0)?;
        if l.dims().len() == 2 { l.get(l.dim(0)? - 1)? } else { l }
    };

    let last = if p.repeat_penalty != 1.0 && !generated.is_empty() {
        let start = generated.len().saturating_sub(32);
        apply_repeat_penalty(&last, p.repeat_penalty, &generated[start..])?
    } else {
        last
    };

    let last = if let Some(g) = grammar {
        g.filter_logits(&last, &state.tokenizer, generated)?
    } else {
        last
    };

    Ok(lp.sample(&last)?)
}

pub fn generate<M: ModelBackend>(
    state: &mut LlamaState<M>,
    prompt: &str,
    p: &GenerateParams,
) -> anyhow::Result<(String, usize)> {
    let GenerationSetup { mut tokens, mut generated, mut logits_processor, grammar } =
        setup_generation(state, prompt, p)?;

    for step in 0..p.max_tokens {
        if INTERRUPT.load(Ordering::SeqCst) { break; }
        let next = next_token(state, &tokens, &generated, &mut logits_processor, &grammar, p, step)?;
        if next == state.eos_token { break; }
        if let Some(token_str) = state.tokenizer.id_to_token(next)
            && (token_str == "<|im_end|>" || token_str == "<|eot_id|>" || token_str == "<|end|>" || token_str == "<end_of_turn>" || token_str == "</s>") {
                break;
            }
        tokens.push(next);
        generated.push(next);
    }

    let text = state.tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Decode: {e}"))?;

    Ok((text, generated.len()))
}

pub fn generate_streaming<M: ModelBackend>(
    state: &mut LlamaState<M>,
    prompt: &str,
    p: &GenerateParams,
    on_token: &impl Fn(String),
    cancelled: &std::sync::atomic::AtomicBool,
) -> anyhow::Result<(String, usize)> {
    let GenerationSetup { mut tokens, mut generated, mut logits_processor, grammar } =
        setup_generation(state, prompt, p)?;

    let mut prev_text_len: usize = 0;

    for step in 0..p.max_tokens {
        if INTERRUPT.load(Ordering::SeqCst) { break; }
        let next = next_token(state, &tokens, &generated, &mut logits_processor, &grammar, p, step)?;
        if next == state.eos_token { break; }
        if let Some(token_str) = state.tokenizer.id_to_token(next)
            && (token_str == "<|im_end|>" || token_str == "<|eot_id|>" || token_str == "<|end|>" || token_str == "<end_of_turn>" || token_str == "</s>") {
                break;
            }
        tokens.push(next);
        generated.push(next);

        if let Ok(full) = state.tokenizer.decode(&generated, true)
            && full.len() > prev_text_len && full.is_char_boundary(prev_text_len) {
                let piece = full[prev_text_len..].to_string();
                prev_text_len = full.len();
                if !piece.is_empty() && !cancelled.load(Ordering::Acquire) {
                    on_token(piece);
                }
            }
    }

    let full_text = state.tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Decode: {e}"))?;

    Ok((full_text, generated.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use tokenizers::Tokenizer;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::str::FromStr;

    struct MockModel {
        pub next_token: u32,
    }

    impl ModelBackend for MockModel {
        fn forward(&mut self, x: &Tensor, _pos: usize) -> candle_core::Result<Tensor> {
            let (b, s) = x.dims2()?;
            let vocab_size = 100;
            let mut data = vec![0.0f32; b * s * vocab_size];
            for i in 0..(b * s) {
                data[i * vocab_size + self.next_token as usize] = 10.0;
            }
            Tensor::from_vec(data, (b, s, vocab_size), x.device())
        }
    }

    fn create_mock_state(next_token: u32) -> LlamaState<MockModel> {
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": { "<pad>": 0, "<s>": 1, "</s>": 2, "x": 42, "y": 43 },
                "merges": []
            }
        }"#;
        let tokenizer = Tokenizer::from_str(json).unwrap();
        LlamaState {
            model: MockModel { next_token },
            tokenizer,
            device: Device::Cpu,
            device_name: "MockCPU".to_string(),
            eos_token: 2,
        }
    }

    #[test]
    fn test_setup_generation_truncation() {
        let state = create_mock_state(42);
        let params = GenerateParams {
            max_tokens: 10,
            temperature: 0.0,
            top_p: None,
            repeat_penalty: 1.0,
            max_context: 15,
            grammar: None,
        };

        let setup = setup_generation(&state, "xxxxxxx", &params).unwrap();
        assert_eq!(setup.tokens.len(), 5);
        assert_eq!(setup.generated.capacity(), 10);
    }

    #[test]
    fn test_generate_stops_at_max_tokens() {
        let mut state = create_mock_state(42);
        let params = GenerateParams {
            max_tokens: 3,
            temperature: 0.0,
            top_p: None,
            repeat_penalty: 1.0,
            max_context: 128,
            grammar: None,
        };
        let (text, len) = generate(&mut state, "x", &params).unwrap();
        assert_eq!(len, 3);
        assert_eq!(text, "x x x");
    }

    #[test]
    fn test_generate_stops_at_eos() {
        let mut state = create_mock_state(2);
        let params = GenerateParams {
            max_tokens: 10,
            temperature: 0.0,
            top_p: None,
            repeat_penalty: 1.0,
            max_context: 128,
            grammar: None,
        };
        let (text, len) = generate(&mut state, "x", &params).unwrap();
        assert_eq!(len, 0);
        assert_eq!(text, "");
    }

    #[test]
    fn test_generate_streaming_basic() {
        let mut state = create_mock_state(43);
        let params = GenerateParams {
            max_tokens: 2,
            temperature: 0.0,
            top_p: None,
            repeat_penalty: 1.0,
            max_context: 128,
            grammar: None,
        };
        let collected = Mutex::new(Vec::new());
        let on_token = |t: String| collected.lock().unwrap().push(t);
        let cancelled = AtomicBool::new(false);

        let (text, len) = generate_streaming(&mut state, "x", &params, &on_token, &cancelled).unwrap();
        
        assert_eq!(len, 2);
        assert_eq!(text, "y y");
        let tokens = collected.lock().unwrap();
        assert_eq!(tokens.len(), 2);
        // The first token might have a space prepended depending on tokenizer settings, 
        // but here they seem to be emitted as "y" and " y".
    }

    #[test]
    fn test_generate_streaming_cancellation() {
        let mut state = create_mock_state(42);
        let params = GenerateParams {
            max_tokens: 5,
            temperature: 0.0,
            top_p: None,
            repeat_penalty: 1.0,
            max_context: 128,
            grammar: None,
        };
        let count = AtomicUsize::new(0);
        let cancelled = AtomicBool::new(false);
        
        let on_token = |_: String| {
            count.fetch_add(1, Ordering::SeqCst);
        };

        let (text, len) = generate_streaming(&mut state, "x", &params, &on_token, &cancelled).unwrap();

        assert_eq!(len, 5);
        assert_eq!(text, "x x x x x");
        assert_eq!(count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_grammar_constraint_filtering() {
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": { "A": 0, "B": 1, "C": 2 },
                "merges": []
            }
        }"#;
        let tokenizer = Tokenizer::from_str(json).unwrap();
        let grammar = GrammarConstraint::new(vec!["A".to_string(), "B".to_string()]);
        
        // Logits for A, B, C: [10.0, 10.0, 10.0]
        let logits = Tensor::from_vec(vec![10.0f32, 10.0, 10.0], (3,), &Device::Cpu).unwrap();
        let generated = vec![]; // Empty text, so "B" should be forbidden
        
        let filtered = grammar.filter_logits(&logits, &tokenizer, &generated).unwrap();
        let data = filtered.to_vec1::<f32>().unwrap();
        
        // "A" (index 0) should remain high
        assert!(data[0] > 0.0);
        // "B" (index 1) should be forbidden because "A" is not yet in generated text
        assert!(data[1] < -100.0);
        // "C" (index 2) remains allowed (not a sequential marker)
        assert!(data[2] > 0.0);
    }

    #[test]
    fn test_max_context_sliding_window() {
        let state = create_mock_state(42);
        let params = GenerateParams {
            max_tokens: 5,
            temperature: 0.0,
            top_p: None,
            repeat_penalty: 1.0,
            max_context: 10, // Very small context
            grammar: None,
        };

        // Prompt has 7 tokens (encoding "xxxxxxx")
        // max_prompt_tokens = 10 - 5 = 5
        // Should truncate first 2 tokens.
        let setup = setup_generation(&state, "xxxxxxx", &params).unwrap();
        assert_eq!(setup.tokens.len(), 5);
    }
}
