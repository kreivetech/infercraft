use candle_core::Tensor;
use tokenizers::Tokenizer;

/// A simplified grammar constraint engine.
#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub markers: Vec<String>,
}

impl GrammarConstraint {
    pub fn new(markers: Vec<String>) -> Self {
        Self { markers }
    }

    /// Filters logits based on the current generation progress.
    /// This ensures markers appear in the correct order and aren't skipped.
    pub fn filter_logits(
        &self, 
        logits: &Tensor, 
        tokenizer: &Tokenizer, 
        generated_tokens: &[u32]
    ) -> candle_core::Result<Tensor> {
        if self.markers.is_empty() {
            return Ok(logits.clone());
        }

        // Decode the current text to see which markers are already present
        let current_text = tokenizer.decode(generated_tokens, true)
            .unwrap_or_default();

        let mut forbidden_token_ids: Vec<u32> = Vec::new();

        // Logic: 
        // 1. If marker[i] is NOT in text, marker[i+1] is FORBIDDEN.
        // 2. This prevents the model from skipping steps (e.g. jumping to EXPLANATION: before VISUAL:).
        for i in 0..self.markers.len() - 1 {
            let current_marker = &self.markers[i];
            let next_marker = &self.markers[i+1];

            if !current_text.contains(current_marker) {
                // If we haven't seen the current marker yet, the NEXT marker is forbidden.
                if let Ok(encoding) = tokenizer.encode(next_marker.as_str(), false) {
                    forbidden_token_ids.extend(encoding.get_ids());
                }
            }
        }

        if forbidden_token_ids.is_empty() {
            return Ok(logits.clone());
        }

        // Apply penalty (-inf) to forbidden tokens
        let mut logits_vec = logits.to_vec1::<f32>()?;
        for &id in &forbidden_token_ids {
            if (id as usize) < logits_vec.len() {
                logits_vec[id as usize] = f32::NEG_INFINITY;
            }
        }

        Tensor::from_vec(logits_vec, logits.dims(), logits.device())
    }
}
