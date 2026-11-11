use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, IndexOp, Module, Result, Tensor};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::{build_causal_mask, repeat_kv};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

const MAX_SEQ_LEN: usize = 4096;

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(candle_core::DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

fn apply_rotary_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    index_pos: usize,
    rope_is_neox: bool,
) -> Result<Tensor> {
    let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
    let cos = cos.narrow(0, index_pos, seq_len)?;
    let sin = sin.narrow(0, index_pos, seq_len)?;
    let x = x.contiguous()?;
    if rope_is_neox {
        candle_nn::rotary_emb::rope(&x, &cos, &sin)
    } else {
        candle_nn::rotary_emb::rope_i(&x, &cos, &sin)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

/// Transient weights for a single transformer block loaded on-demand from mmap.
pub struct StreamingLayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    ffn_norm: RmsNorm,
}

impl StreamingLayerWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        head_count: usize,
        head_count_kv: usize,
        head_dim: usize,
        rope_is_neox: bool,
        cos: &Tensor,
        sin: &Tensor,
        neg_inf: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let residual = x;
        let norm_x = self.attention_norm.forward(x)?;

        let q = self.attention_wq.forward(&norm_x)?;
        let k = self.attention_wk.forward(&norm_x)?;
        let v = self.attention_wv.forward(&norm_x)?;

        let q = q
            .reshape((b_sz, seq_len, head_count, head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, head_count_kv, head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, head_count_kv, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = apply_rotary_emb(&q, cos, sin, index_pos, rope_is_neox)?;
        let k = apply_rotary_emb(&k, cos, sin, index_pos, rope_is_neox)?;

        let (k, v) = match kv_cache.as_ref() {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[prev_k, &k], 2)?;
                    let v = Tensor::cat(&[prev_v, &v], 2)?;
                    (k, v)
                }
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, head_count / head_count_kv)?;
        let v = repeat_kv(v, head_count / head_count_kv)?;

        let att = (q.matmul(&k.t()?)? / (head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        let x = (y + residual)?;

        // Feed forward (MLP)
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let w1 = self.feed_forward_w1.forward(&x)?;
        let w3 = self.feed_forward_w3.forward(&x)?;
        let silu_w1 = candle_nn::ops::silu(&w1)?;
        let mlp_act = silu_w1.broadcast_mul(&w3)?;
        let y = self.feed_forward_w2.forward(&mlp_act)?;
        y + residual
    }
}

/// Zero-Copy, Layer-Streaming model for Llama / SmolLM architectures.
/// Only keeps base embeddings, norm and 1 layer of weights active in RAM at any time.
pub struct StreamingLlamaWeights {
    mmap: Arc<memmap2::Mmap>,
    content: Arc<gguf_file::Content>,
    tok_embeddings: candle_nn::Embedding,
    norm: RmsNorm,
    output: QMatMul,
    block_count: usize,
    head_dim: usize,
    head_count: usize,
    head_count_kv: usize,
    rope_is_neox: bool,
    rms_norm_eps: f64,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    device: Device,
    kv_caches: Vec<Option<(Tensor, Tensor)>>,
    masks: HashMap<(usize, usize), Tensor>,
}

impl StreamingLlamaWeights {
    pub fn from_mmap(
        mmap: Arc<memmap2::Mmap>,
        content: gguf_file::Content,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match content.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")
            .and_then(|v| v.to_u32())
            .unwrap_or((embedding_length / head_count) as u32) as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let rope_is_neox = matches!(arch.as_str(), "qwen" | "qwen2" | "phi2" | "phi3");

        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let mut cursor = Cursor::new(&mmap[..]);

        let tok_embeddings_q = content.tensor(&mut cursor, "token_embd.weight", device)?;
        let tok_embeddings_raw = tok_embeddings_q.dequantize(device)?;
        let tok_embeddings = candle_nn::Embedding::new(tok_embeddings_raw, embedding_length);

        let norm = RmsNorm::from_qtensor(
            content.tensor(&mut cursor, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;

        let output = match content.tensor(&mut cursor, "output.weight", device) {
            Ok(tensor) => QMatMul::from_qtensor(tensor)?,
            Err(_) => QMatMul::from_qtensor(tok_embeddings_q)?,
        };

        let kv_caches = vec![None; block_count];

        Ok(Self {
            mmap,
            content: Arc::new(content),
            tok_embeddings,
            norm,
            output,
            block_count,
            head_dim: embedding_length / head_count,
            head_count,
            head_count_kv,
            rope_is_neox,
            rms_norm_eps,
            cos,
            sin,
            neg_inf,
            device: device.clone(),
            kv_caches,
            masks: HashMap::new(),
        })
    }

    pub fn load_layer(&self, layer_idx: usize) -> Result<StreamingLayerWeights> {
        let prefix = format!("blk.{layer_idx}");
        let mut cursor = Cursor::new(&self.mmap[..]);

        let attention_wq = QMatMul::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.attn_q.weight"), &self.device)?
        )?;
        let attention_wk = QMatMul::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.attn_k.weight"), &self.device)?
        )?;
        let attention_wv = QMatMul::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.attn_v.weight"), &self.device)?
        )?;
        let attention_wo = QMatMul::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.attn_output.weight"), &self.device)?
        )?;
        let attention_norm = RmsNorm::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.attn_norm.weight"), &self.device)?,
            self.rms_norm_eps,
        )?;

        let feed_forward_w1 = QMatMul::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.ffn_gate.weight"), &self.device)?
        )?;
        let feed_forward_w2 = QMatMul::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.ffn_down.weight"), &self.device)?
        )?;
        let feed_forward_w3 = QMatMul::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.ffn_up.weight"), &self.device)?
        )?;
        let ffn_norm = RmsNorm::from_qtensor(
            self.content.tensor(&mut cursor, &format!("{prefix}.ffn_norm.weight"), &self.device)?,
            self.rms_norm_eps,
        )?;

        Ok(StreamingLayerWeights {
            attention_wq,
            attention_wk,
            attention_wv,
            attention_wo,
            attention_norm,
            feed_forward_w1,
            feed_forward_w2,
            feed_forward_w3,
            ffn_norm,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for cache in self.kv_caches.iter_mut() {
            *cache = None;
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };

        let mut layer_in = self.tok_embeddings.forward(x)?;

        for layer_idx in 0..self.block_count {
            // Load ONLY the weights of the current layer from mmap
            let layer = self.load_layer(layer_idx)?;
            let mut kv = self.kv_caches[layer_idx].take();

            let out = layer.forward(
                &layer_in,
                mask.as_ref(),
                index_pos,
                self.head_count,
                self.head_count_kv,
                self.head_dim,
                self.rope_is_neox,
                &self.cos,
                &self.sin,
                &self.neg_inf,
                &mut kv,
            )?;

            // Store back only the KV cache
            self.kv_caches[layer_idx] = kv;
            layer_in = out;

            // Immediately drop layer to release RSS from memory
            drop(layer);
        }

        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }
}
