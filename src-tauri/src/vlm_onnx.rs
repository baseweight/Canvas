// VLM ONNX inference engine for Baseweight Canvas
// Supports ONNX-based Vision-Language Models
// Initially focused on SmolVLM, expandable to other VLMs

use anyhow::{Result, Context};
use image::DynamicImage;
use ndarray::{Array, Array4, Array5, s};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(target_os = "linux")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;

// SmolVLM configuration
struct SmolVLMConfig {
    num_key_value_heads: usize,
    head_dim: usize,
    num_hidden_layers: usize,
    eos_token_id: u32,
    image_token_id: u32,
    max_context_length: usize,
}

pub struct VlmOnnx {
    vision_session: Session,
    embed_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
    config: SmolVLMConfig,
}

impl VlmOnnx {
    pub fn new(
        vision_model_path: &Path,
        embed_model_path: &Path,
        decoder_model_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self> {
        println!("Loading SmolVLM ONNX models...");

        // Create sessions with platform-specific execution providers
        #[cfg(target_os = "windows")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "linux")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "macos")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            // Using CPU only for now - CoreML has compatibility issues
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "windows")]
        let embed_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(embed_model_path)
            .context("Failed to create embed session")?;

        #[cfg(target_os = "linux")]
        let embed_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(embed_model_path)
            .context("Failed to create embed session")?;

        #[cfg(target_os = "macos")]
        let embed_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(embed_model_path)
            .context("Failed to create embed session")?;

        #[cfg(target_os = "windows")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        #[cfg(target_os = "linux")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        #[cfg(target_os = "macos")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Get special tokens
        let image_token_id = tokenizer.token_to_id("<image>")
            .ok_or_else(|| anyhow::anyhow!("Failed to get image token ID"))?;
        let eos_token_id = 2; // SmolVLM2 EOS token

        // Auto-detect configuration from decoder
        let decoder_inputs = &decoder_session.inputs;
        let mut max_layer = 0;
        let detected_kv_heads = 5;

        for input in decoder_inputs {
            if input.name.starts_with("past_key_values.") {
                if let Some(layer_str) = input.name.split('.').nth(1) {
                    if let Ok(layer_num) = layer_str.parse::<usize>() {
                        max_layer = max_layer.max(layer_num);
                    }
                }
            }
        }

        let num_hidden_layers = max_layer + 1;
        println!("Auto-detected: {} layers, {} kv heads", num_hidden_layers, detected_kv_heads);

        let config = SmolVLMConfig {
            num_key_value_heads: detected_kv_heads,
            head_dim: 64,
            num_hidden_layers,
            eos_token_id,
            image_token_id,
            max_context_length: 2048,
        };

        Ok(Self {
            vision_session,
            embed_session,
            decoder_session,
            tokenizer,
            config,
        })
    }

    pub fn generate(&mut self, prompt: &str, image_data: &[u8], width: u32, height: u32) -> Result<String> {
        // Load and preprocess image
        let image = image::load_from_memory(image_data)?;
        let (processed_image, pixel_attention_mask) = preprocess_image(image)?;

        // Expand prompt with image tokens
        let expanded_prompt = expand_prompt_with_image(prompt);

        // Tokenize
        let encoding = self.tokenizer.encode(expanded_prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>();
        let attention_mask = encoding.get_attention_mask().iter().map(|&x| x as i64).collect::<Vec<_>>();

        // Initialize past key values
        let batch_size = 1;
        let mut past_key_values: HashMap<String, Array<f32, _>> = HashMap::new();
        for layer in 0..self.config.num_hidden_layers {
            let key_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).into_dyn();
            let value_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).into_dyn();
            past_key_values.insert(format!("past_key_values.{}.key", layer), key_array);
            past_key_values.insert(format!("past_key_values.{}.value", layer), value_array);
        }

        // Get image features
        let mut vision_inputs: HashMap<&str, Value> = HashMap::new();
        vision_inputs.insert("pixel_values", Value::from_array(processed_image.clone())?.into());

        let pixel_attention_mask_bool = pixel_attention_mask.map(|&x| x != 0);
        vision_inputs.insert("pixel_attention_mask", Value::from_array(pixel_attention_mask_bool)?.into());

        let vision_outputs = self.vision_session.run(vision_inputs)?;
        let image_features = vision_outputs[0].try_extract_array::<f32>()?.to_owned();

        // Reshape features
        let total_size = image_features.shape()[0] * image_features.shape()[1];
        let feature_dim = image_features.shape()[2];
        let image_features_reshaped = image_features.into_shape_with_order((total_size, feature_dim))?;

        // Generation loop
        let max_new_tokens = 1024;
        let mut generated_tokens = Vec::new();
        let mut input_ids = Array::from_vec(input_ids)
            .into_shape_with_order((1, encoding.get_ids().len()))?
            .into_owned();
        let mut attention_mask = Array::from_vec(attention_mask)
            .into_shape_with_order((1, encoding.get_attention_mask().len()))?
            .into_owned();

        // Calculate position_ids
        let mut position_ids_vec = Vec::new();
        let mut cumsum = 0i64;
        for &mask_val in attention_mask.iter() {
            cumsum += mask_val;
            position_ids_vec.push(cumsum);
        }
        let mut position_ids = Array::from_vec(position_ids_vec)
            .into_shape_with_order((1, attention_mask.len()))?
            .into_owned();

        for _ in 0..max_new_tokens {
            // Get input embeddings
            let mut embed_inputs: HashMap<&str, Value> = HashMap::new();
            embed_inputs.insert("input_ids", Value::from_array(input_ids.clone())?.into());
            let embed_outputs = self.embed_session.run(embed_inputs)?;
            let mut input_embeds = embed_outputs[0].try_extract_array::<f32>()?.to_owned();

            // Replace image token embeddings
            let mut feature_idx = 0;
            for i in 0..input_ids.shape()[1] {
                if input_ids[[0, i]] == self.config.image_token_id as i64 {
                    let mut slice = input_embeds.slice_mut(s![0, i, ..]);
                    slice.assign(&image_features_reshaped.slice(s![feature_idx, ..]));
                    feature_idx += 1;
                }
            }

            // Run decoder
            let mut decoder_inputs: HashMap<&str, Value> = HashMap::new();
            decoder_inputs.insert("inputs_embeds", Value::from_array(input_embeds.clone())?.into());
            decoder_inputs.insert("attention_mask", Value::from_array(attention_mask.clone())?.into());
            decoder_inputs.insert("position_ids", Value::from_array(position_ids.clone())?.into());

            for (key, value) in &past_key_values {
                decoder_inputs.insert(key, Value::from_array(value.clone())?.into());
            }

            let decoder_outputs = self.decoder_session.run(decoder_inputs)?;
            let logits = decoder_outputs[0].try_extract_array::<f32>()?.to_owned();

            // Get next token (argmax)
            let last_idx = logits.shape()[1] - 1;
            let logits_slice = logits.slice(s![0, last_idx, ..]);
            let next_token = logits_slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap();

            // Update for next iteration
            input_ids = Array::from_vec(vec![next_token]).into_shape_with_order((1, 1))?;

            let new_attention = Array::<i64, _>::ones((1, 1));
            let total_length = attention_mask.len() + new_attention.len();
            let mut attention_vec = attention_mask.iter().copied().collect::<Vec<_>>();
            attention_vec.extend(new_attention.iter().copied());
            attention_mask = Array::from_vec(attention_vec).into_shape_with_order((1, total_length))?;

            let current_pos = position_ids[[0, position_ids.shape()[1] - 1]] + 1;
            position_ids = Array::from_vec(vec![current_pos]).into_shape_with_order((1, 1))?;

            // Update past key values
            for i in 0..self.config.num_hidden_layers {
                let key = format!("past_key_values.{}.key", i);
                let value = format!("past_key_values.{}.value", i);

                if let Some(past_key) = past_key_values.get_mut(&key) {
                    if i * 2 + 1 < decoder_outputs.len() {
                        let present_key = decoder_outputs[i * 2 + 1].try_extract_array::<f32>()?.to_owned();
                        *past_key = present_key.into_dyn();
                    }
                }

                if let Some(past_value) = past_key_values.get_mut(&value) {
                    if i * 2 + 2 < decoder_outputs.len() {
                        let present_value = decoder_outputs[i * 2 + 2].try_extract_array::<f32>()?.to_owned();
                        *past_value = present_value.into_dyn();
                    }
                }
            }

            generated_tokens.push(next_token as u32);

            // Check for EOS
            if next_token == self.config.eos_token_id as i64 {
                break;
            }
        }

        // Decode
        let generated_text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;
        Ok(generated_text)
    }
}

// Image preprocessing for SmolVLM
fn preprocess_image(image: DynamicImage) -> Result<(Array5<f32>, Array4<i64>)> {
    // Simple preprocessing - resize to 512x512 for now
    let image = image.resize_exact(512, 512, image::imageops::FilterType::Lanczos3);
    let image = image.to_rgb8();

    let mut pixel_values = Array5::<f32>::zeros((1, 1, 3, 512, 512));

    for (x, y, pixel) in image.enumerate_pixels() {
        for c in 0..3 {
            let val = pixel[c] as f32 / 255.0;
            let normalized = (val - 0.5) / 0.5;
            pixel_values[[0, 0, c, y as usize, x as usize]] = normalized;
        }
    }

    let attention_mask = Array4::ones((1, 1, 512, 512));

    Ok((pixel_values, attention_mask))
}

// Expand prompt with image tokens (simplified for now)
fn expand_prompt_with_image(prompt: &str) -> String {
    // For now, just repeat <image> token 64 times as per SmolVLM2
    let image_tokens = "<image>".repeat(64);
    prompt.replace("<image>", &image_tokens)
}
