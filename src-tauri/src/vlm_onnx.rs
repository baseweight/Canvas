// VLM ONNX inference engine for Baseweight Canvas
// Supports ONNX-based Vision-Language Models
// Initially focused on SmolVLM, expandable to other VLMs

use anyhow::{Result, Context};
use image::{DynamicImage, GenericImageView};
use ndarray::{Array, Array4, Array5, s};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

// ============================================================================
// SmolVLM Image Processor - handles 4x4 grid splitting + global image
// ============================================================================

const MAX_IMAGE_SIZE: u32 = 4096;
const SMOLVLM_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SMOLVLM_STD: [f32; 3] = [0.5, 0.5, 0.5];

struct SmolVLMImageProcessor {
    size: HashMap<String, u32>,
    max_image_size: HashMap<String, u32>,
    rescale_factor: f32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

impl Default for SmolVLMImageProcessor {
    fn default() -> Self {
        Self {
            size: HashMap::from([("longest_edge".to_string(), 2048)]),
            max_image_size: HashMap::from([("longest_edge".to_string(), 512)]),
            rescale_factor: 1.0 / 255.0,
            image_mean: SMOLVLM_MEAN,
            image_std: SMOLVLM_STD,
        }
    }
}

impl SmolVLMImageProcessor {
    fn new() -> Self {
        Self::default()
    }

    fn resize_output_size_rescale_to_max_len(
        height: u32,
        width: u32,
        min_len: Option<u32>,
        max_len: Option<u32>,
    ) -> (u32, u32) {
        let max_len = max_len.unwrap_or_else(|| height.max(width));
        let aspect_ratio = width as f32 / height as f32;

        let (mut width, mut height) = if width >= height {
            let width = max_len;
            let height = (width as f32 / aspect_ratio).round() as u32;
            (width, height)
        } else {
            let height = max_len;
            let width = (height as f32 * aspect_ratio).round() as u32;
            (width, height)
        };

        let min_len = min_len.unwrap_or(1);
        height = height.max(min_len);
        width = width.max(min_len);

        (height, width)
    }

    fn resize_output_size_scale_below_upper_bound(
        height: u32,
        width: u32,
        max_len: Option<u32>,
    ) -> (u32, u32) {
        let max_len = max_len.unwrap_or_else(|| height.max(width));
        let aspect_ratio = width as f32 / height as f32;

        let (mut width, mut height) = if width >= height && width > max_len {
            let width = max_len;
            let height = (width as f32 / aspect_ratio).round() as u32;
            (width, height)
        } else if height > width && height > max_len {
            let height = max_len;
            let width = (height as f32 * aspect_ratio).round() as u32;
            (width, height)
        } else {
            (width, height)
        };

        height = height.max(1);
        width = width.max(1);

        (height, width)
    }

    fn get_resize_output_image_size(
        image: &DynamicImage,
        resolution_max_side: u32,
    ) -> (u32, u32) {
        let (height, width) = (image.height(), image.width());

        if height <= resolution_max_side && width <= resolution_max_side {
            return (height, width);
        }

        let (height, width) = Self::resize_output_size_rescale_to_max_len(height, width, None, Some(resolution_max_side));
        let (height, width) = Self::resize_output_size_scale_below_upper_bound(height, width, Some(MAX_IMAGE_SIZE));

        (height, width)
    }

    fn resize(&self, image: DynamicImage, size: HashMap<String, u32>) -> Result<DynamicImage> {
        let (height, width) = if let Some(longest_edge) = size.get("longest_edge") {
            Self::get_resize_output_image_size(&image, *longest_edge)
        } else if let (Some(height), Some(width)) = (size.get("height"), size.get("width")) {
            (*height, *width)
        } else {
            return Err(anyhow::anyhow!("size must have 'longest_edge' or 'height' and 'width'"));
        };

        Ok(image.resize_exact(width, height, image::imageops::FilterType::Lanczos3))
    }

    fn split_image(
        &self,
        image: DynamicImage,
        max_image_size: &HashMap<String, u32>,
    ) -> Result<Vec<DynamicImage>> {
        let (height, width) = (image.height(), image.width());
        let max_size = max_image_size.get("longest_edge").unwrap_or(&512);

        let mut frames = Vec::new();

        // Always do a 4x4 split
        let num_splits_h = 4;
        let num_splits_w = 4;

        let optimal_height = (height as f32 / num_splits_h as f32).ceil() as u32;
        let optimal_width = (width as f32 / num_splits_w as f32).ceil() as u32;

        for r in 0..num_splits_h {
            for c in 0..num_splits_w {
                let start_x = c * optimal_width;
                let start_y = r * optimal_height;
                let end_x = (start_x + optimal_width).min(width);
                let end_y = (start_y + optimal_height).min(height);

                let cropped = image.crop_imm(start_x, start_y, end_x - start_x, end_y - start_y);
                let resized = self.resize(
                    cropped,
                    HashMap::from([
                        ("height".to_string(), *max_size),
                        ("width".to_string(), *max_size),
                    ]),
                )?;
                frames.push(resized);
            }
        }

        // Add the original image resized to max_size (global view)
        let resized = self.resize(
            image,
            HashMap::from([
                ("height".to_string(), *max_size),
                ("width".to_string(), *max_size),
            ]),
        )?;
        frames.push(resized);

        Ok(frames)
    }

    fn preprocess(&self, image: DynamicImage) -> Result<(Array5<f32>, Array4<i64>)> {
        // Convert to RGB
        let image: DynamicImage = image.to_rgb8().into();

        // Resize to longest_edge while preserving aspect ratio
        let image = self.resize(image, self.size.clone())?;

        // Resize to be multiples of max_image_size while preserving aspect ratio
        let max_size = self.max_image_size.get("longest_edge").unwrap_or(&512);
        let (height, width) = (image.height(), image.width());
        let aspect_ratio = width as f32 / height as f32;

        let (new_width, new_height) = if width >= height {
            let new_width = ((width as f32 / *max_size as f32).ceil() * *max_size as f32) as u32;
            let new_height = (new_width as f32 / aspect_ratio).round() as u32;
            let new_height = ((new_height as f32 / *max_size as f32).ceil() * *max_size as f32) as u32;
            (new_width, new_height)
        } else {
            let new_height = ((height as f32 / *max_size as f32).ceil() * *max_size as f32) as u32;
            let new_width = (new_height as f32 * aspect_ratio).round() as u32;
            let new_width = ((new_width as f32 / *max_size as f32).ceil() * *max_size as f32) as u32;
            (new_width, new_height)
        };

        let resized = self.resize(
            image,
            HashMap::from([
                ("height".to_string(), new_height),
                ("width".to_string(), new_width),
            ]),
        )?;

        // Split into 4x4 grid + global = 17 frames
        let frames = self.split_image(resized, &self.max_image_size)?;

        // Convert frames to arrays and normalize
        let mut processed_frames = Vec::new();
        for frame in &frames {
            let mut array = Array5::<f32>::zeros((1, 1, 3, frame.height() as usize, frame.width() as usize));

            // Note: Match the POC's coordinate handling - pixels() returns (x, y) but
            // the POC destructures as (y, x), effectively transposing the image.
            // We match this behavior for model compatibility.
            for (x, y, pixel) in frame.pixels() {
                for c in 0..3 {
                    let val = pixel[c] as f32 * self.rescale_factor;
                    let val = (val - self.image_mean[c]) / self.image_std[c];
                    array[[0, 0, c, x as usize, y as usize]] = val;
                }
            }
            processed_frames.push(array);
        }

        // Stack frames: (batch=1, num_frames=17, channels=3, height=512, width=512)
        let batch = Array5::from_shape_fn(
            (1, processed_frames.len(), 3, processed_frames[0].shape()[3], processed_frames[0].shape()[4]),
            |(_, i, c, y, x)| {
                processed_frames[i][[0, 0, c, y, x]]
            },
        );

        // Create attention mask: (batch=1, num_frames=17, height=512, width=512)
        let height = processed_frames[0].shape()[3];
        let width = processed_frames[0].shape()[4];
        let attention_mask = Array4::ones((1, processed_frames.len(), height, width));

        Ok((batch, attention_mask))
    }
}

// ============================================================================
// Prompt expansion helpers for SmolVLM grid format
// ============================================================================

fn prompt_split_image(
    image_seq_len: usize,
    image_rows: usize,
    image_cols: usize,
) -> String {
    let fake_token = "<fake_token_around_image>";
    let image_token = "<image>";
    let global_token = "<global-img>";

    let mut text = String::new();
    for n_h in 0..image_rows {
        for n_w in 0..image_cols {
            text.push_str(&format!(
                "{}<row_{}_col_{}>{}",
                fake_token,
                n_h + 1,
                n_w + 1,
                image_token.repeat(image_seq_len)
            ));
        }
        text.push('\n');
    }
    text.push_str(&format!(
        "\n{}{}{}{}",
        fake_token,
        global_token,
        image_token.repeat(image_seq_len),
        fake_token
    ));
    text
}

fn get_image_prompt_string(image_seq_len: usize) -> String {
    // SmolVLM uses 4x4 grid + global image
    prompt_split_image(image_seq_len, 4, 4)
}

// JSON structures for parsing config.json
#[derive(Deserialize)]
struct HFConfig {
    image_token_id: Option<u32>,
    text_config: Option<TextConfig>,
}

#[derive(Deserialize)]
struct TextConfig {
    num_key_value_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    head_dim: Option<usize>,
}

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
    eos_token_ids: Vec<u32>,
    image_token_id: u32,
    #[allow(dead_code)]
    max_context_length: usize,
}

pub struct VlmOnnx {
    vision_session: Session,
    embed_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
    image_processor: SmolVLMImageProcessor,
    config: SmolVLMConfig,
}

impl VlmOnnx {
    pub fn new(
        vision_model_path: &Path,
        embed_model_path: &Path,
        decoder_model_path: &Path,
        tokenizer_path: &Path,
        config_path: &Path,
    ) -> Result<Self> {
        println!("Loading SmolVLM ONNX models...");

        // Parse config.json to get model parameters
        let config_str = fs::read_to_string(config_path)
            .context("Failed to read config.json")?;
        let hf_config: HFConfig = serde_json::from_str(&config_str)
            .context("Failed to parse config.json")?;

        let text_config = hf_config.text_config.unwrap_or(TextConfig {
            num_key_value_heads: Some(3),
            num_hidden_layers: Some(30),
            head_dim: Some(64),
        });

        let num_key_value_heads = text_config.num_key_value_heads.unwrap_or(3);
        let num_hidden_layers = text_config.num_hidden_layers.unwrap_or(30);
        let head_dim = text_config.head_dim.unwrap_or(64);
        let image_token_id = hf_config.image_token_id.unwrap_or(49190);

        println!("Config: {} layers, {} kv heads, {} head_dim, image_token: {}",
            num_hidden_layers, num_key_value_heads, head_dim, image_token_id);

        // Create sessions with platform-specific execution providers
        #[cfg(target_os = "windows")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "linux")]
        let vision_session = {
            // Try CUDA first, fall back to CPU if it fails (e.g., cuDNN version mismatch)
            let cuda_result = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CUDAExecutionProvider::default().build()])?
                .commit_from_file(vision_model_path);

            match cuda_result {
                Ok(session) => {
                    println!("Vision encoder: using CUDA");
                    session
                }
                Err(e) => {
                    println!("CUDA failed for vision encoder ({}), falling back to CPU", e);
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_file(vision_model_path)
                        .context("Failed to create vision session")?
                }
            }
        };

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
        let embed_session = {
            let cuda_result = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CUDAExecutionProvider::default().build()])?
                .commit_from_file(embed_model_path);

            match cuda_result {
                Ok(session) => {
                    println!("Embed tokens: using CUDA");
                    session
                }
                Err(e) => {
                    println!("CUDA failed for embed tokens ({}), falling back to CPU", e);
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_file(embed_model_path)
                        .context("Failed to create embed session")?
                }
            }
        };

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
        let decoder_session = {
            let cuda_result = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CUDAExecutionProvider::default().build()])?
                .commit_from_file(decoder_model_path);

            match cuda_result {
                Ok(session) => {
                    println!("Decoder: using CUDA");
                    session
                }
                Err(e) => {
                    println!("CUDA failed for decoder ({}), falling back to CPU", e);
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_file(decoder_model_path)
                        .context("Failed to create decoder session")?
                }
            }
        };

        #[cfg(target_os = "macos")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let image_processor = SmolVLMImageProcessor::new();

        // Auto-detect the number of layers from decoder inputs
        // This is more reliable than config.json which may not match the actual model
        let mut max_layer = 0;
        for input in decoder_session.inputs.iter() {
            if input.name.starts_with("past_key_values.") {
                if let Some(layer_str) = input.name.split('.').nth(1) {
                    if let Ok(layer_num) = layer_str.parse::<usize>() {
                        max_layer = max_layer.max(layer_num);
                    }
                }
            }
        }
        let detected_layers = max_layer + 1;
        println!("Auto-detected {} layers from decoder model (config had {})", detected_layers, num_hidden_layers);

        // SmolVLM uses multiple EOS tokens
        let mut eos_token_ids = vec![2u32]; // Standard EOS
        if let Some(end_of_utterance_id) = tokenizer.token_to_id("<end_of_utterance>") {
            if !eos_token_ids.contains(&end_of_utterance_id) {
                eos_token_ids.push(end_of_utterance_id);
            }
        }
        if let Some(eos_s_id) = tokenizer.token_to_id("</s>") {
            if !eos_token_ids.contains(&eos_s_id) {
                eos_token_ids.push(eos_s_id);
            }
        }
        println!("EOS token IDs: {:?}", eos_token_ids);

        let config = SmolVLMConfig {
            num_key_value_heads,
            head_dim,
            num_hidden_layers: detected_layers,
            eos_token_ids,
            image_token_id,
            max_context_length: 2048,
        };

        Ok(Self {
            vision_session,
            embed_session,
            decoder_session,
            tokenizer,
            image_processor,
            config,
        })
    }

    pub fn generate(&mut self, prompt: &str, image_data: &[u8], _width: u32, _height: u32) -> Result<String> {
        // Load and preprocess image using SmolVLM image processor (4x4 grid + global)
        let image = image::load_from_memory(image_data)?;
        let (processed_image, pixel_attention_mask) = self.image_processor.preprocess(image)?;

        println!("Processed image shape: {:?}", processed_image.shape());
        println!("Pixel attention mask shape: {:?}", pixel_attention_mask.shape());

        // Run vision encoder to get image features
        let mut vision_inputs: HashMap<&str, Value> = HashMap::new();
        vision_inputs.insert("pixel_values", Value::from_array(processed_image.clone())?.into());

        let pixel_attention_mask_bool = pixel_attention_mask.map(|&x| x != 0);
        vision_inputs.insert("pixel_attention_mask", Value::from_array(pixel_attention_mask_bool)?.into());

        let vision_outputs = self.vision_session.run(vision_inputs)?;
        let image_features = vision_outputs[0].try_extract_array::<f32>()?.to_owned();

        // Vision encoder output shape: (17, 64, 576) -> reshape to (1088, 576)
        println!("Vision encoder output shape: {:?}", image_features.shape());
        let total_size = image_features.shape()[0] * image_features.shape()[1];
        let feature_dim = image_features.shape()[2];
        println!("Image features: {} tokens, {} dims", total_size, feature_dim);
        let image_features_reshaped = image_features.into_shape_with_order((total_size, feature_dim))?;

        // Expand prompt with proper grid structure
        // SmolVLM uses 64 image tokens per frame: ((512 // 16) ** 2) / (4**2) = 64
        let image_seq_len = 64;
        let image_prompt = get_image_prompt_string(image_seq_len);

        // Replace <image> in prompt with expanded grid, or prepend if not present
        let expanded_prompt = if prompt.contains("<image>") {
            prompt.replace("<image>", &image_prompt)
        } else {
            format!("{}\n{}", image_prompt, prompt)
        };
        println!("Expanded prompt: {} chars, {} image tokens", expanded_prompt.len(), total_size);

        // Tokenize
        let encoding = self.tokenizer.encode(expanded_prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>();
        let attention_mask = encoding.get_attention_mask().iter().map(|&x| x as i64).collect::<Vec<_>>();

        println!("Tokenized {} tokens", input_ids.len());

        // Initialize past key values
        let batch_size = 1;
        let mut past_key_values: HashMap<String, Array<f32, _>> = HashMap::new();
        for layer in 0..self.config.num_hidden_layers {
            let key_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).into_dyn();
            let value_array = Array::zeros((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).into_dyn();
            past_key_values.insert(format!("past_key_values.{}.key", layer), key_array);
            past_key_values.insert(format!("past_key_values.{}.value", layer), value_array);
        }

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

            // Replace image token embeddings with vision features
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

            // Check for any EOS token
            if self.config.eos_token_ids.contains(&(next_token as u32)) {
                break;
            }
        }

        // Decode
        let generated_text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;
        Ok(generated_text)
    }
}
