// SAM3 ONNX Engine for Baseweight Canvas
// Segment Anything 3 - interactive image segmentation

use anyhow::{Result, Context};
use ndarray::{Array2, Array3, Array4, ArrayD};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use serde::Serialize;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::path::Path;

use crate::sam3_image_processor::Sam3ImageProcessor;

#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(target_os = "linux")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;

/// Cached image embeddings from the vision encoder
struct CachedEmbeddings {
    /// Hash of the input image data for cache invalidation
    image_hash: u64,
    /// Multi-scale embeddings from vision encoder
    embeddings_0: ArrayD<f32>,
    embeddings_1: ArrayD<f32>,
    embeddings_2: ArrayD<f32>,
    /// Original image dimensions
    original_size: (u32, u32),
}

/// Result from SAM3 segmentation
#[derive(Debug, Clone, Serialize)]
pub struct Sam3SegmentResult {
    /// Three masks as base64-encoded PNGs (different granularities)
    pub masks: Vec<String>,
    /// IoU confidence scores for each mask
    pub iou_scores: Vec<f32>,
    /// Object presence probability (sigmoid of object_score_logits)
    pub object_score: f32,
    /// Original image dimensions
    pub width: u32,
    pub height: u32,
}

/// SAM3 ONNX inference engine
pub struct Sam3Engine {
    vision_session: Session,
    decoder_session: Session,
    image_processor: Sam3ImageProcessor,
    cached_embeddings: Option<CachedEmbeddings>,
}

impl Sam3Engine {
    /// Load SAM3 models from disk
    pub fn load(vision_model_path: &Path, decoder_model_path: &Path) -> Result<Self> {
        println!("Loading SAM3 models...");
        println!("  Vision encoder: {:?}", vision_model_path);
        println!("  Decoder: {:?}", decoder_model_path);

        // Create vision session with platform-specific execution providers
        let vision_session = Self::create_session(vision_model_path, "vision encoder")?;

        // Create decoder session
        let decoder_session = Self::create_session(decoder_model_path, "decoder")?;

        // Print model info
        println!("Vision encoder inputs:");
        for input in &vision_session.inputs {
            println!("  {}: {:?}", input.name, input.input_type);
        }
        println!("Vision encoder outputs:");
        for output in &vision_session.outputs {
            println!("  {}: {:?}", output.name, output.output_type);
        }

        println!("Decoder inputs:");
        for input in &decoder_session.inputs {
            println!("  {}: {:?}", input.name, input.input_type);
        }
        println!("Decoder outputs:");
        for output in &decoder_session.outputs {
            println!("  {}: {:?}", output.name, output.output_type);
        }

        let image_processor = Sam3ImageProcessor::new();

        Ok(Self {
            vision_session,
            decoder_session,
            image_processor,
            cached_embeddings: None,
        })
    }

    /// Create an ONNX session with platform-specific execution providers
    fn create_session(model_path: &Path, name: &str) -> Result<Session> {
        #[cfg(target_os = "windows")]
        {
            // Try DirectML, fall back to CPU
            let dml_result = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([DirectMLExecutionProvider::default().build()])?
                .commit_from_file(model_path);

            match dml_result {
                Ok(session) => {
                    println!("  {}: using DirectML", name);
                    Ok(session)
                }
                Err(e) => {
                    println!("  DirectML failed for {} ({}), falling back to CPU", name, e);
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_file(model_path)
                        .context(format!("Failed to create {} session", name))
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            // Try CUDA first, fall back to CPU if it fails
            let cuda_result = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CUDAExecutionProvider::default().build()])?
                .commit_from_file(model_path);

            match cuda_result {
                Ok(session) => {
                    println!("  {}: using CUDA", name);
                    Ok(session)
                }
                Err(e) => {
                    println!("  CUDA failed for {} ({}), falling back to CPU", name, e);
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_file(model_path)
                        .context(format!("Failed to create {} session", name))
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Try CoreML, fall back to CPU
            let coreml_result = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CoreMLExecutionProvider::default().build()])?
                .commit_from_file(model_path);

            match coreml_result {
                Ok(session) => {
                    println!("  {}: using CoreML", name);
                    Ok(session)
                }
                Err(e) => {
                    println!("  CoreML failed for {} ({}), falling back to CPU", name, e);
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_file(model_path)
                        .context(format!("Failed to create {} session", name))
                }
            }
        }
    }

    /// Compute a hash of image data for cache invalidation
    fn compute_image_hash(image_data: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        image_data.hash(&mut hasher);
        hasher.finish()
    }

    /// Encode an image with the vision encoder
    /// Results are cached for subsequent segmentation calls
    pub fn encode_image(&mut self, image_data: &[u8]) -> Result<()> {
        let image_hash = Self::compute_image_hash(image_data);

        // Check if we already have cached embeddings for this image
        if let Some(ref cached) = self.cached_embeddings {
            if cached.image_hash == image_hash {
                println!("Using cached image embeddings");
                return Ok(());
            }
        }

        println!("Encoding image with vision encoder...");

        // Load and preprocess image
        let image = image::load_from_memory(image_data)
            .context("Failed to load image")?;
        let (pixel_values, original_size, _resized_size) = self.image_processor.preprocess(image)?;

        println!("  Original size: {:?}", original_size);
        println!("  Pixel values shape: {:?}", pixel_values.shape());

        // Run vision encoder
        let mut vision_inputs: HashMap<&str, Value> = HashMap::new();
        vision_inputs.insert("pixel_values", Value::from_array(pixel_values)?.into());

        let vision_outputs = self.vision_session.run(vision_inputs)?;

        // Extract multi-scale image embeddings
        let embeddings_0 = vision_outputs["image_embeddings.0"]
            .try_extract_array::<f32>()?
            .to_owned();
        let embeddings_1 = vision_outputs["image_embeddings.1"]
            .try_extract_array::<f32>()?
            .to_owned();
        let embeddings_2 = vision_outputs["image_embeddings.2"]
            .try_extract_array::<f32>()?
            .to_owned();

        println!("  Embeddings shapes: {:?}, {:?}, {:?}",
            embeddings_0.shape(),
            embeddings_1.shape(),
            embeddings_2.shape()
        );

        // Cache the embeddings
        self.cached_embeddings = Some(CachedEmbeddings {
            image_hash,
            embeddings_0,
            embeddings_1,
            embeddings_2,
            original_size,
        });

        println!("Image encoding complete");
        Ok(())
    }

    /// Run segmentation with the given prompts
    /// Returns 3 masks with different granularities
    pub fn segment(
        &mut self,
        points: &[(f32, f32)],
        labels: &[i64],
        bbox: Option<(f32, f32, f32, f32)>,
    ) -> Result<Sam3SegmentResult> {
        // Ensure we have cached embeddings
        let cached = self.cached_embeddings.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No image encoded. Call encode_image first."))?;

        let original_size = cached.original_size;

        // Transform points to resized image coordinates
        let transformed_points = self.image_processor.transform_points(points, original_size);

        // Prepare prompt inputs
        // input_points: [batch, 1, num_points, 2]
        let num_points = transformed_points.len().max(1);
        let mut input_points = Array4::<f32>::zeros((1, 1, num_points, 2));
        let mut input_labels = Array3::<i64>::zeros((1, 1, num_points));

        for (i, (x, y)) in transformed_points.iter().enumerate() {
            input_points[[0, 0, i, 0]] = *x;
            input_points[[0, 0, i, 1]] = *y;
        }
        for (i, &label) in labels.iter().enumerate() {
            input_labels[[0, 0, i]] = label;
        }

        // Prepare box input: [batch, num_boxes, 4]
        let input_boxes = if let Some((x1, y1, x2, y2)) = bbox {
            let transformed = self.image_processor.transform_boxes(&[(x1, y1, x2, y2)], original_size);
            let (tx1, ty1, tx2, ty2) = transformed[0];
            let mut boxes = Array3::<f32>::zeros((1, 1, 4));
            boxes[[0, 0, 0]] = tx1;
            boxes[[0, 0, 1]] = ty1;
            boxes[[0, 0, 2]] = tx2;
            boxes[[0, 0, 3]] = ty2;
            boxes
        } else {
            // No box - use empty array
            Array3::<f32>::zeros((1, 0, 4))
        };

        // Run decoder
        let mut decoder_inputs: HashMap<&str, Value> = HashMap::new();
        decoder_inputs.insert("input_points", Value::from_array(input_points)?.into());
        decoder_inputs.insert("input_labels", Value::from_array(input_labels)?.into());
        decoder_inputs.insert("input_boxes", Value::from_array(input_boxes)?.into());
        decoder_inputs.insert("image_embeddings.0", Value::from_array(cached.embeddings_0.clone())?.into());
        decoder_inputs.insert("image_embeddings.1", Value::from_array(cached.embeddings_1.clone())?.into());
        decoder_inputs.insert("image_embeddings.2", Value::from_array(cached.embeddings_2.clone())?.into());

        let decoder_outputs = self.decoder_session.run(decoder_inputs)?;

        // Extract outputs
        let iou_scores_raw = decoder_outputs["iou_scores"]
            .try_extract_array::<f32>()?
            .to_owned();
        let pred_masks = decoder_outputs["pred_masks"]
            .try_extract_array::<f32>()?
            .to_owned();
        let object_score_logits = decoder_outputs["object_score_logits"]
            .try_extract_array::<f32>()?
            .to_owned();

        // Extract IoU scores for the 3 masks
        let iou_scores = vec![
            iou_scores_raw[[0, 0, 0]],
            iou_scores_raw[[0, 0, 1]],
            iou_scores_raw[[0, 0, 2]],
        ];

        // Compute object presence probability
        let object_logit = object_score_logits[[0, 0, 0]];
        let object_score = 1.0 / (1.0 + (-object_logit).exp());

        // Extract and encode each mask
        let shape = pred_masks.shape();
        let h = shape[3];
        let w = shape[4];

        let mut masks = Vec::with_capacity(3);
        for mask_idx in 0..3 {
            // Extract mask
            let mut mask = Array2::<f32>::zeros((h, w));
            for y in 0..h {
                for x in 0..w {
                    mask[[y, x]] = pred_masks[[0, 0, mask_idx, y, x]];
                }
            }

            // Postprocess (sigmoid, threshold, resize to original)
            let mask_image = self.image_processor.postprocess_mask(&mask.view(), original_size);

            // Encode as base64 PNG
            let base64_png = self.image_processor.mask_to_base64_png(&mask_image)?;
            masks.push(base64_png);
        }

        Ok(Sam3SegmentResult {
            masks,
            iou_scores,
            object_score,
            width: original_size.0,
            height: original_size.1,
        })
    }

    /// Clear cached embeddings to free memory
    pub fn clear_cache(&mut self) {
        self.cached_embeddings = None;
    }

    /// Check if an image is currently cached
    pub fn has_cached_embeddings(&self) -> bool {
        self.cached_embeddings.is_some()
    }

    /// Get a specific mask as a grayscale image for export
    pub fn get_mask_for_export(&self, mask_index: usize) -> Result<image::GrayImage> {
        let cached = self.cached_embeddings.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No image encoded"))?;

        // We need to re-run segmentation to get the mask
        // This is a limitation - for now, we'll return an error
        // In the future, we could cache the last segmentation result
        Err(anyhow::anyhow!("Export requires re-running segmentation. Use the masks from the segment() result."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_hash() {
        let data1 = vec![1u8, 2, 3, 4, 5];
        let data2 = vec![1u8, 2, 3, 4, 5];
        let data3 = vec![1u8, 2, 3, 4, 6];

        let hash1 = Sam3Engine::compute_image_hash(&data1);
        let hash2 = Sam3Engine::compute_image_hash(&data2);
        let hash3 = Sam3Engine::compute_image_hash(&data3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
