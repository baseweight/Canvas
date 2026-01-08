// SAM3 Integration Tests
// Tests SAM3 engine loading, encoding, and segmentation

use baseweightcanvas_lib::sam3_engine::Sam3Engine;
use std::path::PathBuf;

fn get_sam3_model_paths() -> (PathBuf, PathBuf) {
    let home = std::env::var("HOME").expect("HOME not set");
    let models_dir = PathBuf::from(home).join(".local/share/ai/baseweight/Canvas/models/sam3-tracker");

    let vision_path = models_dir.join("vision_encoder_q4.onnx");
    let decoder_path = models_dir.join("prompt_encoder_mask_decoder_q4.onnx");

    (vision_path, decoder_path)
}

fn get_test_image_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home).join("packraft.jpg")
}

#[test]
fn test_sam3_model_loading() {
    let (vision_path, decoder_path) = get_sam3_model_paths();

    if !vision_path.exists() || !decoder_path.exists() {
        eprintln!("Skipping test - SAM3 models not found at {:?}", vision_path.parent());
        return;
    }

    let engine = Sam3Engine::load(&vision_path, &decoder_path);
    assert!(engine.is_ok(), "Failed to load SAM3 engine: {:?}", engine.err());

    println!("SAM3 engine loaded successfully");
}

#[test]
fn test_sam3_image_encoding() {
    let (vision_path, decoder_path) = get_sam3_model_paths();
    let image_path = get_test_image_path();

    if !vision_path.exists() || !decoder_path.exists() {
        eprintln!("Skipping test - SAM3 models not found");
        return;
    }

    if !image_path.exists() {
        eprintln!("Skipping test - test image not found at {:?}", image_path);
        return;
    }

    let mut engine = Sam3Engine::load(&vision_path, &decoder_path)
        .expect("Failed to load SAM3 engine");

    // Load test image
    let image_data = std::fs::read(&image_path).expect("Failed to read test image");

    // Encode image
    let result = engine.encode_image(&image_data);
    assert!(result.is_ok(), "Failed to encode image: {:?}", result.err());

    // Check that embeddings are cached
    assert!(engine.has_cached_embeddings(), "Embeddings should be cached after encoding");

    println!("Image encoded successfully");
}

#[test]
fn test_sam3_segmentation_with_point() {
    let (vision_path, decoder_path) = get_sam3_model_paths();
    let image_path = get_test_image_path();

    if !vision_path.exists() || !decoder_path.exists() {
        eprintln!("Skipping test - SAM3 models not found");
        return;
    }

    if !image_path.exists() {
        eprintln!("Skipping test - test image not found");
        return;
    }

    let mut engine = Sam3Engine::load(&vision_path, &decoder_path)
        .expect("Failed to load SAM3 engine");

    // Load and encode image
    let image_data = std::fs::read(&image_path).expect("Failed to read test image");
    engine.encode_image(&image_data).expect("Failed to encode image");

    // Run segmentation with a single point
    let points = vec![(400.0, 300.0)];
    let labels = vec![1i64]; // foreground
    let result = engine.segment(&points, &labels, None);

    assert!(result.is_ok(), "Failed to segment: {:?}", result.err());

    let seg_result = result.unwrap();

    // Verify we got 3 masks
    assert_eq!(seg_result.masks.len(), 3, "Should have 3 masks");

    // Verify we got 3 IoU scores
    assert_eq!(seg_result.iou_scores.len(), 3, "Should have 3 IoU scores");

    // Verify object score is reasonable
    assert!(seg_result.object_score >= 0.0 && seg_result.object_score <= 1.0,
            "Object score should be between 0 and 1");

    // Verify masks are base64 encoded PNGs
    for (i, mask) in seg_result.masks.iter().enumerate() {
        assert!(!mask.is_empty(), "Mask {} should not be empty", i);
        // Base64 PNG starts with specific bytes
        let decoded = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, mask);
        assert!(decoded.is_ok(), "Mask {} should be valid base64", i);

        let bytes = decoded.unwrap();
        // PNG magic bytes: 89 50 4E 47
        assert!(bytes.len() > 4, "Decoded mask should have PNG data");
        assert_eq!(bytes[0], 0x89, "Mask {} should start with PNG magic", i);
        assert_eq!(bytes[1], 0x50, "Mask {} should be PNG", i);
    }

    println!("Segmentation completed successfully");
    println!("  IoU scores: {:?}", seg_result.iou_scores);
    println!("  Object score: {}", seg_result.object_score);
    println!("  Dimensions: {}x{}", seg_result.width, seg_result.height);
}

#[test]
fn test_sam3_segmentation_with_box() {
    let (vision_path, decoder_path) = get_sam3_model_paths();
    let image_path = get_test_image_path();

    if !vision_path.exists() || !decoder_path.exists() {
        eprintln!("Skipping test - SAM3 models not found");
        return;
    }

    if !image_path.exists() {
        eprintln!("Skipping test - test image not found");
        return;
    }

    let mut engine = Sam3Engine::load(&vision_path, &decoder_path)
        .expect("Failed to load SAM3 engine");

    // Load and encode image
    let image_data = std::fs::read(&image_path).expect("Failed to read test image");
    engine.encode_image(&image_data).expect("Failed to encode image");

    // Run segmentation with a bounding box
    let points: Vec<(f32, f32)> = vec![];
    let labels: Vec<i64> = vec![];
    let bbox = Some((100.0, 100.0, 800.0, 800.0));

    let result = engine.segment(&points, &labels, bbox);

    assert!(result.is_ok(), "Failed to segment with box: {:?}", result.err());

    let seg_result = result.unwrap();
    assert_eq!(seg_result.masks.len(), 3, "Should have 3 masks");

    println!("Box segmentation completed");
    println!("  IoU scores: {:?}", seg_result.iou_scores);
}

#[test]
fn test_sam3_embedding_caching() {
    let (vision_path, decoder_path) = get_sam3_model_paths();
    let image_path = get_test_image_path();

    if !vision_path.exists() || !decoder_path.exists() {
        eprintln!("Skipping test - SAM3 models not found");
        return;
    }

    if !image_path.exists() {
        eprintln!("Skipping test - test image not found");
        return;
    }

    let mut engine = Sam3Engine::load(&vision_path, &decoder_path)
        .expect("Failed to load SAM3 engine");

    let image_data = std::fs::read(&image_path).expect("Failed to read test image");

    // First encoding
    let start1 = std::time::Instant::now();
    engine.encode_image(&image_data).expect("First encode failed");
    let time1 = start1.elapsed();

    // Second encoding (should use cache)
    let start2 = std::time::Instant::now();
    engine.encode_image(&image_data).expect("Second encode failed");
    let time2 = start2.elapsed();

    // Cache hit should be much faster
    assert!(time2 < time1 / 2, "Cached encoding should be faster: {:?} vs {:?}", time2, time1);

    println!("First encoding: {:?}", time1);
    println!("Cached encoding: {:?}", time2);

    // Clear cache
    engine.clear_cache();
    assert!(!engine.has_cached_embeddings(), "Cache should be cleared");
}
