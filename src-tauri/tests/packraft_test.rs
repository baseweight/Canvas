use baseweightcanvas_lib::inference_engine::InferenceEngine;
use image::GenericImageView;

#[test]
fn test_packraft_image_inference() {
    let model_path = "/home/bowserj/.local/share/canvas/models/smolvlm2-2.2b-instruct/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf";
    let mmproj_path = "/home/bowserj/.local/share/canvas/models/smolvlm2-2.2b-instruct/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf";
    let image_path = "/home/bowserj/packraft.jpg";

    // Load the packraft image
    println!("Loading image from: {}", image_path);
    let img = image::open(image_path).expect("Failed to load packraft image");

    let (width, height) = img.dimensions();
    println!("Image dimensions: {}x{}", width, height);

    // Convert to RGB bytes
    let rgb_img = img.to_rgb8();
    let image_data: Vec<u8> = rgb_img.into_raw();
    println!("Image data size: {} bytes", image_data.len());

    // Load the model
    println!("\nLoading model...");
    let mut engine = InferenceEngine::new(model_path, mmproj_path, 999)
        .expect("Failed to load model");
    println!("Model loaded!\n");

    // Run inference
    println!("Running inference with prompt: 'What is in this image?'\n");
    let result = engine.generate_with_image(
        "What is in this image?",
        &image_data,
        width,
        height
    );

    match result {
        Ok(response) => {
            println!("\n\n=== RESPONSE ===");
            println!("{}", response);
            println!("================\n");

            // Basic sanity checks
            assert!(!response.is_empty(), "Response should not be empty");
            assert!(response.len() > 10, "Response should be more than 10 characters");

            // The response should contain actual words, not garbage
            let has_spaces = response.contains(' ');
            let has_letters = response.chars().any(|c| c.is_alphabetic());
            assert!(has_spaces || has_letters, "Response should contain readable text");

            println!("Test passed! Got coherent response of {} characters", response.len());
        }
        Err(e) => {
            panic!("Inference failed: {}", e);
        }
    }
}
