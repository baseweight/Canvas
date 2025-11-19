use baseweightcanvas_lib::inference_engine::InferenceEngine;

#[test]
fn test_model_loading() {
    let model_path = "/home/bowserj/.local/share/canvas/models/smolvlm2-2.2b-instruct/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf";
    let mmproj_path = "/home/bowserj/.local/share/canvas/models/smolvlm2-2.2b-instruct/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf";

    println!("Loading model...");
    let engine = InferenceEngine::new(model_path, mmproj_path, 999);

    assert!(engine.is_ok(), "Model should load successfully");
    println!("Model loaded successfully!");
}

#[test]
fn test_simple_inference() {
    let model_path = "/home/bowserj/.local/share/canvas/models/smolvlm2-2.2b-instruct/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf";
    let mmproj_path = "/home/bowserj/.local/share/canvas/models/smolvlm2-2.2b-instruct/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf";

    println!("Loading model...");
    let mut engine = InferenceEngine::new(model_path, mmproj_path, 999).expect("Failed to load model");
    println!("Model loaded!");

    // Create a simple test image (100x100 blue image)
    let width = 100u32;
    let height = 100u32;
    let image_data: Vec<u8> = vec![0, 0, 255].repeat((width * height) as usize);

    println!("Running inference...");
    let result = engine.generate_with_image(
        "What color is this image?",
        &image_data,
        width,
        height
    );

    match result {
        Ok(response) => {
            println!("Response: {}", response);
            assert!(!response.is_empty(), "Response should not be empty");
        }
        Err(e) => {
            panic!("Inference failed: {}", e);
        }
    }
}
