// Simple test for VLM ONNX inference
// Run with: cargo run --example test_vlm_onnx

use std::path::Path;

// Import the vlm_onnx module from the main crate
use baseweightcanvas_lib::vlm_onnx::VlmOnnx;

fn main() -> anyhow::Result<()> {
    println!("=== SmolVLM ONNX Test ===\n");

    // Paths to the proof-of-concept models
    let base_path = Path::new("/home/bowserj/vlm/smolvlm_rust");
    let vision_path = base_path.join("onnx_models/vision_encoder_q4.onnx");
    let embed_path = base_path.join("onnx_models/embed_tokens_q4.onnx");
    let decoder_path = base_path.join("onnx_models/decoder_model_merged_q4.onnx");
    let tokenizer_path = base_path.join("tokenizer.json");

    // We need a config.json - create a minimal one if missing
    let config_path = base_path.join("config.json");
    if !config_path.exists() {
        println!("Creating minimal config.json...");
        let config = r#"{
            "image_token_id": 49190,
            "text_config": {
                "num_key_value_heads": 3,
                "num_hidden_layers": 30,
                "head_dim": 64
            }
        }"#;
        std::fs::write(&config_path, config)?;
    }

    println!("Loading models...");
    println!("  Vision: {:?}", vision_path);
    println!("  Embed: {:?}", embed_path);
    println!("  Decoder: {:?}", decoder_path);
    println!("  Tokenizer: {:?}", tokenizer_path);
    println!("  Config: {:?}", config_path);

    let mut vlm = VlmOnnx::new(
        &vision_path,
        &embed_path,
        &decoder_path,
        &tokenizer_path,
        &config_path,
    )?;

    println!("\nModels loaded successfully!\n");

    // Load test image
    let image_path = base_path.join("boat.png");
    println!("Loading test image: {:?}", image_path);
    let image_data = std::fs::read(&image_path)?;

    // Test prompt
    let prompt = "<|im_start|>User:<image>Can you describe this image?<end_of_utterance>\nAssistant:";
    println!("Prompt: {}\n", prompt);

    println!("Generating response...\n");
    let response = vlm.generate(prompt, &image_data, 0, 0)?;

    println!("\n=== Response ===");
    println!("{}", response);

    Ok(())
}
