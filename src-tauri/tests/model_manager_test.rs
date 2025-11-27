// Integration tests for model_manager
//
// These tests verify the end-to-end functionality of model download and management.
// Some tests are marked with #[ignore] because they make real API calls to HuggingFace.
//
// Run all tests: cargo test
// Run ignored tests: cargo test -- --ignored --test-threads=1

use baseweightcanvas_lib::model_manager::{ModelManager, get_bundled_model_info};
use tempfile::TempDir;

#[tokio::test]
async fn test_model_manager_initialization() {
    let manager = ModelManager::new();
    assert!(manager.is_ok(), "ModelManager should initialize successfully");
}

#[tokio::test]
async fn test_bundled_model_structure() {
    let model_info = get_bundled_model_info();

    assert_eq!(model_info.id, "smolvlm2-2.2b-instruct");
    assert_eq!(model_info.name, "SmolVLM2 2.2B Instruct");
    assert!(model_info.model_file.contains("SmolVLM2"));
    assert!(model_info.model_file.ends_with(".gguf"));
    assert!(model_info.mmproj_file.contains("mmproj"));
    assert!(model_info.mmproj_file.ends_with(".gguf"));
}

#[tokio::test]
async fn test_empty_models_directory() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::with_models_dir(temp_dir.path().to_path_buf());

    // List should be empty for new temp directory
    let models = manager.list_downloaded_models().await.unwrap();
    assert_eq!(models.len(), 0, "New directory should have no models");
}

#[tokio::test]
async fn test_create_models_directory() {
    let temp_dir = TempDir::new().unwrap();
    let models_path = temp_dir.path().join("models");
    let manager = ModelManager::with_models_dir(models_path.clone());

    let result = manager.ensure_models_directory().await;
    assert!(result.is_ok(), "Should create models directory");
    assert!(models_path.exists(), "Models directory should exist after ensure");
}

#[tokio::test]
async fn test_is_model_downloaded_with_temp_dir() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::with_models_dir(temp_dir.path().to_path_buf());

    // Check non-existent model
    let is_downloaded = manager.is_model_downloaded("fake-model").await.unwrap();
    assert!(!is_downloaded, "Fake model should not be downloaded");

    // Create a model directory with a GGUF file
    let model_dir = temp_dir.path().join("test-model");
    std::fs::create_dir(&model_dir).unwrap();
    std::fs::write(model_dir.join("model.gguf"), b"fake gguf content").unwrap();

    // Now it should be detected
    let is_downloaded = manager.is_model_downloaded("test-model").await.unwrap();
    assert!(is_downloaded, "Model with GGUF file should be detected");
}

#[tokio::test]
async fn test_list_downloaded_models_with_mock_data() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::with_models_dir(temp_dir.path().to_path_buf());

    // Create multiple model directories
    for model_name in &["model1", "model2", "model3"] {
        let model_dir = temp_dir.path().join(model_name);
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.gguf"), b"fake").unwrap();
    }

    // Create a directory without GGUF (should not be listed)
    let empty_dir = temp_dir.path().join("empty-model");
    std::fs::create_dir(&empty_dir).unwrap();

    let models = manager.list_downloaded_models().await.unwrap();
    assert_eq!(models.len(), 3, "Should list 3 models with GGUF files");
    assert!(models.contains(&"model1".to_string()));
    assert!(models.contains(&"model2".to_string()));
    assert!(models.contains(&"model3".to_string()));
    assert!(!models.contains(&"empty-model".to_string()));
}

#[tokio::test]
async fn test_get_model_paths_with_separate_files() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::with_models_dir(temp_dir.path().to_path_buf());

    // Create model directory with separate language and mmproj files
    let model_dir = temp_dir.path().join("test-vlm");
    std::fs::create_dir(&model_dir).unwrap();
    std::fs::write(model_dir.join("language-model.gguf"), b"fake lang").unwrap();
    std::fs::write(model_dir.join("mmproj-vision.gguf"), b"fake mmproj").unwrap();

    let result = manager.get_model_paths("test-vlm").await;
    assert!(result.is_ok(), "Should find model paths");

    let (lang_path, mmproj_path) = result.unwrap();
    assert!(lang_path.to_string_lossy().contains("language-model.gguf"));
    assert!(mmproj_path.to_string_lossy().contains("mmproj-vision.gguf"));
}

#[tokio::test]
async fn test_get_model_paths_with_unified_model() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ModelManager::with_models_dir(temp_dir.path().to_path_buf());

    // Create model directory with only one GGUF (unified model)
    let model_dir = temp_dir.path().join("test-unified");
    std::fs::create_dir(&model_dir).unwrap();
    std::fs::write(model_dir.join("unified-model.gguf"), b"fake unified").unwrap();

    let result = manager.get_model_paths("test-unified").await;
    assert!(result.is_ok(), "Should handle unified model");

    let (lang_path, mmproj_path) = result.unwrap();
    // Both should point to the same file for unified models
    assert_eq!(lang_path, mmproj_path, "Unified model should use same file for both");
}

// ===== Tests with real HuggingFace API (marked #[ignore]) =====
// Note: Private methods like verify_multimodal_model, scan_repo_for_gguf, and get_file_size
// are tested in unit tests (in model_manager.rs). Integration tests focus on the public API.

#[tokio::test]
#[ignore]
async fn integration_test_validate_valid_repo() {
    let manager = ModelManager::new().unwrap();

    // Test validation of SmolVLM2 repo via the public API
    let result = manager.validate_huggingface_repo("ggml-org/SmolVLM2-2.2B-Instruct-GGUF").await;
    assert!(result.is_ok(), "Should validate SmolVLM2 repo");

    let (lang_file, vision_file, lang_size, vision_size) = result.unwrap();

    // Verify file structure
    assert!(lang_file.ends_with(".gguf"), "Language file should be GGUF");
    assert!(vision_file.ends_with(".gguf"), "Vision file should be GGUF");
    assert!(vision_file.contains("mmproj") || vision_file.contains("vision"),
        "Vision file should contain mmproj or vision");

    // Verify sizes are reasonable
    assert!(lang_size > 1_000_000, "Language model should be > 1MB");
    assert!(vision_size > 100_000 || vision_size == 0,
        "Vision model should be > 100KB or 0 for unified");
}

#[tokio::test]
#[ignore]
async fn integration_test_validate_invalid_repo() {
    let manager = ModelManager::new().unwrap();

    // Test with completely fake repo
    let result = manager.validate_huggingface_repo("totally-fake/nonexistent-repo-xyz-12345").await;
    assert!(result.is_err(), "Should fail on non-existent repo");
}

#[tokio::test]
#[ignore]
async fn integration_test_download_bundled_model() {
    let manager = ModelManager::new().unwrap();

    // Test checking if bundled model is downloaded
    let is_downloaded = manager.is_model_downloaded("smolvlm2-2.2b-instruct").await;
    assert!(is_downloaded.is_ok(), "Should check bundled model status");
}
