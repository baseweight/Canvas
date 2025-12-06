// Model download and management for Baseweight Canvas

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use futures_util::StreamExt;
use reqwest::header::USER_AGENT;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub model_file: String,
    pub mmproj_file: String,
    pub downloaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub current: u64,
    pub total: u64,
    pub percentage: f32,
    pub file: String,
}

// SmolVLM2 2.2B Instruct bundled model info
pub fn get_bundled_model_info() -> ModelInfo {
    ModelInfo {
        id: "smolvlm2-2.2b-instruct".to_string(),
        name: "SmolVLM2 2.2B Instruct".to_string(),
        model_file: "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf".to_string(),
        mmproj_file: "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf".to_string(),
        downloaded: false,
    }
}

pub struct ModelManager {
    pub models_dir: PathBuf,
}

impl ModelManager {
    pub fn new() -> Result<Self> {
        let models_dir = Self::get_models_directory()?;
        Ok(Self { models_dir })
    }

    /// Create a ModelManager with a custom models directory (for testing)
    pub fn with_models_dir(models_dir: std::path::PathBuf) -> Self {
        Self { models_dir }
    }

    pub fn get_models_directory() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("ai", "baseweight", "Canvas")
            .ok_or_else(|| anyhow!("Failed to get project directories"))?;

        let models_dir = dirs.data_dir().join("models");
        Ok(models_dir)
    }

    pub async fn ensure_models_directory(&self) -> Result<()> {
        fs::create_dir_all(&self.models_dir).await?;
        Ok(())
    }

    pub async fn is_model_downloaded(&self, model_id: &str) -> Result<bool> {
        let model_dir = self.models_dir.join(model_id);

        if !model_dir.exists() {
            return Ok(false);
        }

        // Check if directory has at least one GGUF file
        let mut entries = tokio::fs::read_dir(&model_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if let Some(filename) = entry.file_name().to_str() {
                if filename.ends_with(".gguf") {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    pub async fn list_downloaded_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();

        if !self.models_dir.exists() {
            return Ok(models);
        }

        let mut entries = tokio::fs::read_dir(&self.models_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if let Ok(file_type) = entry.file_type().await {
                if file_type.is_dir() {
                    if let Some(model_id) = entry.file_name().to_str() {
                        // Check if this directory contains at least one GGUF file
                        if self.is_model_downloaded(model_id).await? {
                            models.push(model_id.to_string());
                        }
                    }
                }
            }
        }

        Ok(models)
    }

    pub async fn get_model_paths(&self, model_id: &str) -> Result<(PathBuf, PathBuf)> {
        let model_dir = self.models_dir.join(model_id);

        if !model_dir.exists() {
            return Err(anyhow!("Model directory not found: {}", model_id));
        }

        // Find GGUF files in directory
        let mut language_file: Option<PathBuf> = None;
        let mut mmproj_file_f16: Option<PathBuf> = None;
        let mut mmproj_file_other: Option<PathBuf> = None;

        let mut entries = tokio::fs::read_dir(&model_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                let filename_upper = filename_str.to_uppercase();
                if filename_str.ends_with(".gguf") {
                    if filename_str.contains("mmproj") || filename_str.contains("vision") {
                        // Prefer F16 mmproj files
                        if filename_upper.contains("F16") || filename_upper.contains("_F16") {
                            mmproj_file_f16 = Some(path.clone());
                        } else if mmproj_file_other.is_none() {
                            mmproj_file_other = Some(path.clone());
                        }
                    } else if language_file.is_none() {
                        language_file = Some(path.clone());
                    }
                }
            }
        }

        // Prefer F16 mmproj if available, otherwise use other quantization
        let mmproj_file = mmproj_file_f16.or(mmproj_file_other);

        match (language_file, mmproj_file) {
            (Some(lang), Some(mmproj)) => Ok((lang, mmproj)),
            (Some(lang), None) => {
                // Unified model - use same file for both
                Ok((lang.clone(), lang))
            }
            _ => Err(anyhow!("No GGUF files found in model directory")),
        }
    }

    pub async fn download_smolvlm2<F>(
        &self,
        progress_callback: F,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<()>
    where
        F: Fn(DownloadProgress) + Send + 'static,
    {
        self.ensure_models_directory().await?;

        let model_dir = self.models_dir.join("smolvlm2-2.2b-instruct");
        fs::create_dir_all(&model_dir).await?;

        // HuggingFace URLs for SmolVLM2 2.2B Instruct GGUF from ggml-org
        let files = vec![
            (
                "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
                "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf".to_string()
            ),
            (
                "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
                "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf".to_string()
            ),
        ];

        for (filename, url) in files {
            let file_path = model_dir.join(filename);

            // Skip if already downloaded
            if file_path.exists() {
                progress_callback(DownloadProgress {
                    current: 100,
                    total: 100,
                    percentage: 100.0,
                    file: format!("{} (already downloaded)", filename),
                });
                continue;
            }

            self.download_file(&url, &file_path, filename, &progress_callback, cancel_flag.clone()).await?;
        }

        Ok(())
    }

    /// Verify that downloaded model files are actually multimodal
    /// This is done by loading them with mtmd and checking the GGUF metadata
    pub async fn verify_downloaded_model(&self, model_id: &str) -> Result<(bool, bool)> {
        use crate::llama_inference::LlamaInference;

        let model_dir = self.models_dir.join(model_id);

        // Find the main model file and potential mmproj file
        let mut language_file: Option<std::path::PathBuf> = None;
        let mut mmproj_file_f16: Option<std::path::PathBuf> = None;
        let mut mmproj_file_other: Option<std::path::PathBuf> = None;

        let mut entries = tokio::fs::read_dir(&model_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                let filename_upper = filename_str.to_uppercase();
                if filename_str.ends_with(".gguf") {
                    if filename_str.contains("mmproj") || filename_str.contains("vision") {
                        // Prefer F16 mmproj files
                        if filename_upper.contains("F16") || filename_upper.contains("_F16") {
                            mmproj_file_f16 = Some(path.clone());
                        } else if mmproj_file_other.is_none() {
                            mmproj_file_other = Some(path.clone());
                        }
                    } else if language_file.is_none() {
                        language_file = Some(path.clone());
                    }
                }
            }
        }

        let lang_path = language_file.ok_or_else(|| anyhow!("No GGUF model file found"))?;

        // Prefer F16 mmproj if available, otherwise use other quantization
        let mmproj_file = mmproj_file_f16.or(mmproj_file_other);

        // Try mmproj file first, fall back to language file for unified models
        let mmproj_path = mmproj_file.as_ref().unwrap_or(&lang_path);

        // Check multimodal support by actually loading the model
        // Run in blocking task to avoid blocking the async runtime
        let lang_path_str = lang_path.to_str().unwrap().to_string();
        let mmproj_path_str = mmproj_path.to_str().unwrap().to_string();

        tokio::task::spawn_blocking(move || {
            LlamaInference::check_multimodal_support(&lang_path_str, &mmproj_path_str)
        })
        .await
        .map_err(|e| anyhow!("Verification task failed: {}", e))?
    }

    async fn download_file<F>(
        &self,
        url: &str,
        destination: &Path,
        filename: &str,
        progress_callback: F,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<()>
    where
        F: Fn(DownloadProgress),
    {
        println!("Downloading {} from {}", filename, url);

        let client = reqwest::Client::new();
        let response = client.get(url).send().await?;

        if !response.status().is_success() {
            let error_msg = format!(
                "Failed to download {}: HTTP {} from URL: {}",
                filename,
                response.status(),
                url
            );
            println!("Download error: {}", error_msg);
            return Err(anyhow!(error_msg));
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        let mut file = tokio::fs::File::create(destination).await?;

        while let Some(chunk) = stream.next().await {
            // Check for cancellation
            if cancel_flag.load(Ordering::SeqCst) {
                drop(file); // Close file handle
                // Delete partial file
                let _ = tokio::fs::remove_file(destination).await;
                println!("Download cancelled: {}", filename);
                return Err(anyhow!("Download cancelled"));
            }

            let chunk = chunk?;
            file.write_all(&chunk).await?;

            downloaded += chunk.len() as u64;

            let percentage = if total_size > 0 {
                (downloaded as f32 / total_size as f32) * 100.0
            } else {
                0.0
            };

            progress_callback(DownloadProgress {
                current: downloaded,
                total: total_size,
                percentage,
                file: filename.to_string(),
            });
        }

        file.flush().await?;
        Ok(())
    }

    /// Validate and get model files from HuggingFace repository
    pub async fn validate_huggingface_repo(&self, repo: &str, quantization: &str) -> Result<(String, String, u64, u64)> {
        // Try manifest API first (supports unified models)
        let manifest_url = format!("https://huggingface.co/v2/{}/manifests/latest", repo);
        let client = reqwest::Client::new();

        let response = client
            .get(&manifest_url)
            .header(USER_AGENT, "llama-cpp")
            .send()
            .await;

        if let Ok(resp) = response {
            if resp.status().is_success() {
                #[derive(Deserialize)]
                struct ManifestResponse {
                    #[serde(rename = "ggufFile")]
                    gguf_file: Option<String>,
                    #[serde(rename = "mmprojFile")]
                    mmproj_file: Option<String>,
                }

                if let Ok(manifest) = resp.json::<ManifestResponse>().await {
                    if let Some(language_file) = manifest.gguf_file {
                        // Strip any suffix like "(F16 mmproj)" from quantization string
                        let clean_quant = quantization.split('(').next().unwrap_or(quantization).trim();
                        // Check if this file matches the desired quantization
                        if language_file.to_uppercase().contains(&clean_quant.to_uppercase()) {
                            let vision_file = manifest.mmproj_file.unwrap_or_else(|| language_file.clone());

                            // Get file sizes
                            let lang_size = self.get_file_size(repo, &language_file).await?;
                            let vision_size = if vision_file == language_file {
                                0 // Unified model, don't count twice
                            } else {
                                self.get_file_size(repo, &vision_file).await?
                            };

                            return Ok((language_file, vision_file, lang_size, vision_size));
                        }
                    }
                }
            }
        }

        // Fallback: scan for GGUF files with specific quantization
        self.scan_repo_for_gguf(repo, quantization).await
    }

    async fn get_file_size(&self, repo: &str, filename: &str) -> Result<u64> {
        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
        let client = reqwest::Client::new();

        let response = client.head(&url).send().await?;

        if let Some(content_length) = response.headers().get("content-length") {
            Ok(content_length.to_str()?.parse()?)
        } else {
            Ok(0)
        }
    }

    async fn scan_repo_for_gguf(&self, repo: &str, quantization: &str) -> Result<(String, String, u64, u64)> {
        let api_url = format!("https://huggingface.co/api/models/{}/tree/main", repo);
        let client = reqwest::Client::new();

        #[derive(Deserialize)]
        struct FileInfo {
            path: String,
            size: Option<u64>,
        }

        let response = client.get(&api_url).send().await?;
        let files: Vec<FileInfo> = response.json().await?;

        let mut language_file: Option<(String, u64)> = None;
        let mut vision_file_f16: Option<(String, u64)> = None;
        let mut vision_file_other: Option<(String, u64)> = None;

        // Strip any suffix like "(F16 mmproj)" from quantization string
        let clean_quant = quantization.split('(').next().unwrap_or(quantization).trim();
        let quant_upper = clean_quant.to_uppercase();

        for file in files {
            let path = file.path.to_lowercase();
            let path_upper = file.path.to_uppercase();
            let size = file.size.unwrap_or(0);

            if path.ends_with(".gguf") {
                if path.contains("mmproj") || path.contains("vision") {
                    // Prefer F16 mmproj files, but keep track of other quantizations
                    if path_upper.contains("F16") || path_upper.contains("_F16") {
                        vision_file_f16 = Some((file.path, size));
                    } else if vision_file_other.is_none() {
                        vision_file_other = Some((file.path, size));
                    }
                } else if language_file.is_none() && path_upper.contains(&quant_upper) {
                    // Only select language files matching the desired quantization
                    language_file = Some((file.path, size));
                }
            }
        }

        // Prefer F16 mmproj if available, otherwise use other quantization
        let vision_file = vision_file_f16.or(vision_file_other);

        match (language_file, vision_file) {
            (Some((lang, lang_size)), Some((vis, vis_size))) => {
                Ok((lang, vis, lang_size, vis_size))
            }
            (Some((lang, lang_size)), None) => {
                // Unified model
                Ok((lang.clone(), lang, lang_size, 0))
            }
            _ => Err(anyhow!("No GGUF files found in repository with quantization: {}", quantization)),
        }
    }

    /// Generic model download from HuggingFace
    pub async fn download_from_huggingface<F>(
        &self,
        repo: &str,
        quantization: &str,
        progress_callback: F,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<()>
    where
        F: Fn(DownloadProgress) + Send + 'static,
    {
        self.ensure_models_directory().await?;

        // Validate and get file info
        let (language_file, vision_file, lang_size, vision_size) = self.validate_huggingface_repo(repo, quantization).await?;

        let is_unified = language_file == vision_file;
        let model_id = repo.replace("/", "_").replace("-", "_").to_lowercase();
        let model_dir = self.models_dir.join(&model_id);
        fs::create_dir_all(&model_dir).await?;

        // Download language model
        let lang_url = format!("https://huggingface.co/{}/resolve/main/{}", repo, language_file);
        let lang_path = model_dir.join(&language_file);

        if !lang_path.exists() {
            self.download_file(&lang_url, &lang_path, &language_file, &progress_callback, cancel_flag.clone()).await?;
        }

        // Download vision model if separate
        if !is_unified {
            let vision_url = format!("https://huggingface.co/{}/resolve/main/{}", repo, vision_file);
            let vision_path = model_dir.join(&vision_file);

            if !vision_path.exists() {
                self.download_file(&vision_url, &vision_path, &vision_file, &progress_callback, cancel_flag.clone()).await?;
            }
        }

        println!("Model downloaded successfully");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_get_models_directory() {
        let result = ModelManager::get_models_directory();
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.to_string_lossy().contains("canvas"));
        assert!(path.to_string_lossy().contains("models"));
    }

    #[tokio::test]
    async fn test_model_manager_creation() {
        let manager = ModelManager::new();
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_is_model_downloaded_nonexistent() {
        let manager = ModelManager::new().unwrap();
        let result = manager.is_model_downloaded("nonexistent-model-xyz-123").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
    }

    #[tokio::test]
    async fn test_list_downloaded_models_empty() {
        // Create a temporary directory for this test
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::with_models_dir(temp_dir.path().to_path_buf());

        let result = manager.list_downloaded_models().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_model_paths_nonexistent() {
        let manager = ModelManager::new().unwrap();
        let result = manager.get_model_paths("nonexistent-model-xyz-123").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    // Test with real HuggingFace API - marked as #[ignore] so it doesn't run by default
    // Run with: cargo test -- --ignored --test-threads=1
    #[tokio::test]
    #[ignore]
    async fn test_validate_huggingface_repo_valid() {
        let manager = ModelManager::new().unwrap();

        // Test with SmolVLM2
        let result = manager.validate_huggingface_repo("ggml-org/SmolVLM2-2.2B-Instruct-GGUF", "Q4_K_M").await;
        assert!(result.is_ok());

        let (lang_file, vision_file, lang_size, vision_size) = result.unwrap();
        assert!(lang_file.ends_with(".gguf"));
        assert!(vision_file.ends_with(".gguf"));
        assert!(lang_size > 0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_validate_huggingface_repo_invalid() {
        let manager = ModelManager::new().unwrap();

        // Test with non-existent repo
        let result = manager.validate_huggingface_repo("nonexistent/fake-model-12345", "Q4_K_M").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    #[ignore]
    async fn test_scan_repo_for_gguf_with_mmproj() {
        let manager = ModelManager::new().unwrap();

        // Test scanning a real repo with separate mmproj file
        let result = manager.scan_repo_for_gguf("ggml-org/SmolVLM2-2.2B-Instruct-GGUF", "Q4_K_M").await;
        assert!(result.is_ok());

        let (lang_file, vision_file, lang_size, vision_size) = result.unwrap();
        assert!(lang_file.ends_with(".gguf"));
        assert!(vision_file.contains("mmproj") || vision_file.contains("vision"));
        assert_ne!(lang_file, vision_file, "Should have separate files");
        assert!(lang_size > 0);
        assert!(vision_size > 0);
    }

    // Test bundled model info
    #[test]
    fn test_bundled_model_info() {
        let info = get_bundled_model_info();
        assert_eq!(info.id, "smolvlm2-2.2b-instruct");
        assert!(info.model_file.ends_with(".gguf"));
        assert!(info.mmproj_file.contains("mmproj"));
        assert!(info.mmproj_file.ends_with(".gguf"));
    }

    // Test model info structure
    #[test]
    fn test_model_info_creation() {
        let info = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            model_file: "model.gguf".to_string(),
            mmproj_file: "mmproj.gguf".to_string(),
            downloaded: false,
        };

        assert_eq!(info.id, "test-model");
        assert!(!info.downloaded);
    }

    // Test download progress structure
    #[test]
    fn test_download_progress_creation() {
        let progress = DownloadProgress {
            current: 100,
            total: 1000,
            percentage: 10.0,
            file: "test.gguf".to_string(),
        };

        assert_eq!(progress.current, 100);
        assert_eq!(progress.total, 1000);
        assert_eq!(progress.percentage, 10.0);
    }

    // Tests for verify_downloaded_model
    #[tokio::test]
    #[ignore] // Requires actual downloaded model
    async fn test_verify_downloaded_model_vision() {
        let manager = ModelManager::new().unwrap();

        // Test with SmolVLM2 if it exists
        let model_id = "smolvlm2-2.2b-instruct";
        let model_dir = manager.models_dir.join(model_id);

        if !model_dir.exists() {
            eprintln!("Skipping test - model not downloaded: {}", model_id);
            return;
        }

        let result = manager.verify_downloaded_model(model_id).await;

        assert!(result.is_ok(), "Should successfully verify SmolVLM2");
        let (supports_vision, supports_audio) = result.unwrap();
        assert!(supports_vision, "SmolVLM2 should support vision");
        assert!(!supports_audio, "SmolVLM2 should not support audio");
    }

    #[tokio::test]
    #[ignore] // Requires actual downloaded model
    async fn test_verify_downloaded_model_unified() {
        let manager = ModelManager::new().unwrap();

        // Look for any unified model (single GGUF file with embedded vision)
        let mut unified_model_id: Option<String> = None;

        if let Ok(entries) = tokio::fs::read_dir(&manager.models_dir).await {
            let mut entries = entries;
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Ok(file_type) = entry.file_type().await {
                    if file_type.is_dir() {
                        if let Some(model_id) = entry.file_name().to_str() {
                            let model_path = manager.models_dir.join(model_id);

                            // Check if this directory has a single GGUF file (unified model)
                            let mut has_main_gguf = false;
                            let mut has_mmproj = false;

                            if let Ok(files) = tokio::fs::read_dir(&model_path).await {
                                let mut files = files;
                                while let Ok(Some(file)) = files.next_entry().await {
                                    if let Some(filename) = file.file_name().to_str() {
                                        if filename.ends_with(".gguf") {
                                            if filename.contains("mmproj") || filename.contains("vision") {
                                                has_mmproj = true;
                                            } else {
                                                has_main_gguf = true;
                                            }
                                        }
                                    }
                                }
                            }

                            // Unified model has main GGUF but no separate mmproj
                            if has_main_gguf && !has_mmproj {
                                unified_model_id = Some(model_id.to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }

        if let Some(model_id) = unified_model_id {
            println!("Testing unified model: {}", model_id);

            let result = manager.verify_downloaded_model(&model_id).await;

            assert!(result.is_ok(), "Should successfully verify unified model");
            let (supports_vision, supports_audio) = result.unwrap();
            assert!(
                supports_vision || supports_audio,
                "Unified model should support at least vision or audio"
            );
        } else {
            eprintln!("Skipping test - no unified model found");
        }
    }

    #[tokio::test]
    async fn test_verify_downloaded_model_nonexistent() {
        let manager = ModelManager::new().unwrap();

        let result = manager.verify_downloaded_model("nonexistent-model-xyz-123").await;

        assert!(result.is_err(), "Should fail for nonexistent model");
    }

    #[tokio::test]
    #[ignore] // Requires actual downloaded model
    async fn test_verify_downloaded_model_audio() {
        let manager = ModelManager::new().unwrap();

        // Look for audio models
        let audio_model_patterns = ["ultravox", "qwen2_audio"];
        let mut audio_model_id: Option<String> = None;

        if let Ok(entries) = tokio::fs::read_dir(&manager.models_dir).await {
            let mut entries = entries;
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Ok(file_type) = entry.file_type().await {
                    if file_type.is_dir() {
                        if let Some(model_id) = entry.file_name().to_str() {
                            let model_id_lower = model_id.to_lowercase();
                            if audio_model_patterns.iter().any(|p| model_id_lower.contains(p)) {
                                audio_model_id = Some(model_id.to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }

        if let Some(model_id) = audio_model_id {
            println!("Testing audio model: {}", model_id);

            let result = manager.verify_downloaded_model(&model_id).await;

            assert!(result.is_ok(), "Should successfully verify audio model");
            let (supports_vision, supports_audio) = result.unwrap();
            assert!(supports_audio, "Audio model should support audio");
        } else {
            eprintln!("Skipping test - no audio model found");
        }
    }

    // Integration test for get_model_paths with multimodal detection
    #[tokio::test]
    #[ignore] // Requires actual downloaded model
    async fn test_get_model_paths_with_verification() {
        let manager = ModelManager::new().unwrap();

        // Test with SmolVLM2 if it exists
        let model_id = "smolvlm2-2.2b-instruct";
        let model_dir = manager.models_dir.join(model_id);

        if !model_dir.exists() {
            eprintln!("Skipping test - model not downloaded: {}", model_id);
            return;
        }

        // First verify the model is multimodal
        let verify_result = manager.verify_downloaded_model(model_id).await;
        assert!(verify_result.is_ok(), "Model should pass verification");
        let (supports_vision, _) = verify_result.unwrap();
        assert!(supports_vision, "SmolVLM2 should support vision");

        // Then get model paths
        let paths_result = manager.get_model_paths(model_id).await;
        assert!(paths_result.is_ok(), "Should get model paths");

        let (lang_path, mmproj_path) = paths_result.unwrap();
        assert!(lang_path.exists(), "Language model file should exist");
        assert!(mmproj_path.exists(), "MMProj file should exist");
    }
}
