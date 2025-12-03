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
    models_dir: PathBuf,
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
        let mut mmproj_file: Option<PathBuf> = None;

        let mut entries = tokio::fs::read_dir(&model_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                if filename_str.ends_with(".gguf") {
                    if filename_str.contains("mmproj") || filename_str.contains("vision") {
                        mmproj_file = Some(path.clone());
                    } else if language_file.is_none() {
                        language_file = Some(path.clone());
                    }
                }
            }
        }

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

    /// Verify that a model is multimodal by checking HuggingFace API metadata
    async fn verify_multimodal_model(&self, repo: &str) -> Result<()> {
        let api_url = format!("https://huggingface.co/api/models/{}", repo);
        let client = reqwest::Client::new();

        #[derive(Deserialize)]
        struct ModelInfo {
            pipeline_tag: Option<String>,
            tags: Option<Vec<String>>,
        }

        let response = client.get(&api_url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to fetch model info from HuggingFace API"));
        }

        let model_info: ModelInfo = response.json().await?;

        // Check if it's a multimodal model
        let is_multimodal = match model_info.pipeline_tag.as_deref() {
            Some("image-text-to-text") => true,
            Some("audio-text-to-text") => true,
            Some("visual-question-answering") => true,
            Some("video-text-to-text") => true,
            _ => {
                // Also check tags (case-insensitive)
                let has_multimodal_tag = model_info.tags.as_ref().map_or(false, |tags| {
                    tags.iter().any(|tag| {
                        let tag_lower = tag.to_lowercase();
                        tag_lower.contains("vision") ||
                        tag_lower.contains("multimodal") ||
                        tag_lower.contains("vlm") ||
                        tag_lower.contains("audio") ||
                        tag_lower.contains("video") ||
                        tag_lower.contains("image-text")
                    })
                });

                // Also check repo name for known indicators
                let repo_lower = repo.to_lowercase();
                let has_multimodal_name = repo_lower.contains("vlm") ||
                    repo_lower.contains("vision") ||
                    repo_lower.contains("video") ||
                    repo_lower.contains("audio") ||
                    repo_lower.contains("multimodal") ||
                    repo_lower.contains("smolvlm") ||
                    repo_lower.contains("pixtral") ||
                    repo_lower.contains("ultravox") ||
                    repo_lower.contains("moondream") ||
                    repo_lower.contains("ministral") ||
                    repo_lower.contains("llava") ||
                    repo_lower.contains("qwen") ||
                    repo_lower.contains("internvl");

                has_multimodal_tag || has_multimodal_name
            }
        };

        if !is_multimodal {
            return Err(anyhow!(
                "Model '{}' does not appear to be a multimodal (vision/audio) model. \
                Baseweight Canvas requires vision-language or audio models with mmproj support.",
                repo
            ));
        }

        Ok(())
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
        // First, check if the model is actually multimodal via HF API
        self.verify_multimodal_model(repo).await?;

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
                        // Check if this file matches the desired quantization
                        if language_file.to_uppercase().contains(&quantization.to_uppercase()) {
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
        let mut vision_file: Option<(String, u64)> = None;
        let quant_upper = quantization.to_uppercase();

        for file in files {
            let path = file.path.to_lowercase();
            let path_upper = file.path.to_uppercase();
            let size = file.size.unwrap_or(0);

            if path.ends_with(".gguf") {
                if path.contains("mmproj") || path.contains("vision") {
                    vision_file = Some((file.path, size));
                } else if language_file.is_none() && path_upper.contains(&quant_upper) {
                    // Only select language files matching the desired quantization
                    language_file = Some((file.path, size));
                }
            }
        }

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
    async fn test_verify_multimodal_model_valid_vision() {
        let manager = ModelManager::new().unwrap();

        // Test with SmolVLM2 (known vision model)
        let result = manager.verify_multimodal_model("ggml-org/SmolVLM2-2.2B-Instruct-GGUF").await;
        assert!(result.is_ok(), "SmolVLM2 should be recognized as multimodal");
    }

    #[tokio::test]
    #[ignore]
    async fn test_verify_multimodal_model_valid_audio() {
        let manager = ModelManager::new().unwrap();

        // Test with Ultravox (known audio model)
        let result = manager.verify_multimodal_model("ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF").await;
        assert!(result.is_ok(), "Ultravox should be recognized as multimodal");
    }

    #[tokio::test]
    #[ignore]
    async fn test_verify_multimodal_model_invalid_text_only() {
        let manager = ModelManager::new().unwrap();

        // Test with a text-only model (should fail)
        let result = manager.verify_multimodal_model("meta-llama/Llama-2-7b-hf").await;
        assert!(result.is_err(), "Text-only Llama model should be rejected");
        assert!(result.unwrap_err().to_string().contains("multimodal"));
    }

    #[tokio::test]
    #[ignore]
    async fn test_validate_huggingface_repo_valid() {
        let manager = ModelManager::new().unwrap();

        // Test with SmolVLM2
        let result = manager.validate_huggingface_repo("ggml-org/SmolVLM2-2.2B-Instruct-GGUF").await;
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
        let result = manager.validate_huggingface_repo("nonexistent/fake-model-12345").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    #[ignore]
    async fn test_scan_repo_for_gguf_with_mmproj() {
        let manager = ModelManager::new().unwrap();

        // Test scanning a real repo with separate mmproj file
        let result = manager.scan_repo_for_gguf("ggml-org/SmolVLM2-2.2B-Instruct-GGUF").await;
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
}
