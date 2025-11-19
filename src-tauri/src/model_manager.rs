// Model download and management for Baseweight Canvas

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use futures_util::StreamExt;

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

        // Check if directory exists
        if !model_dir.exists() {
            return Ok(false);
        }

        // For SmolVLM2, check if both required files exist
        if model_id == "smolvlm2-2.2b-instruct" {
            let model_file = model_dir.join("SmolVLM2-2.2B-Instruct-Q4_K_M.gguf");
            let mmproj_file = model_dir.join("mmproj-SmolVLM2-2.2B-Instruct-f16.gguf");

            Ok(model_file.exists() && mmproj_file.exists())
        } else {
            // For other models, just check if directory exists
            Ok(true)
        }
    }

    pub async fn get_model_paths(&self, model_id: &str) -> Result<(PathBuf, PathBuf)> {
        let model_dir = self.models_dir.join(model_id);

        // For SmolVLM2, use the correct capitalized file names
        let model_file = model_dir.join("SmolVLM2-2.2B-Instruct-Q4_K_M.gguf");
        let mmproj_file = model_dir.join("mmproj-SmolVLM2-2.2B-Instruct-f16.gguf");

        if !model_file.exists() || !mmproj_file.exists() {
            return Err(anyhow!("Model files not found"));
        }

        Ok((model_file, mmproj_file))
    }

    pub async fn download_smolvlm2<F>(&self, progress_callback: F) -> Result<()>
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

            self.download_file(&url, &file_path, filename, &progress_callback).await?;
        }

        Ok(())
    }

    async fn download_file<F>(
        &self,
        url: &str,
        destination: &Path,
        filename: &str,
        progress_callback: F,
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
}
