// Model download and management for Baseweight Canvas

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
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

    /// Validate and get model files from HuggingFace repository
    pub async fn validate_huggingface_repo(&self, repo: &str) -> Result<(String, String, u64, u64)> {
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

        // Fallback: scan for GGUF files
        self.scan_repo_for_gguf(repo).await
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

    async fn scan_repo_for_gguf(&self, repo: &str) -> Result<(String, String, u64, u64)> {
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

        for file in files {
            let path = file.path.to_lowercase();
            let size = file.size.unwrap_or(0);

            if path.ends_with(".gguf") {
                if path.contains("mmproj") || path.contains("vision") {
                    vision_file = Some((file.path, size));
                } else if language_file.is_none() {
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
            _ => Err(anyhow!("No GGUF files found in repository")),
        }
    }

    /// Generic model download from HuggingFace
    pub async fn download_from_huggingface<F>(&self, repo: &str, progress_callback: F) -> Result<()>
    where
        F: Fn(DownloadProgress) + Send + 'static,
    {
        self.ensure_models_directory().await?;

        // Validate and get file info
        let (language_file, vision_file, lang_size, vision_size) = self.validate_huggingface_repo(repo).await?;

        let is_unified = language_file == vision_file;
        let model_id = repo.replace("/", "_").replace("-", "_").to_lowercase();
        let model_dir = self.models_dir.join(&model_id);
        fs::create_dir_all(&model_dir).await?;

        // Download language model
        let lang_url = format!("https://huggingface.co/{}/resolve/main/{}", repo, language_file);
        let lang_path = model_dir.join(&language_file);

        if !lang_path.exists() {
            self.download_file(&lang_url, &lang_path, &language_file, &progress_callback).await?;
        }

        // Download vision model if separate
        if !is_unified {
            let vision_url = format!("https://huggingface.co/{}/resolve/main/{}", repo, vision_file);
            let vision_path = model_dir.join(&vision_file);

            if !vision_path.exists() {
                self.download_file(&vision_url, &vision_path, &vision_file, &progress_callback).await?;
            }
        }

        Ok(())
    }
}
