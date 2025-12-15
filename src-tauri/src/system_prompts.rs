use std::fs;
use std::path::PathBuf;

pub struct SystemPromptManager {
    config_dir: PathBuf,
}

impl SystemPromptManager {
    pub fn new(config_dir: PathBuf) -> Result<Self, String> {
        // Create config directory if it doesn't exist
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir)
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
        }
        Ok(Self { config_dir })
    }

    pub fn load(&self, model_id: &str) -> Result<Option<String>, String> {
        let file_path = self.get_prompt_path(model_id);

        if !file_path.exists() {
            return Ok(None);
        }

        fs::read_to_string(&file_path)
            .map(|content| {
                if content.is_empty() {
                    None
                } else {
                    Some(content)
                }
            })
            .map_err(|e| format!("Failed to read system prompt: {}", e))
    }

    pub fn save(&self, model_id: &str, prompt: &str) -> Result<(), String> {
        let file_path = self.get_prompt_path(model_id);

        fs::write(&file_path, prompt)
            .map_err(|e| format!("Failed to save system prompt: {}", e))
    }

    pub fn clear(&self, model_id: &str) -> Result<(), String> {
        let file_path = self.get_prompt_path(model_id);

        if file_path.exists() {
            fs::remove_file(&file_path)
                .map_err(|e| format!("Failed to clear system prompt: {}", e))?;
        }

        Ok(())
    }

    fn get_prompt_path(&self, model_id: &str) -> PathBuf {
        // Sanitize model_id to create a valid filename
        let safe_filename = model_id.replace('/', "_").replace('\\', "_");
        self.config_dir.join(format!("{}.txt", safe_filename))
    }
}
