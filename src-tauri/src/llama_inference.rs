// Simplified llama.cpp FFI bindings for Baseweight Canvas v0
// This is a minimal implementation to get multimodal inference working quickly

use anyhow::{Result, anyhow};
use std::ffi::{CString, c_char, c_void, c_int, c_float};
use std::ptr;

// Opaque types matching llama.cpp
#[repr(C)]
pub struct LlamaModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdBitmap {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdInputChunks {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdInputText {
    text: *const c_char,
    add_special: bool,
    parse_special: bool,
}

#[repr(C)]
pub struct LlamaModelParams {
    // Pointers come first
    devices: *mut c_void,
    tensor_buft_overrides: *const c_void,
    // Then the fields we care about
    n_gpu_layers: c_int,
    split_mode: c_int,
    main_gpu: c_int,
    tensor_split: *const c_float,
    progress_callback: *const c_void,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const c_void,
    // Booleans at the end
    vocab_only: bool,
    use_mmap: bool,
    use_mlock: bool,
    check_tensors: bool,
    // Padding to reach 72 bytes
    _padding: [u8; 4],
}

#[repr(C)]
pub struct LlamaContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: c_int,
    n_threads_batch: c_int,
    rope_scaling_type: c_int,
    pooling_type: c_int,
    attention_type: c_int,
    flash_attn_type: c_int,
    rope_freq_base: c_float,
    rope_freq_scale: c_float,
    yarn_ext_factor: c_float,
    yarn_attn_factor: c_float,
    yarn_beta_fast: c_float,
    yarn_beta_slow: c_float,
    yarn_orig_ctx: u32,
    defrag_thold: c_float,
    cb_eval: *const c_void,
    cb_eval_user_data: *mut c_void,
    type_k: c_int,
    type_v: c_int,
    logits_all: bool,
    embeddings: bool,
    offload_kqv: bool,
    flash_attn: bool,
    no_perf: bool,
    abort_callback: *const c_void,
    abort_callback_data: *mut c_void,
}

#[repr(C)]
pub struct MtmdContextParams {
    use_gpu: bool,
    print_timings: bool,
    n_threads: c_int,
    image_marker: *const c_char,
    media_marker: *const c_char,
    flash_attn_type: c_int,
    image_min_tokens: c_int,
    image_max_tokens: c_int,
}

pub type LlamaToken = i32;

// External C functions from llama.cpp
#[link(name = "llama")]
extern "C" {
    fn llama_model_default_params() -> LlamaModelParams;
    fn llama_model_load_from_file(path: *const c_char, params: LlamaModelParams) -> *mut LlamaModel;
    fn llama_model_free(model: *mut LlamaModel);

    fn llama_context_default_params() -> LlamaContextParams;
    fn llama_init_from_model(model: *const LlamaModel, params: LlamaContextParams) -> *mut LlamaContext;
    fn llama_free(ctx: *mut LlamaContext);

    fn llama_decode(ctx: *mut LlamaContext, batch: *const c_void) -> c_int;
    fn llama_tokenize(
        model: *const LlamaModel,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut LlamaToken,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;

    fn llama_token_to_piece(
        model: *const LlamaModel,
        token: LlamaToken,
        buf: *mut c_char,
        length: c_int,
        special: bool,
    ) -> c_int;

    fn llama_get_logits(ctx: *mut LlamaContext) -> *const c_float;
    fn llama_sample_token_greedy(ctx: *mut LlamaContext, logits: *const c_float) -> LlamaToken;
    fn llama_token_eos(model: *const LlamaModel) -> LlamaToken;
}

// External C functions from libmtmd
#[link(name = "mtmd")]
extern "C" {
    fn mtmd_context_params_default() -> MtmdContextParams;
    fn mtmd_init_from_file(
        mmproj_fname: *const c_char,
        text_model: *const LlamaModel,
        ctx_params: MtmdContextParams,
    ) -> *mut MtmdContext;
    fn mtmd_free(ctx: *mut MtmdContext);

    fn mtmd_support_vision(ctx: *const MtmdContext) -> bool;
    fn mtmd_support_audio(ctx: *const MtmdContext) -> bool;

    fn mtmd_bitmap_init(nx: u32, ny: u32, data: *const u8) -> *mut MtmdBitmap;
    fn mtmd_bitmap_free(bitmap: *mut MtmdBitmap);

    fn mtmd_input_chunks_init() -> *mut MtmdInputChunks;
    fn mtmd_input_chunks_free(chunks: *mut MtmdInputChunks);

    fn mtmd_tokenize(
        ctx: *mut MtmdContext,
        output: *mut MtmdInputChunks,
        text: *const MtmdInputText,
        bitmaps: *const *mut MtmdBitmap,
        n_bitmaps: usize,  // size_t in C
    ) -> i32;

    fn mtmd_default_marker() -> *const c_char;
}

// Safe Rust wrapper
pub struct LlamaInference {
    model: *mut LlamaModel,
    context: *mut LlamaContext,
    mtmd_ctx: Option<*mut MtmdContext>,
}

impl LlamaInference {
    pub fn new(model_path: &str, mmproj_path: Option<&str>, n_gpu_layers: i32) -> Result<Self> {
        unsafe {
            // Load model
            let model_path_c = CString::new(model_path)?;
            let mut model_params = llama_model_default_params();
            model_params.n_gpu_layers = n_gpu_layers;
            model_params.use_mmap = true;
            model_params.use_mlock = false;

            let model = llama_model_load_from_file(model_path_c.as_ptr(), model_params);
            if model.is_null() {
                return Err(anyhow!("Failed to load model from {}", model_path));
            }

            // Create context
            let mut ctx_params = llama_context_default_params();
            ctx_params.n_ctx = 4096;
            ctx_params.n_batch = 512;
            ctx_params.n_threads = 8;

            let context = llama_init_from_model(model, ctx_params);
            if context.is_null() {
                llama_model_free(model);
                return Err(anyhow!("Failed to create llama context"));
            }

            // Initialize mtmd if mmproj provided
            let mtmd_ctx = if let Some(mmproj) = mmproj_path {
                let mmproj_c = CString::new(mmproj)?;
                let mut mtmd_params = mtmd_context_params_default();
                mtmd_params.use_gpu = n_gpu_layers > 0;
                mtmd_params.n_threads = 8;

                let mtmd = mtmd_init_from_file(mmproj_c.as_ptr(), model, mtmd_params);
                if mtmd.is_null() {
                    llama_free(context);
                    llama_model_free(model);
                    return Err(anyhow!("Failed to load mmproj from {}", mmproj));
                }
                Some(mtmd)
            } else {
                None
            };

            Ok(Self {
                model,
                context,
                mtmd_ctx,
            })
        }
    }

    pub fn generate_simple(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // For v0, this is a placeholder that will be implemented
        // with proper batching, sampling, and token generation
        Ok(format!("Mock response to: {}", prompt))
    }

    /// Check if a GGUF file supports vision or audio by loading it with mtmd
    /// This is the proper way to detect multimodal capabilities, as used by llama-mtmd-cli
    pub fn check_multimodal_support(model_path: &str, mmproj_path: &str) -> Result<(bool, bool)> {
        unsafe {
            // Load the main model temporarily
            let model_path_c = CString::new(model_path)?;
            let mut model_params = llama_model_default_params();
            model_params.n_gpu_layers = 0; // CPU only for quick check
            model_params.use_mmap = true;
            model_params.use_mlock = false;

            let model = llama_model_load_from_file(model_path_c.as_ptr(), model_params);
            if model.is_null() {
                return Err(anyhow!("Failed to load model from {}", model_path));
            }

            // Try to load the mmproj file
            let mmproj_c = CString::new(mmproj_path)?;
            let mtmd_params = mtmd_context_params_default();
            let mtmd = mtmd_init_from_file(mmproj_c.as_ptr(), model, mtmd_params);

            if mtmd.is_null() {
                llama_model_free(model);
                return Err(anyhow!("Failed to load mmproj from {}", mmproj_path));
            }

            // Check what the model supports
            let supports_vision = mtmd_support_vision(mtmd);
            let supports_audio = mtmd_support_audio(mtmd);

            // Clean up
            mtmd_free(mtmd);
            llama_model_free(model);

            Ok((supports_vision, supports_audio))
        }
    }
}

impl Drop for LlamaInference {
    fn drop(&mut self) {
        unsafe {
            if let Some(mtmd) = self.mtmd_ctx {
                mtmd_free(mtmd);
            }
            if !self.context.is_null() {
                llama_free(self.context);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
        }
    }
}

unsafe impl Send for LlamaInference {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Helper to get test model paths
    fn get_test_model_dir() -> PathBuf {
        // Use the models directory from ModelManager
        crate::model_manager::ModelManager::get_models_directory()
            .unwrap_or_else(|_| PathBuf::from("test_models"))
    }

    #[test]
    #[ignore] // Requires actual model files
    fn test_check_multimodal_support_vision_model() {
        // Test with SmolVLM2 (separate mmproj file)
        let model_dir = get_test_model_dir().join("smolvlm2-2.2b-instruct");

        if !model_dir.exists() {
            eprintln!("Skipping test - model directory not found: {:?}", model_dir);
            return;
        }

        let lang_file = model_dir.join("SmolVLM2-2.2B-Instruct-Q4_K_M.gguf");
        let mmproj_file = model_dir.join("mmproj-SmolVLM2-2.2B-Instruct-f16.gguf");

        if !lang_file.exists() || !mmproj_file.exists() {
            eprintln!("Skipping test - model files not found");
            return;
        }

        let result = LlamaInference::check_multimodal_support(
            lang_file.to_str().unwrap(),
            mmproj_file.to_str().unwrap()
        );

        assert!(result.is_ok(), "Should successfully check model");
        let (supports_vision, supports_audio) = result.unwrap();
        assert!(supports_vision, "SmolVLM2 should support vision");
        assert!(!supports_audio, "SmolVLM2 should not support audio");
    }

    #[test]
    #[ignore] // Requires actual model files
    fn test_check_multimodal_support_unified_model() {
        // Test with Ministral or other unified model
        let model_dir = get_test_model_dir();

        // Look for any unified model (vision encoder embedded in main file)
        // This would be a model like Ministral-8B-Instruct-2410-Q4_K_M.gguf
        let mut unified_model_path: Option<PathBuf> = None;

        if let Ok(entries) = std::fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(files) = std::fs::read_dir(&path) {
                        for file in files.flatten() {
                            let file_path = file.path();
                            let file_name = file_path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("");

                            // Look for a single GGUF file (unified model)
                            if file_name.ends_with(".gguf")
                                && !file_name.contains("mmproj")
                                && !file_name.contains("vision") {
                                unified_model_path = Some(file_path);
                                break;
                            }
                        }
                    }
                    if unified_model_path.is_some() {
                        break;
                    }
                }
            }
        }

        if let Some(model_path) = unified_model_path {
            println!("Testing unified model: {:?}", model_path);

            // For unified models, mmproj_path is the same as model_path
            let result = LlamaInference::check_multimodal_support(
                model_path.to_str().unwrap(),
                model_path.to_str().unwrap()
            );

            assert!(result.is_ok(), "Should successfully check unified model");
            let (supports_vision, supports_audio) = result.unwrap();
            assert!(
                supports_vision || supports_audio,
                "Unified model should support at least vision or audio"
            );
        } else {
            eprintln!("Skipping test - no unified model found");
        }
    }

    #[test]
    #[ignore] // Requires actual model files
    fn test_check_multimodal_support_audio_model() {
        // Test with Ultravox or other audio model
        let model_dir = get_test_model_dir();

        // Look for audio model directory
        let audio_model_dirs = ["ultravox", "qwen2_audio"];
        let mut audio_model_path: Option<PathBuf> = None;
        let mut audio_mmproj_path: Option<PathBuf> = None;

        for dir_name in audio_model_dirs {
            let test_dir = model_dir.join(dir_name);
            if test_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&test_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let file_name = path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("");

                        if file_name.ends_with(".gguf") {
                            if file_name.contains("mmproj") || file_name.contains("audio") {
                                audio_mmproj_path = Some(path);
                            } else {
                                audio_model_path = Some(path);
                            }
                        }
                    }
                }
                if audio_model_path.is_some() {
                    break;
                }
            }
        }

        if let (Some(model_path), Some(mmproj_path)) = (audio_model_path.clone(), audio_mmproj_path.clone()) {
            println!("Testing audio model: {:?}", model_path);

            let result = LlamaInference::check_multimodal_support(
                model_path.to_str().unwrap(),
                mmproj_path.to_str().unwrap()
            );

            assert!(result.is_ok(), "Should successfully check audio model");
            let (supports_vision, supports_audio) = result.unwrap();
            assert!(supports_audio, "Audio model should support audio");
        } else if let Some(model_path) = audio_model_path {
            // Try as unified model
            println!("Testing audio model (unified): {:?}", model_path);

            let result = LlamaInference::check_multimodal_support(
                model_path.to_str().unwrap(),
                model_path.to_str().unwrap()
            );

            assert!(result.is_ok(), "Should successfully check unified audio model");
            let (supports_vision, supports_audio) = result.unwrap();
            assert!(supports_audio, "Audio model should support audio");
        } else {
            eprintln!("Skipping test - no audio model found");
        }
    }

    #[test]
    fn test_check_multimodal_support_nonexistent_files() {
        let result = LlamaInference::check_multimodal_support(
            "/nonexistent/model.gguf",
            "/nonexistent/mmproj.gguf"
        );

        assert!(result.is_err(), "Should fail with nonexistent files");
    }

    #[test]
    #[ignore] // Requires actual model files
    fn test_check_multimodal_support_text_only_model() {
        // This test would require a text-only GGUF model
        // which should fail multimodal detection
        let model_dir = get_test_model_dir();

        // Look for a text-only model (no vision/audio support)
        // These typically won't be in the models directory for this app
        // but this test documents expected behavior

        eprintln!("Note: This test requires a text-only GGUF model");
        eprintln!("Text-only models should fail with 'Failed to load mmproj' error");
    }
}
