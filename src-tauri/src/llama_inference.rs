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
    // Simplified - add fields as needed
    n_gpu_layers: c_int,
    use_mmap: bool,
    use_mlock: bool,
}

#[repr(C)]
pub struct LlamaContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_threads: c_int,
    flash_attn: bool,
}

#[repr(C)]
pub struct MtmdContextParams {
    use_gpu: bool,
    print_timings: bool,
    n_threads: c_int,
    media_marker: *const c_char,
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
