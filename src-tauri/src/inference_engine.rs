// VLM Inference Engine for Baseweight Canvas
// Manages model loading, image processing, and text generation

use anyhow::{Result, anyhow};
use std::ffi::{CString, c_char, c_void, c_int, c_float, c_uint};
use std::sync::Arc;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tokio::sync::Mutex;

// Re-export ChatMessage from lib for use in inference engine
pub use crate::ChatMessage;

// Opaque C types from llama.cpp
#[repr(C)]
struct LlamaModel {
    _private: [u8; 0],
}

#[repr(C)]
struct LlamaContext {
    _private: [u8; 0],
}

#[repr(C)]
struct LlamaMemory {
    _private: [u8; 0],
}

#[repr(C)]
struct MtmdContext {
    _private: [u8; 0],
}

#[repr(C)]
struct MtmdBitmap {
    _private: [u8; 0],
}

#[repr(C)]
struct MtmdInputChunks {
    _private: [u8; 0],
}

#[repr(C)]
struct MtmdInputChunk {
    _private: [u8; 0],
}

#[repr(C)]
struct MtmdInputText {
    text: *const c_char,
    add_special: bool,
    parse_special: bool,
}

// Note: These structs must match the exact size and layout from llama.cpp
// llama_model_params is 72 bytes, llama_context_params is 120 bytes

#[repr(C)]
struct LlamaModelParams {
    // Pointers come first
    devices: *mut c_void,                    // 8 bytes
    tensor_buft_overrides: *const c_void,    // 8 bytes
    // Then the fields we care about
    n_gpu_layers: c_int,                     // 4 bytes (offset 16)
    split_mode: c_int,                       // 4 bytes (offset 20)
    main_gpu: c_int,                         // 4 bytes (offset 24)
    tensor_split: *const c_float,            // 8 bytes (offset 28, but aligned to 32)
    progress_callback: *const c_void,        // 8 bytes
    progress_callback_user_data: *mut c_void, // 8 bytes
    kv_overrides: *const c_void,             // 8 bytes
    // Booleans at the end
    vocab_only: bool,
    use_mmap: bool,
    use_mlock: bool,
    check_tensors: bool,
    // Padding to reach 72 bytes
    _padding: [u8; 4],
}

#[repr(C)]
struct LlamaContextParams {
    n_ctx: u32,                   // 4 bytes
    n_batch: u32,                 // 4 bytes
    n_ubatch: u32,                // 4 bytes
    n_seq_max: u32,               // 4 bytes
    n_threads: c_int,             // 4 bytes
    n_threads_batch: c_int,       // 4 bytes
    rope_scaling_type: c_int,     // 4 bytes (enum)
    pooling_type: c_int,          // 4 bytes (enum)
    attention_type: c_int,        // 4 bytes (enum)
    flash_attn_type: c_int,       // 4 bytes (enum)
    rope_freq_base: c_float,      // 4 bytes
    rope_freq_scale: c_float,     // 4 bytes
    yarn_ext_factor: c_float,     // 4 bytes
    yarn_attn_factor: c_float,    // 4 bytes
    yarn_beta_fast: c_float,      // 4 bytes
    yarn_beta_slow: c_float,      // 4 bytes
    yarn_orig_ctx: u32,           // 4 bytes
    defrag_thold: c_float,        // 4 bytes
    cb_eval: *const c_void,       // 8 bytes
    cb_eval_user_data: *mut c_void, // 8 bytes
    type_k: c_int,                // 4 bytes (enum ggml_type)
    type_v: c_int,                // 4 bytes (enum ggml_type)
    abort_callback: *const c_void, // 8 bytes
    abort_callback_data: *mut c_void, // 8 bytes
    // Booleans at the end
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    // Padding to reach 120 bytes
    _padding: [u8; 2],
}

#[repr(C)]
struct MtmdContextParams {
    use_gpu: bool,
    print_timings: bool,
    n_threads: c_int,
    image_marker: *const c_char, // this is deprecated, do we need this?
    media_marker: *const c_char,
    flash_attn_type: c_int,  // enum llama_flash_attn_type
    warmup: bool,  // whether to run a warmup encode pass after initialization
    image_min_tokens: c_int,
    image_max_tokens: c_int,
}

type LlamaToken = i32;
type LlamaPos = i32;
type LlamaSeqId = i32;

#[repr(C)]
struct LlamaBatch {
    n_tokens: i32,
    token: *mut LlamaToken,
    embd: *mut c_float,
    pos: *mut LlamaPos,
    n_seq_id: *mut i32,
    seq_id: *mut *mut LlamaSeqId,
    logits: *mut i8,
}

#[repr(C)]
struct LlamaSampler {
    _private: [u8; 0],
}

#[repr(C)]
struct LlamaVocab {
    _private: [u8; 0],
}

#[repr(C)]
struct LlamaSamplerChainParams {
    no_perf: bool,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct LlamaChatMessage {
    role: *const c_char,
    content: *const c_char,
}

// External functions from llama.cpp
#[link(name = "llama")]
extern "C" {
    fn llama_model_default_params() -> LlamaModelParams;
    fn llama_model_load_from_file(path: *const c_char, params: LlamaModelParams) -> *mut LlamaModel;
    fn llama_model_free(model: *mut LlamaModel);

    fn llama_context_default_params() -> LlamaContextParams;
    fn llama_init_from_model(model: *const LlamaModel, params: LlamaContextParams) -> *mut LlamaContext;
    fn llama_free(ctx: *mut LlamaContext);

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
        vocab: *const LlamaVocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;

    fn llama_vocab_eos(vocab: *const LlamaVocab) -> LlamaToken;
    fn llama_vocab_eot(vocab: *const LlamaVocab) -> LlamaToken;

    // Chat template functions
    fn llama_chat_apply_template(
        tmpl: *const c_char,
        chat: *const LlamaChatMessage,
        n_msg: usize,
        add_ass: bool,
        buf: *mut c_char,
        length: c_int,
    ) -> c_int;

    // Batch functions
    fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch;
    fn llama_batch_free(batch: LlamaBatch);

    // Decode function
    fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> i32;

    // Logits functions
    fn llama_get_logits_ith(ctx: *mut LlamaContext, i: i32) -> *mut c_float;

    // Model/vocab functions
    fn llama_get_model(ctx: *const LlamaContext) -> *const LlamaModel;
    fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> i32;

    // Sampler functions
    fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);
    fn llama_sampler_init_top_k(k: i32) -> *mut LlamaSampler;
    fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut LlamaSampler;
    fn llama_sampler_init_temp(t: c_float) -> *mut LlamaSampler;
    fn llama_sampler_init_dist(seed: c_uint) -> *mut LlamaSampler;
    fn llama_sampler_sample(smpl: *mut LlamaSampler, ctx: *mut LlamaContext, idx: i32) -> LlamaToken;
    fn llama_sampler_accept(smpl: *mut LlamaSampler, token: LlamaToken);
    fn llama_sampler_free(smpl: *mut LlamaSampler);

    // Memory/KV cache functions
    fn llama_get_memory(ctx: *const LlamaContext) -> *mut LlamaMemory;
    fn llama_memory_clear(mem: *mut LlamaMemory, data: bool);
}

// External functions from libmtmd
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
    fn mtmd_bitmap_init_from_audio(n_samples: usize, data: *const c_float) -> *mut MtmdBitmap;
    fn mtmd_bitmap_is_audio(bitmap: *const MtmdBitmap) -> bool;
    fn mtmd_bitmap_free(bitmap: *mut MtmdBitmap);
    fn mtmd_support_vision(ctx: *const MtmdContext) -> bool;
    fn mtmd_support_audio(ctx: *const MtmdContext) -> bool;
    fn mtmd_get_audio_bitrate(ctx: *mut MtmdContext) -> c_int;

    fn mtmd_input_chunks_init() -> *mut MtmdInputChunks;
    fn mtmd_input_chunks_free(chunks: *mut MtmdInputChunks);
    fn mtmd_input_chunks_size(chunks: *const MtmdInputChunks) -> usize;
    fn mtmd_input_chunks_get(chunks: *const MtmdInputChunks, idx: usize) -> *const MtmdInputChunk;

    fn mtmd_input_chunk_get_type(chunk: *const MtmdInputChunk) -> c_int;
    fn mtmd_input_chunk_get_tokens_text(chunk: *const MtmdInputChunk, n_tokens_output: *mut usize) -> *const LlamaToken;
    fn mtmd_input_chunk_get_n_tokens(chunk: *const MtmdInputChunk) -> usize;
    fn mtmd_input_chunk_get_n_pos(chunk: *const MtmdInputChunk) -> LlamaPos;

    fn mtmd_tokenize(
        ctx: *mut MtmdContext,
        output: *mut MtmdInputChunks,
        text: *const MtmdInputText,
        bitmaps: *const *mut MtmdBitmap,
        n_bitmaps: usize,  // size_t in C
    ) -> i32;

    fn mtmd_default_marker() -> *const c_char;

    // Helper function that automatically handles both text and image chunks
    fn mtmd_helper_eval_chunks(
        ctx: *mut MtmdContext,
        lctx: *mut LlamaContext,
        chunks: *const MtmdInputChunks,
        n_past: LlamaPos,
        seq_id: LlamaSeqId,
        n_batch: i32,
        logits_last: bool,
        new_n_past: *mut LlamaPos,
    ) -> i32;

    fn mtmd_helper_get_n_pos(chunks: *const MtmdInputChunks) -> LlamaPos;
}

pub struct InferenceEngine {
    model: *mut LlamaModel,
    context: *mut LlamaContext,
    mtmd_ctx: *mut MtmdContext,
    // State for multi-turn conversations
    n_past: LlamaPos,
    cached_image_hash: Option<u64>,
}

// Helper function to compute hash of image data
fn compute_image_hash(image_data: &[u8], width: u32, height: u32) -> u64 {
    let mut hasher = DefaultHasher::new();
    image_data.hash(&mut hasher);
    width.hash(&mut hasher);
    height.hash(&mut hasher);
    hasher.finish()
}

unsafe impl Send for InferenceEngine {}
unsafe impl Sync for InferenceEngine {}

impl InferenceEngine {
    pub fn new(model_path: &str, mmproj_path: &str, n_gpu_layers: i32) -> Result<Self> {
        unsafe {
            println!("Loading model from: {}", model_path);

            // Load the language model
            let model_path_c = CString::new(model_path)?;
            let mut model_params = llama_model_default_params();
            model_params.n_gpu_layers = n_gpu_layers;
            model_params.use_mmap = true;
            model_params.use_mlock = false;

            let model = llama_model_load_from_file(model_path_c.as_ptr(), model_params);
            if model.is_null() {
                return Err(anyhow!("Failed to load model from {}", model_path));
            }
            println!("Model loaded successfully");

            // Create llama context
            let mut ctx_params = llama_context_default_params();
            ctx_params.n_ctx = 4096;
            ctx_params.n_batch = 512;
            ctx_params.n_ubatch = 512;
            ctx_params.n_threads = 8;
            ctx_params.n_threads_batch = 8;

            let context = llama_init_from_model(model, ctx_params);
            if context.is_null() {
                llama_model_free(model);
                return Err(anyhow!("Failed to create llama context"));
            }
            println!("Context created successfully");

            // Load multimodal projector
            println!("Loading mmproj from: {}", mmproj_path);

            // Verify mmproj file exists
            if !std::path::Path::new(mmproj_path).exists() {
                llama_free(context);
                llama_model_free(model);
                return Err(anyhow!("MMProj file does not exist: {}", mmproj_path));
            }
            println!("MMProj file verified to exist");

            let mmproj_path_c = CString::new(mmproj_path)?;
            println!("Creating mtmd context params...");
            let mut mtmd_params = mtmd_context_params_default();
            mtmd_params.use_gpu = n_gpu_layers > 0;
            mtmd_params.n_threads = 8;
            mtmd_params.print_timings = true;
            mtmd_params.warmup = true;  // Re-enable warmup to get better error messages
            // image_min_tokens and image_max_tokens are set by mtmd_context_params_default()
            // They default to -1 (auto)

            println!("Calling mtmd_init_from_file... (this may take a moment)");
            println!("  mmproj_path: {}", mmproj_path);
            println!("  use_gpu: {}", mtmd_params.use_gpu);
            println!("  n_threads: {}", mtmd_params.n_threads);

            let mtmd_ctx = mtmd_init_from_file(mmproj_path_c.as_ptr(), model, mtmd_params);

            println!("mtmd_init_from_file returned, checking result...");
            if mtmd_ctx.is_null() {
                llama_free(context);
                llama_model_free(model);
                return Err(anyhow!("Failed to load mmproj from {}", mmproj_path));
            }
            println!("MMProj loaded successfully");

            Ok(Self {
                model,
                context,
                mtmd_ctx,
                n_past: 0,
                cached_image_hash: None,
            })
        }
    }

    pub fn generate_with_image(
        &mut self,
        prompt: &str,
        image_rgb: &[u8],
        image_width: u32,
        image_height: u32,
    ) -> Result<String> {
        unsafe {
            println!("Starting inference with image {}x{}", image_width, image_height);

            // Compute hash of current image
            let current_image_hash = compute_image_hash(image_rgb, image_width, image_height);

            // Check if this is a new image or continuation
            let image_changed = match self.cached_image_hash {
                Some(cached_hash) => cached_hash != current_image_hash,
                None => true, // First call
            };

            let new_n_past: LlamaPos;

            if image_changed {
                // New image - clear KV cache and process from scratch
                println!("New image detected, clearing KV cache and processing image+text");
                let mem = llama_get_memory(self.context);
                llama_memory_clear(mem, true);
                self.n_past = 0;
                self.cached_image_hash = Some(current_image_hash);

                // Create bitmap from image
                let bitmap = mtmd_bitmap_init(image_width, image_height, image_rgb.as_ptr());
                if bitmap.is_null() {
                    return Err(anyhow!("Failed to create bitmap"));
                }

                // Get default media marker and add to prompt
                let marker = std::ffi::CStr::from_ptr(mtmd_default_marker()).to_str()?;
                let prompt_with_image = if !prompt.contains(marker) {
                    format!(" {} {}", marker, prompt)
                } else {
                    prompt.to_string()
                };

                // Apply chat template
                let role_c = CString::new("user")?;
                let content_c = CString::new(prompt_with_image)?;

                let chat_msg = LlamaChatMessage {
                    role: role_c.as_ptr(),
                    content: content_c.as_ptr(),
                };

                let mut formatted_prompt = vec![0u8; 8192];
                let n_bytes = llama_chat_apply_template(
                    std::ptr::null(),
                    &chat_msg,
                    1,
                    true,
                    formatted_prompt.as_mut_ptr() as *mut c_char,
                    formatted_prompt.len() as c_int,
                );

                if n_bytes < 0 {
                    mtmd_bitmap_free(bitmap);
                    return Err(anyhow!("Failed to apply chat template"));
                }

                formatted_prompt.truncate(n_bytes as usize);
                let formatted_str = String::from_utf8(formatted_prompt)?;
                println!("Formatted prompt: {}", formatted_str);

                let formatted_c = CString::new(formatted_str)?;

                let input_text = MtmdInputText {
                    text: formatted_c.as_ptr(),
                    add_special: true,
                    parse_special: true,
                };

                // Tokenize with image
                let chunks = mtmd_input_chunks_init();
                let bitmaps = [bitmap];
                let ret = mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    &input_text,
                    bitmaps.as_ptr(),
                    1,
                );

                if ret != 0 {
                    mtmd_bitmap_free(bitmap);
                    mtmd_input_chunks_free(chunks);
                    return Err(anyhow!("Failed to tokenize input"));
                }

                // Evaluate all chunks including image
                let mut temp_n_past: LlamaPos = 0;
                let ret = mtmd_helper_eval_chunks(
                    self.mtmd_ctx,
                    self.context,
                    chunks,
                    0,
                    0,
                    512,
                    true,
                    &mut temp_n_past,
                );

                mtmd_bitmap_free(bitmap);
                mtmd_input_chunks_free(chunks);

                if ret != 0 {
                    return Err(anyhow!("Failed to evaluate input chunks, error code: {}", ret));
                }

                new_n_past = temp_n_past;
                println!("Input chunks evaluated successfully, n_past: {}", new_n_past);
            } else {
                // Same image - only process new text (continuation)
                println!("Same image detected, continuing from n_past: {}", self.n_past);

                // Apply chat template for just the new user message
                let role_c = CString::new("user")?;
                let content_c = CString::new(prompt)?;

                let chat_msg = LlamaChatMessage {
                    role: role_c.as_ptr(),
                    content: content_c.as_ptr(),
                };

                let mut formatted_prompt = vec![0u8; 8192];
                let n_bytes = llama_chat_apply_template(
                    std::ptr::null(),
                    &chat_msg,
                    1,
                    true,
                    formatted_prompt.as_mut_ptr() as *mut c_char,
                    formatted_prompt.len() as c_int,
                );

                if n_bytes < 0 {
                    return Err(anyhow!("Failed to apply chat template"));
                }

                formatted_prompt.truncate(n_bytes as usize);
                let formatted_str = String::from_utf8(formatted_prompt)?;
                println!("Formatted prompt (continuation): {}", formatted_str);

                let formatted_c = CString::new(formatted_str)?;

                let input_text = MtmdInputText {
                    text: formatted_c.as_ptr(),
                    add_special: false, // Don't add BOS for continuation
                    parse_special: true,
                };

                // Tokenize text only (no image)
                let chunks = mtmd_input_chunks_init();
                let ret = mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    &input_text,
                    std::ptr::null(),
                    0,
                );

                if ret != 0 {
                    mtmd_input_chunks_free(chunks);
                    return Err(anyhow!("Failed to tokenize continuation input"));
                }

                // Evaluate text chunks from current position
                let mut temp_n_past: LlamaPos = 0;
                let ret = mtmd_helper_eval_chunks(
                    self.mtmd_ctx,
                    self.context,
                    chunks,
                    self.n_past,
                    0,
                    512,
                    true,
                    &mut temp_n_past,
                );

                mtmd_input_chunks_free(chunks);

                if ret != 0 {
                    return Err(anyhow!("Failed to evaluate continuation chunks, error code: {}", ret));
                }

                new_n_past = temp_n_past;
                println!("Continuation evaluated successfully, n_past: {}", new_n_past);
            }

            // Create sampler for token generation
            let sampler_params = llama_sampler_chain_default_params();
            let sampler = llama_sampler_chain_init(sampler_params);

            // Add sampling strategies
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));
            // Get vocab and EOS tokens
            let vocab = llama_model_get_vocab(self.model);
            let eos_token = llama_vocab_eos(vocab);
            let eot_token = llama_vocab_eot(vocab);

            // Generate response tokens
            let mut response = String::new();
            let max_tokens = 512;
            let mut n_generated = 0;
            let mut n_past = new_n_past;

            // Create a batch for single token generation
            let batch = llama_batch_init(1, 0, 1);

            for _ in 0..max_tokens {
                // Sample next token
                let new_token = llama_sampler_sample(sampler, self.context, -1);

                // Check for end of generation
                if new_token == eos_token || new_token == eot_token {
                    break;
                }

                // Accept the token (updates sampler state)
                llama_sampler_accept(sampler, new_token);

                // Decode token to text
                let mut token_str = vec![0u8; 256];
                let n_bytes = llama_token_to_piece(
                    vocab,
                    new_token,
                    token_str.as_mut_ptr() as *mut c_char,
                    token_str.len() as c_int,
                    0, // lstrip
                    false, // special
                );

                if n_bytes > 0 {
                    token_str.truncate(n_bytes as usize);
                    if let Ok(s) = String::from_utf8(token_str) {
                        response.push_str(&s);
                        print!("{}", s);
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                }

                n_generated += 1;

                // Prepare batch for next token
                // Update batch fields via raw pointers (batch doesn't implement Copy)
                let batch_for_decode = LlamaBatch {
                    n_tokens: 1,
                    token: batch.token,
                    embd: batch.embd,
                    pos: batch.pos,
                    n_seq_id: batch.n_seq_id,
                    seq_id: batch.seq_id,
                    logits: batch.logits,
                };

                *batch_for_decode.token = new_token;
                *batch_for_decode.pos = n_past;
                *batch_for_decode.n_seq_id = 1;
                let seq_id_ptr = *batch_for_decode.seq_id;
                *seq_id_ptr = 0;
                *batch_for_decode.logits = 1;

                // Decode next token
                let ret = llama_decode(self.context, batch_for_decode);
                if ret != 0 {
                    println!("\nWarning: Failed to decode token at position {}", n_generated);
                    break;
                }

                n_past += 1;
            }

            println!("\n\nGenerated {} tokens", n_generated);

            // Update our n_past for future calls
            self.n_past = n_past;

            // Cleanup
            llama_sampler_free(sampler);
            llama_batch_free(batch);

            Ok(response)
        }
    }

    /// Generate response with full conversation history
    pub fn generate_with_conversation(
        &mut self,
        conversation: &[ChatMessage],
        image_rgb: &[u8],
        image_width: u32,
        image_height: u32,
    ) -> Result<String> {
        unsafe {
            println!("Starting inference with {} messages", conversation.len());

            // Compute hash of current image
            let current_image_hash = compute_image_hash(image_rgb, image_width, image_height);

            // Check if this is a new image or continuation
            let image_changed = match self.cached_image_hash {
                Some(cached_hash) => cached_hash != current_image_hash,
                None => true, // First call
            };

            if image_changed {
                // New image - clear KV cache and process from scratch
                println!("New image detected, clearing KV cache");
                let mem = llama_get_memory(self.context);
                llama_memory_clear(mem, true);
                self.n_past = 0;
                self.cached_image_hash = Some(current_image_hash);
            }

            // Get the last user message (the new prompt)
            let last_message = conversation.last()
                .ok_or_else(|| anyhow!("Conversation cannot be empty"))?;

            if last_message.role != "user" {
                return Err(anyhow!("Last message must be from user"));
            }

            // For new image, add image marker to current (last) user message
            // For continuation, just use the conversation as-is
            let conversation_with_image: Vec<ChatMessage>;
            let conversation_to_use = if image_changed {
                // New image detected - add marker to the current message (last in conversation)
                let marker = std::ffi::CStr::from_ptr(mtmd_default_marker()).to_str()?;
                let last_idx = conversation.len() - 1;

                conversation_with_image = conversation.iter().enumerate().map(|(i, msg)| {
                    if i == last_idx && msg.role == "user" {
                        // Add image marker to the current user message
                        ChatMessage {
                            role: msg.role.clone(),
                            content: if msg.content.contains(marker) {
                                msg.content.clone()
                            } else {
                                format!(" {} {}", marker, msg.content)
                            }
                        }
                    } else {
                        msg.clone()
                    }
                }).collect();

                &conversation_with_image[..]
            } else {
                // Continuation - use conversation as-is
                conversation
            };

            // Convert Rust ChatMessages to C LlamaChatMessage array
            let c_messages: Vec<_> = conversation_to_use.iter()
                .map(|msg| {
                    let role_c = CString::new(msg.role.as_str()).unwrap();
                    let content_c = CString::new(msg.content.as_str()).unwrap();

                    (
                        LlamaChatMessage {
                            role: role_c.as_ptr(),
                            content: content_c.as_ptr(),
                        },
                        role_c,
                        content_c,
                    )
                })
                .collect();

            // Keep the CStrings alive
            let chat_msgs: Vec<LlamaChatMessage> = c_messages.iter().map(|(msg, _, _)| *msg).collect();

            // Apply chat template to full conversation
            let mut formatted_prompt = vec![0u8; 32768]; // Larger buffer for full conversation
            let n_bytes = llama_chat_apply_template(
                std::ptr::null(),
                chat_msgs.as_ptr(),
                chat_msgs.len(),
                true, // add_assistant (prepare for model to respond)
                formatted_prompt.as_mut_ptr() as *mut c_char,
                formatted_prompt.len() as c_int,
            );

            if n_bytes < 0 {
                return Err(anyhow!("Failed to apply chat template to conversation"));
            }

            formatted_prompt.truncate(n_bytes as usize);
            let formatted_str = String::from_utf8(formatted_prompt)?;
            println!("Formatted conversation:\n{}", formatted_str);

            let formatted_c = CString::new(formatted_str)?;

            let input_text = MtmdInputText {
                text: formatted_c.as_ptr(),
                add_special: image_changed, // Only add BOS for new image
                parse_special: true,
            };

            // Tokenize with image if new, without if continuation
            let chunks = mtmd_input_chunks_init();

            let ret = if image_changed {
                // New image - tokenize with bitmap
                let bitmap = mtmd_bitmap_init(image_width, image_height, image_rgb.as_ptr());
                if bitmap.is_null() {
                    mtmd_input_chunks_free(chunks);
                    return Err(anyhow!("Failed to create bitmap"));
                }

                let bitmaps = [bitmap];
                let result = mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    &input_text,
                    bitmaps.as_ptr(),
                    1,
                );

                mtmd_bitmap_free(bitmap);
                result
            } else {
                // Continuation - tokenize text only
                mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    &input_text,
                    std::ptr::null(),
                    0,
                )
            };

            if ret != 0 {
                mtmd_input_chunks_free(chunks);
                return Err(anyhow!("Failed to tokenize conversation"));
            }

            // Evaluate chunks
            let mut new_n_past: LlamaPos = 0;
            let eval_ret = mtmd_helper_eval_chunks(
                self.mtmd_ctx,
                self.context,
                chunks,
                self.n_past,
                0,
                512,
                true,
                &mut new_n_past,
            );

            mtmd_input_chunks_free(chunks);

            if eval_ret != 0 {
                return Err(anyhow!("Failed to evaluate chunks, error code: {}", eval_ret));
            }

            println!("Chunks evaluated successfully, n_past: {} -> {}", self.n_past, new_n_past);

            // Create sampler for token generation
            let sampler_params = llama_sampler_chain_default_params();
            let sampler = llama_sampler_chain_init(sampler_params);

            // Add sampling strategies
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

            // Get vocab and EOS tokens
            let vocab = llama_model_get_vocab(self.model);
            let eos_token = llama_vocab_eos(vocab);
            let eot_token = llama_vocab_eot(vocab);

            // Generate response tokens
            let mut response = String::new();
            let max_tokens = 512;
            let mut n_generated = 0;
            let mut n_past = new_n_past;

            // Create a batch for single token generation
            let batch = llama_batch_init(1, 0, 1);

            for _ in 0..max_tokens {
                // Sample next token
                let new_token = llama_sampler_sample(sampler, self.context, -1);

                // Check for end of generation
                if new_token == eos_token || new_token == eot_token {
                    break;
                }

                // Accept the token (updates sampler state)
                llama_sampler_accept(sampler, new_token);

                // Decode token to text
                let mut token_str = vec![0u8; 256];
                let n_bytes = llama_token_to_piece(
                    vocab,
                    new_token,
                    token_str.as_mut_ptr() as *mut c_char,
                    token_str.len() as c_int,
                    0, // lstrip
                    false, // special
                );

                if n_bytes > 0 {
                    token_str.truncate(n_bytes as usize);
                    if let Ok(s) = String::from_utf8(token_str) {
                        response.push_str(&s);
                        print!("{}", s);
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                }

                n_generated += 1;

                // Prepare batch for next token
                let batch_for_decode = LlamaBatch {
                    n_tokens: 1,
                    token: batch.token,
                    embd: batch.embd,
                    pos: batch.pos,
                    n_seq_id: batch.n_seq_id,
                    seq_id: batch.seq_id,
                    logits: batch.logits,
                };

                *batch_for_decode.token = new_token;
                *batch_for_decode.pos = n_past;
                *batch_for_decode.n_seq_id = 1;
                let seq_id_ptr = *batch_for_decode.seq_id;
                *seq_id_ptr = 0;
                *batch_for_decode.logits = 1;

                // Decode next token
                let ret = llama_decode(self.context, batch_for_decode);
                if ret != 0 {
                    println!("\nWarning: Failed to decode token at position {}", n_generated);
                    break;
                }

                n_past += 1;
            }

            println!("\n\nGenerated {} tokens", n_generated);

            // Update our n_past for future calls
            self.n_past = n_past;

            // Cleanup
            llama_sampler_free(sampler);
            llama_batch_free(batch);

            Ok(response)
        }
    }

    /// Generate response with audio input
    pub fn generate_with_audio(
        &mut self,
        prompt: &str,
        audio_samples: &[f32],
    ) -> Result<String> {
        unsafe {
            println!("Starting inference with {} audio samples", audio_samples.len());

            // Clear KV cache for audio (always treat as new)
            println!("Clearing KV cache for audio input");
            let mem = llama_get_memory(self.context);
            llama_memory_clear(mem, true);
            self.n_past = 0;
            self.cached_image_hash = None;

            // Create bitmap from audio
            let bitmap = mtmd_bitmap_init_from_audio(audio_samples.len(), audio_samples.as_ptr());
            if bitmap.is_null() {
                return Err(anyhow!("Failed to create audio bitmap"));
            }

            // Verify it's recognized as audio
            if !mtmd_bitmap_is_audio(bitmap) {
                mtmd_bitmap_free(bitmap);
                return Err(anyhow!("Bitmap not recognized as audio"));
            }

            println!("Audio bitmap created successfully");

            // Get default media marker and add to prompt
            let marker = std::ffi::CStr::from_ptr(mtmd_default_marker()).to_str()?;
            let prompt_with_audio = if !prompt.contains(marker) {
                format!(" {} {}", marker, prompt)
            } else {
                prompt.to_string()
            };

            // Apply chat template
            let role_c = CString::new("user")?;
            let content_c = CString::new(prompt_with_audio)?;

            let chat_msg = LlamaChatMessage {
                role: role_c.as_ptr(),
                content: content_c.as_ptr(),
            };

            let mut formatted_prompt = vec![0u8; 8192];
            let n_bytes = llama_chat_apply_template(
                std::ptr::null(),
                &chat_msg,
                1,
                true,
                formatted_prompt.as_mut_ptr() as *mut c_char,
                formatted_prompt.len() as c_int,
            );

            if n_bytes < 0 {
                mtmd_bitmap_free(bitmap);
                return Err(anyhow!("Failed to apply chat template"));
            }

            formatted_prompt.truncate(n_bytes as usize);
            let formatted_str = String::from_utf8(formatted_prompt)?;
            println!("Formatted prompt: {}", formatted_str);

            let formatted_c = CString::new(formatted_str)?;

            let input_text = MtmdInputText {
                text: formatted_c.as_ptr(),
                add_special: true,
                parse_special: true,
            };

            // Tokenize with audio
            let chunks = mtmd_input_chunks_init();
            let bitmaps = [bitmap];
            let ret = mtmd_tokenize(
                self.mtmd_ctx,
                chunks,
                &input_text,
                bitmaps.as_ptr(),
                1,
            );

            if ret != 0 {
                mtmd_bitmap_free(bitmap);
                mtmd_input_chunks_free(chunks);
                return Err(anyhow!("Failed to tokenize audio input"));
            }

            // Evaluate all chunks including audio
            let mut temp_n_past: LlamaPos = 0;
            let ret = mtmd_helper_eval_chunks(
                self.mtmd_ctx,
                self.context,
                chunks,
                0,
                0,
                512,
                true,
                &mut temp_n_past,
            );

            mtmd_bitmap_free(bitmap);
            mtmd_input_chunks_free(chunks);

            if ret != 0 {
                return Err(anyhow!("Failed to evaluate audio chunks, error code: {}", ret));
            }

            let new_n_past = temp_n_past;
            println!("Audio chunks evaluated successfully, n_past: {}", new_n_past);

            // Create sampler for token generation
            let sampler_params = llama_sampler_chain_default_params();
            let sampler = llama_sampler_chain_init(sampler_params);

            // Add sampling strategies
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

            // Get vocab and EOS tokens
            let vocab = llama_model_get_vocab(self.model);
            let eos_token = llama_vocab_eos(vocab);
            let eot_token = llama_vocab_eot(vocab);

            // Generate response tokens
            let mut response = String::new();
            let max_tokens = 512;
            let mut n_generated = 0;
            let mut n_past = new_n_past;

            // Create a batch for single token generation
            let batch = llama_batch_init(1, 0, 1);

            for _ in 0..max_tokens {
                // Sample next token
                let new_token = llama_sampler_sample(sampler, self.context, -1);

                // Check for end of generation
                if new_token == eos_token || new_token == eot_token {
                    break;
                }

                // Accept the token (updates sampler state)
                llama_sampler_accept(sampler, new_token);

                // Decode token to text
                let mut token_str = vec![0u8; 256];
                let n_bytes = llama_token_to_piece(
                    vocab,
                    new_token,
                    token_str.as_mut_ptr() as *mut c_char,
                    token_str.len() as c_int,
                    0, // lstrip
                    false, // special
                );

                if n_bytes > 0 {
                    token_str.truncate(n_bytes as usize);
                    if let Ok(s) = String::from_utf8(token_str) {
                        response.push_str(&s);
                        print!("{}", s);
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                }

                n_generated += 1;

                // Prepare batch for next token
                let batch_for_decode = LlamaBatch {
                    n_tokens: 1,
                    token: batch.token,
                    embd: batch.embd,
                    pos: batch.pos,
                    n_seq_id: batch.n_seq_id,
                    seq_id: batch.seq_id,
                    logits: batch.logits,
                };

                *batch_for_decode.token = new_token;
                *batch_for_decode.pos = n_past;
                *batch_for_decode.n_seq_id = 1;
                let seq_id_ptr = *batch_for_decode.seq_id;
                *seq_id_ptr = 0;
                *batch_for_decode.logits = 1;

                // Decode next token
                let ret = llama_decode(self.context, batch_for_decode);
                if ret != 0 {
                    println!("\nWarning: Failed to decode token at position {}", n_generated);
                    break;
                }

                n_past += 1;
            }

            println!("\n\nGenerated {} tokens", n_generated);

            // Update our n_past for future calls
            self.n_past = n_past;

            // Cleanup
            llama_sampler_free(sampler);
            llama_batch_free(batch);

            Ok(response)
        }
    }

    /// Check if the model supports audio input
    pub fn supports_audio(&self) -> bool {
        unsafe {
            mtmd_support_audio(self.mtmd_ctx)
        }
    }

    /// Get the expected audio sample rate (bitrate) for this model
    pub fn get_audio_sample_rate(&self) -> i32 {
        unsafe {
            mtmd_get_audio_bitrate(self.mtmd_ctx)
        }
    }

    /// Clear the conversation state to start fresh
    pub fn clear_conversation(&mut self) {
        unsafe {
            let mem = llama_get_memory(self.context);
            llama_memory_clear(mem, true);
            self.n_past = 0;
            self.cached_image_hash = None;
            println!("Conversation cleared");
        }
    }
}

impl Drop for InferenceEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.mtmd_ctx.is_null() {
                mtmd_free(self.mtmd_ctx);
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

// Global inference engine wrapped in Arc<Mutex<>> for thread-safe access
pub type SharedInferenceEngine = Arc<Mutex<Option<InferenceEngine>>>;

pub fn create_shared_engine() -> SharedInferenceEngine {
    Arc::new(Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_struct_sizes() {
        println!("\n=== FFI Struct Size Debug (Cross-Platform) ===");
        println!("sizeof(MtmdContextParams) = {}", std::mem::size_of::<MtmdContextParams>());
        println!("sizeof(MtmdInputText) = {}", std::mem::size_of::<MtmdInputText>());
        println!("sizeof(LlamaModelParams) = {}", std::mem::size_of::<LlamaModelParams>());
        println!("sizeof(LlamaContextParams) = {}", std::mem::size_of::<LlamaContextParams>());
        println!("sizeof(LlamaBatch) = {}", std::mem::size_of::<LlamaBatch>());
        println!("sizeof(bool) = {}", std::mem::size_of::<bool>());
        println!("sizeof(c_int) = {}", std::mem::size_of::<c_int>());
        println!("sizeof(*const c_char) = {}", std::mem::size_of::<*const c_char>());
        println!("==============================================\n");

        // Expected sizes based on C struct layout (64-bit):
        // MtmdContextParams: 2 bools + padding(2) + int(4) + 2 pointers(16) + 3 ints(12) = 36, aligned to 8 = 40
        // MtmdInputText: pointer(8) + 2 bools(2) + padding(6) = 16

        // These assertions help catch alignment issues across platforms
        assert!(std::mem::size_of::<*const c_char>() == 8, "Expected 64-bit pointers");
        assert!(std::mem::size_of::<c_int>() == 4, "Expected 32-bit int");
        assert!(std::mem::size_of::<bool>() == 1, "Expected 1-byte bool");
    }
}
