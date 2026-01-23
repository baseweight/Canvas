use tauri::{Manager, Emitter, State};
use tauri::menu::{Menu, MenuItem, Submenu};
use tauri_plugin_dialog::DialogExt;
use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde::{Deserialize, Serialize};

mod llama_inference;
pub mod model_manager;
pub mod inference_engine;
mod audio_decoder;
pub mod vlm_onnx;
pub mod sam3_engine;
pub mod sam3_image_processor;

use model_manager::{ModelManager, DownloadProgress};
use inference_engine::{InferenceEngine, SharedInferenceEngine, create_shared_engine};
use audio_decoder::decode_audio_file;
use vlm_onnx::VlmOnnx;
use sam3_engine::{Sam3Engine, Sam3SegmentResult};

// Download cancellation state
pub type DownloadCancellation = Arc<AtomicBool>;

// Shared ONNX engine
pub type SharedOnnxEngine = Arc<tokio::sync::Mutex<Option<VlmOnnx>>>;

// Shared SAM3 engine
pub type SharedSam3Engine = Arc<tokio::sync::Mutex<Option<Sam3Engine>>>;

// Chat message for conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,     // "user" or "assistant"
    pub content: String,  // The message text
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// Check if bundled model is downloaded
#[tauri::command]
async fn is_bundled_model_downloaded() -> Result<bool, String> {
    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    manager
        .is_model_downloaded("smolvlm2-2.2b-instruct")
        .await
        .map_err(|e| e.to_string())
}

// Get the models directory path
#[tauri::command]
async fn get_models_directory() -> Result<String, String> {
    ModelManager::get_models_directory()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| e.to_string())
}

// Cancel ongoing download
#[tauri::command]
fn cancel_download(cancellation: State<'_, DownloadCancellation>) -> Result<(), String> {
    cancellation.store(true, Ordering::SeqCst);
    Ok(())
}

// Download SmolVLM2 2.2B Instruct
#[tauri::command]
async fn download_bundled_model(
    app: tauri::AppHandle,
    cancellation: State<'_, DownloadCancellation>,
) -> Result<(), String> {
    // Reset cancellation flag
    cancellation.store(false, Ordering::SeqCst);

    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    let cancel_flag = cancellation.inner().clone();

    manager
        .download_smolvlm2(
            move |progress: DownloadProgress| {
                // Emit progress to frontend
                let _ = app.emit("download-progress", &progress);
            },
            cancel_flag,
        )
        .await
        .map_err(|e| e.to_string())
}

// Get model file paths
#[tauri::command]
async fn get_model_paths(model_id: String) -> Result<(String, String), String> {
    let manager = ModelManager::new().map_err(|e| e.to_string())?;

    manager
        .get_model_paths(&model_id)
        .await
        .map(|(model, mmproj)| {
            (
                model.to_string_lossy().to_string(),
                mmproj.to_string_lossy().to_string(),
            )
        })
        .map_err(|e| e.to_string())
}

// List all downloaded models
#[tauri::command]
async fn list_downloaded_models() -> Result<Vec<String>, String> {
    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    manager.list_downloaded_models().await.map_err(|e| e.to_string())
}

// Load the inference model
#[tauri::command]
async fn load_model(
    model_id: String,
    n_gpu_layers: i32,
    engine: State<'_, SharedInferenceEngine>,
) -> Result<(), String> {
    println!("Loading model: {}", model_id);

    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    let (model_path, mmproj_path) = manager
        .get_model_paths(&model_id)
        .await
        .map_err(|e| e.to_string())?;

    let model_path_str = model_path.to_string_lossy().to_string();
    let mmproj_path_str = mmproj_path.to_string_lossy().to_string();

    println!("Model path: {}", model_path_str);
    println!("MMProj path: {}", mmproj_path_str);

    // Load the model in a blocking task to avoid blocking the async runtime
    let inference_engine = tokio::task::spawn_blocking(move || {
        InferenceEngine::new(&model_path_str, &mmproj_path_str, n_gpu_layers)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
    .map_err(|e| e.to_string())?;

    // Store the engine
    let mut engine_lock = engine.lock().await;
    *engine_lock = Some(inference_engine);

    println!("Model loaded successfully");
    Ok(())
}

// Run inference with an image and conversation history
#[tauri::command]
async fn generate_response(
    conversation: Vec<ChatMessage>,
    image_data: Vec<u8>,
    image_width: u32,
    image_height: u32,
    engine: State<'_, SharedInferenceEngine>,
) -> Result<String, String> {
    println!("Generating response for conversation with {} messages", conversation.len());

    // Take ownership of the engine temporarily
    let mut engine_lock = engine.lock().await;
    let mut engine_opt = engine_lock.take();

    if engine_opt.is_none() {
        return Err("Model not loaded".to_string());
    }

    // Drop the lock before the blocking operation
    drop(engine_lock);

    // Run inference in a blocking task
    let (response, engine_instance) = tokio::task::spawn_blocking(move || {
        let mut eng = engine_opt.take().unwrap();
        let result = eng.generate_with_conversation(&conversation, &image_data, image_width, image_height);
        (result, eng)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    // Put the engine back
    let mut engine_lock = engine.lock().await;
    *engine_lock = Some(engine_instance);

    response.map_err(|e| e.to_string())
}

// Check if the current model supports audio
#[tauri::command]
async fn check_audio_support(
    engine: State<'_, SharedInferenceEngine>,
) -> Result<bool, String> {
    let engine_lock = engine.lock().await;

    if let Some(eng) = engine_lock.as_ref() {
        Ok(eng.supports_audio())
    } else {
        Err("Model not loaded".to_string())
    }
}

// Download a model from HuggingFace
#[tauri::command]
async fn download_model(
    repo: String,
    quantization: String,
    app: tauri::AppHandle,
    cancellation: State<'_, DownloadCancellation>,
) -> Result<String, String> {
    // Reset cancellation flag
    cancellation.store(false, Ordering::SeqCst);

    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    let cancel_flag = cancellation.inner().clone();

    // Convert repo to model_id
    let model_id = repo.replace("/", "_").replace("-", "_").to_lowercase();

    manager
        .download_from_huggingface(
            &repo,
            &quantization,
            move |progress: DownloadProgress| {
                // Emit progress to frontend
                let _ = app.emit("download-progress", &progress);
            },
            cancel_flag,
        )
        .await
        .map_err(|e| e.to_string())?;

    Ok(model_id)
}

// Run inference with an audio file
#[tauri::command]
async fn generate_response_audio(
    prompt: String,
    audio_path: String,
    engine: State<'_, SharedInferenceEngine>,
) -> Result<String, String> {
    println!("Generating response for audio file: {}", audio_path);

    // Decode audio file
    let audio_data = tokio::task::spawn_blocking(move || {
        decode_audio_file(&audio_path)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
    .map_err(|e| format!("Failed to decode audio: {}", e))?;

    println!("Audio decoded: {} samples at {} Hz",
             audio_data.samples.len(), audio_data.sample_rate);

    // Take ownership of the engine temporarily
    let mut engine_lock = engine.lock().await;
    let mut engine_opt = engine_lock.take();

    if engine_opt.is_none() {
        return Err("Model not loaded".to_string());
    }

    // Get the expected sample rate from the model
    let target_sample_rate = engine_opt.as_ref().unwrap().get_audio_sample_rate() as u32;
    println!("Model expects audio at {} Hz", target_sample_rate);

    // Drop the lock before the blocking operation
    drop(engine_lock);

    // Resample audio if needed and run inference in a blocking task
    let (response, engine_instance) = tokio::task::spawn_blocking(move || {
        let mut eng = engine_opt.take().unwrap();

        // Resample audio to target sample rate
        use crate::audio_decoder::resample;
        let resampled_samples = resample(&audio_data.samples, audio_data.sample_rate, target_sample_rate);
        println!("Resampled from {} Hz to {} Hz: {} -> {} samples",
                 audio_data.sample_rate, target_sample_rate,
                 audio_data.samples.len(), resampled_samples.len());

        let result = eng.generate_with_audio(&prompt, &resampled_samples);
        (result, eng)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    // Put the engine back
    let mut engine_lock = engine.lock().await;
    *engine_lock = Some(engine_instance);

    response.map_err(|e| e.to_string())
}

// Download ONNX model from HuggingFace
#[tauri::command]
async fn download_onnx_model(
    repo: String,
    quantization: String,
    app: tauri::AppHandle,
    cancellation: State<'_, DownloadCancellation>,
) -> Result<String, String> {
    cancellation.store(false, Ordering::SeqCst);

    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    let cancel_flag = cancellation.inner().clone();

    let model_id = manager
        .download_smolvlm_onnx(
            &repo,
            &quantization,
            move |progress: DownloadProgress| {
                let _ = app.emit("download-progress", &progress);
            },
            cancel_flag,
        )
        .await
        .map_err(|e| e.to_string())?;

    Ok(model_id)
}

// Load ONNX model
#[tauri::command]
async fn load_onnx_model(
    model_id: String,
    onnx_engine: State<'_, SharedOnnxEngine>,
) -> Result<(), String> {
    println!("Loading ONNX model: {}", model_id);

    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    let (vision_path, embed_path, decoder_path, tokenizer_path, config_path) = manager
        .get_onnx_model_paths(&model_id)
        .await
        .map_err(|e| e.to_string())?;

    println!("Vision: {:?}", vision_path);
    println!("Embed: {:?}", embed_path);
    println!("Decoder: {:?}", decoder_path);
    println!("Tokenizer: {:?}", tokenizer_path);
    println!("Config: {:?}", config_path);

    // Load the ONNX model in a blocking task
    let engine = tokio::task::spawn_blocking(move || {
        VlmOnnx::new(&vision_path, &embed_path, &decoder_path, &tokenizer_path, &config_path)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
    .map_err(|e| e.to_string())?;

    // Store the engine
    let mut engine_lock = onnx_engine.lock().await;
    *engine_lock = Some(engine);

    println!("ONNX model loaded successfully");
    Ok(())
}

// Generate response with ONNX model
#[tauri::command]
async fn generate_onnx_response(
    prompt: String,
    image_data: Vec<u8>,
    image_width: u32,
    image_height: u32,
    onnx_engine: State<'_, SharedOnnxEngine>,
) -> Result<String, String> {
    println!("Generating ONNX response");

    let mut engine_lock = onnx_engine.lock().await;
    let mut engine_opt = engine_lock.take();

    if engine_opt.is_none() {
        return Err("ONNX model not loaded".to_string());
    }

    drop(engine_lock);

    // Run inference in a blocking task
    let (response, engine_instance) = tokio::task::spawn_blocking(move || {
        let mut eng = engine_opt.take().unwrap();
        let result = eng.generate(&prompt, &image_data, image_width, image_height);
        (result, eng)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    // Put the engine back
    let mut engine_lock = onnx_engine.lock().await;
    *engine_lock = Some(engine_instance);

    response.map_err(|e| e.to_string())
}

// Check if ONNX model is downloaded
#[tauri::command]
async fn is_onnx_model_downloaded(model_id: String) -> Result<bool, String> {
    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    manager
        .is_onnx_model_downloaded(&model_id)
        .await
        .map_err(|e| e.to_string())
}

// ============================================================================
// SAM3 Commands
// ============================================================================

// Load SAM3 models
#[tauri::command]
async fn sam3_load(
    sam3_engine: State<'_, SharedSam3Engine>,
) -> Result<(), String> {
    println!("Loading SAM3 models...");

    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    let (vision_path, decoder_path) = manager
        .get_sam3_model_paths()
        .await
        .map_err(|e| e.to_string())?;

    println!("SAM3 Vision: {:?}", vision_path);
    println!("SAM3 Decoder: {:?}", decoder_path);

    // Load the SAM3 model in a blocking task
    let engine = tokio::task::spawn_blocking(move || {
        Sam3Engine::load(&vision_path, &decoder_path)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
    .map_err(|e| e.to_string())?;

    // Store the engine
    let mut engine_lock = sam3_engine.lock().await;
    *engine_lock = Some(engine);

    println!("SAM3 loaded successfully");
    Ok(())
}

// Unload SAM3 models to free GPU memory
#[tauri::command]
async fn sam3_unload(
    sam3_engine: State<'_, SharedSam3Engine>,
) -> Result<(), String> {
    println!("Unloading SAM3...");

    let mut engine_lock = sam3_engine.lock().await;
    *engine_lock = None;

    println!("SAM3 unloaded");
    Ok(())
}

// Check if SAM3 is loaded
#[tauri::command]
async fn sam3_is_loaded(
    sam3_engine: State<'_, SharedSam3Engine>,
) -> Result<bool, String> {
    let engine_lock = sam3_engine.lock().await;
    Ok(engine_lock.is_some())
}

// Encode an image with SAM3 vision encoder
#[tauri::command]
async fn sam3_encode_image(
    image_data: Vec<u8>,
    sam3_engine: State<'_, SharedSam3Engine>,
) -> Result<(), String> {
    println!("Encoding image with SAM3...");

    let mut engine_lock = sam3_engine.lock().await;
    let engine = engine_lock.as_mut()
        .ok_or_else(|| "SAM3 not loaded".to_string())?;

    // Clone the image data for the blocking task
    let image_data_clone = image_data.clone();

    // We need to work around the borrow checker by taking ownership
    let mut engine_taken = engine_lock.take()
        .ok_or_else(|| "SAM3 not loaded".to_string())?;
    drop(engine_lock);

    // Run encoding in a blocking task
    let result = tokio::task::spawn_blocking(move || {
        engine_taken.encode_image(&image_data_clone)?;
        Ok::<_, anyhow::Error>(engine_taken)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    // Put the engine back
    let engine_back = result.map_err(|e| e.to_string())?;
    let mut engine_lock = sam3_engine.lock().await;
    *engine_lock = Some(engine_back);

    println!("Image encoded");
    Ok(())
}

// Run SAM3 segmentation with prompts
#[tauri::command]
async fn sam3_segment(
    points: Vec<(f32, f32)>,
    labels: Vec<i64>,
    bbox: Option<(f32, f32, f32, f32)>,
    sam3_engine: State<'_, SharedSam3Engine>,
) -> Result<Sam3SegmentResult, String> {
    let mut engine_lock = sam3_engine.lock().await;
    let engine = engine_lock.as_mut()
        .ok_or_else(|| "SAM3 not loaded".to_string())?;

    // Run segmentation (fast, can run in async context)
    engine.segment(&points, &labels, bbox)
        .map_err(|e| e.to_string())
}

// Check if SAM3 model is downloaded
#[tauri::command]
async fn sam3_is_model_downloaded() -> Result<bool, String> {
    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    manager
        .is_sam3_downloaded()
        .await
        .map_err(|e| e.to_string())
}

// Download SAM3 model
#[tauri::command]
async fn sam3_download_model(
    app: tauri::AppHandle,
    cancellation: State<'_, DownloadCancellation>,
) -> Result<(), String> {
    cancellation.store(false, Ordering::SeqCst);

    let manager = ModelManager::new().map_err(|e| e.to_string())?;
    let cancel_flag = cancellation.inner().clone();

    manager
        .download_sam3_model(
            move |progress: DownloadProgress| {
                let _ = app.emit("download-progress", &progress);
            },
            cancel_flag,
        )
        .await
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(create_shared_engine())
        .manage(Arc::new(tokio::sync::Mutex::new(None::<VlmOnnx>))) // ONNX engine
        .manage(Arc::new(tokio::sync::Mutex::new(None::<Sam3Engine>))) // SAM3 engine
        .manage(Arc::new(AtomicBool::new(false))) // Download cancellation flag
        .invoke_handler(tauri::generate_handler![
            greet,
            is_bundled_model_downloaded,
            get_models_directory,
            download_bundled_model,
            download_model,
            cancel_download,
            get_model_paths,
            list_downloaded_models,
            load_model,
            generate_response,
            check_audio_support,
            generate_response_audio,
            download_onnx_model,
            load_onnx_model,
            generate_onnx_response,
            is_onnx_model_downloaded,
            // SAM3 commands
            sam3_load,
            sam3_unload,
            sam3_is_loaded,
            sam3_encode_image,
            sam3_segment,
            sam3_is_model_downloaded,
            sam3_download_model
        ])
        .setup(|app| {
            // Create menu
            let open_item = MenuItem::with_id(app, "open", "Open...", true, Some("CmdOrCtrl+O"))?;
            let file_menu = Submenu::with_items(app, "File", true, &[&open_item])?;

            let splash_window = app.get_webview_window("splash").unwrap();
            let main_window = app.get_webview_window("main").unwrap();

            // Set menu at app level for macOS, window level for other platforms
            #[cfg(target_os = "macos")]
            {
                // On macOS, create app menu + file menu
                let app_menu = Submenu::new(app, "", true)?;
                let menu = Menu::with_items(app, &[&app_menu, &file_menu])?;
                app.set_menu(menu)?;
            }

            #[cfg(not(target_os = "macos"))]
            {
                let menu = Menu::with_items(app, &[&file_menu])?;
                main_window.set_menu(menu)?;
            }

            // Clone the windows for the async block
            let splash = splash_window.clone();
            let main = main_window.clone();

            // Show splash screen for 2 seconds, then transition to main window
            tauri::async_runtime::spawn(async move {
                // Wait for 2 seconds to show the splash screen
                tokio::time::sleep(Duration::from_secs(2)).await;

                // Close splash and show main window
                splash.close().unwrap();
                main.show().unwrap();
                main.set_focus().unwrap();
            });

            Ok(())
        })
        .on_menu_event(|app, event| {
            println!("Menu event received: {:?}", event.id());
            if event.id() == "open" {
                println!("Open menu clicked");
                let app_handle = app.clone();

                // No filters - show all files
                app.dialog()
                    .file()
                    .pick_file(move |file_path| {
                    println!("File picker callback - file_path: {:?}", file_path);
                    if let Some(path) = file_path {
                        let path_str = path.to_string();
                        println!("Emitting file-opened event with path: {}", path_str);
                        let result = app_handle.emit("file-opened", path_str);
                        println!("Emit result: {:?}", result);
                    }
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
