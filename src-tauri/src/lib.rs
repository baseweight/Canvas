use tauri::{Manager, Emitter, State};
use tauri::menu::{Menu, MenuItem, Submenu};
use tauri_plugin_dialog::DialogExt;
use std::time::Duration;
use serde::{Deserialize, Serialize};

mod llama_inference;
mod model_manager;
pub mod inference_engine;

use model_manager::{ModelManager, DownloadProgress};
use inference_engine::{InferenceEngine, SharedInferenceEngine, create_shared_engine};

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

// Download SmolVLM2 2.2B Instruct
#[tauri::command]
async fn download_bundled_model(app: tauri::AppHandle) -> Result<(), String> {
    let manager = ModelManager::new().map_err(|e| e.to_string())?;

    manager
        .download_smolvlm2(move |progress: DownloadProgress| {
            // Emit progress to frontend
            let _ = app.emit("download-progress", &progress);
        })
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(create_shared_engine())
        .invoke_handler(tauri::generate_handler![
            greet,
            is_bundled_model_downloaded,
            get_models_directory,
            download_bundled_model,
            get_model_paths,
            load_model,
            generate_response
        ])
        .setup(|app| {
            // Create menu for main window only
            let open_item = MenuItem::with_id(app, "open", "Open...", true, Some("CmdOrCtrl+O"))?;
            let file_menu = Submenu::with_items(app, "File", true, &[&open_item])?;
            let menu = Menu::with_items(app, &[&file_menu])?;

            let splash_window = app.get_webview_window("splash").unwrap();
            let main_window = app.get_webview_window("main").unwrap();

            // Set menu only on main window
            main_window.set_menu(menu)?;

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
                app.dialog()
                    .file()
                    .add_filter("Images", &["png", "jpg", "jpeg", "gif", "bmp", "webp"])
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
