use tauri::Manager;
use std::time::Duration;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet])
        .setup(|app| {
            let splash_window = app.get_webview_window("splash").unwrap();
            let main_window = app.get_webview_window("main").unwrap();

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
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
