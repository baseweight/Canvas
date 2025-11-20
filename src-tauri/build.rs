fn main() {
    // Determine the library subdirectory based on target OS and architecture
    let lib_subdir = if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "darwin-arm64"
        } else {
            "darwin-x64"
        }
    } else if cfg!(target_os = "linux") {
        if cfg!(target_arch = "aarch64") {
            "linux-arm64"
        } else {
            "linux-x64"
        }
    } else if cfg!(target_os = "windows") {
        // Check LLAMA_BACKEND env var: "cuda" (default) or "vulkan"
        let backend = std::env::var("LLAMA_BACKEND").unwrap_or_else(|_| "cuda".to_string());
        match backend.to_lowercase().as_str() {
            "vulkan" => "windows-x64-vulkan",
            "cuda" | _ => "windows-x64-cuda",
        }
    } else {
        panic!("Unsupported target OS")
    };

    // Re-run build script if LLAMA_BACKEND changes
    println!("cargo:rerun-if-env-changed=LLAMA_BACKEND");

    // Tell cargo to look for shared libraries in the libs directory
    let lib_dir = std::env::current_dir()
        .unwrap()
        .join("libs")
        .join(lib_subdir);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Tell cargo to tell rustc to link the llama.cpp libraries
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");
    println!("cargo:rustc-link-lib=dylib=ggml");

    // Set RPATH so the binary can find the shared libraries at runtime
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // Re-run build script if libraries change
    println!("cargo:rerun-if-changed=libs/{}", lib_subdir);

    tauri_build::build()
}
