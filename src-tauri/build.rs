fn main() {
    // Tell cargo to look for shared libraries in the libs directory
    let lib_dir = std::env::current_dir()
        .unwrap()
        .join("libs")
        .join("linux-x64");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Tell cargo to tell rustc to link the llama.cpp libraries
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");
    println!("cargo:rustc-link-lib=dylib=ggml");

    // Set RPATH so the binary can find the shared libraries at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // Re-run build script if libraries change
    println!("cargo:rerun-if-changed=libs/linux-x64");

    tauri_build::build()
}
