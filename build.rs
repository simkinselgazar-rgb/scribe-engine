// Compiles the macOS system-audio shim and links the frameworks it needs.
// macOS-only; a no-op everywhere else. The far channel is captured via
// ScreenCaptureKit (`macos_sckaudio.m`) — the Core Audio process-tap shim
// (`macos_systap.m`) is kept in the tree for reference but no longer built
// (it cannot reliably capture a global mix on macOS 15/26).
fn main() {
    #[cfg(target_os = "macos")]
    {
        cc::Build::new()
            .file("src/macos_sckaudio.m")
            .flag("-fobjc-arc")
            // ScreenCaptureKit audio capture is macOS 13+. The host app
            // gates higher at runtime; compile against 13 so the symbols
            // resolve, with an @available guard in the shim for safety.
            .flag("-mmacosx-version-min=13.0")
            .compile("krono_sckaudio");
        println!("cargo:rustc-link-lib=framework=ScreenCaptureKit");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=CoreAudio");
        println!("cargo:rustc-link-lib=framework=AudioToolbox");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rerun-if-changed=src/macos_sckaudio.m");
    }
}
