// Compiles the Core Audio process-tap shim and links the frameworks it
// needs. macOS-only; a no-op everywhere else.
fn main() {
    #[cfg(target_os = "macos")]
    {
        cc::Build::new()
            .file("src/macos_systap.m")
            .flag("-fobjc-arc")
            // The Core Audio process-tap APIs are macOS 14.4+. The host
            // app gates on 14.6 at runtime; compile the shim against that
            // floor so the symbols are available without @available churn.
            .flag("-mmacosx-version-min=14.6")
            .compile("krono_systap");
        println!("cargo:rustc-link-lib=framework=CoreAudio");
        println!("cargo:rustc-link-lib=framework=AudioToolbox");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rerun-if-changed=src/macos_systap.m");
    }
}
