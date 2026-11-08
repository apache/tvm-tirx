/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
//! Links `libtvm_tirx_compiler` from the directory `tvm-tirx-config --libdir` prints, as
//! `tvm-ffi-sys` links libtvm_ffi from the one `tvm-ffi-config --libdir` prints. The loader-path
//! environment `tvm-ffi-sys` emits only applies to its own package, so this script emits it again
//! for both directories, so that a plain `cargo test` finds the libraries at startup.
//!
//! A documentation build links nothing, so with `DOCS_RS` set, as docs.rs sets it, the script
//! only warns when it finds no library, as `tvm-ffi-sys` does.

use std::env;
use std::path::Path;
use std::process::Command;

/// The library directory a configuration helper prints for `--libdir`.
fn lib_dir(config: &str) -> Result<String, String> {
    let output = Command::new(config).arg("--libdir").output().map_err(|err| {
        format!("could not run `{config}`: {err}; put the directory of `{config}` on PATH")
    })?;
    if !output.status.success() {
        return Err(format!(
            "`{config} --libdir` failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let lib_dir = String::from_utf8(output.stdout).unwrap().trim().to_string();
    if lib_dir.is_empty() {
        return Err(format!("`{config} --libdir` printed no library directory"));
    }
    Ok(lib_dir)
}

/// Checks that `lib_dir` holds the file the linker looks for under `-ltvm_tirx_compiler`.
///
/// `tvm-tirx-config --libdir` falls back to `<prefix>/lib` when no directory it knows holds the
/// libraries, as before the first build. Failing here rather than in the link step makes cargo
/// run the script again on the next build, instead of keeping a search path that has no library.
fn check_compiler_library(lib_dir: &str, target_os: &str) -> Result<(), String> {
    let names: &[&str] = match target_os {
        "windows" => &["tvm_tirx_compiler.lib", "libtvm_tirx_compiler.dll.a"],
        "macos" | "ios" => &["libtvm_tirx_compiler.dylib"],
        _ => &["libtvm_tirx_compiler.so"],
    };
    let dir = Path::new(lib_dir);
    if names.iter().any(|name| dir.join(name).is_file()) {
        return Ok(());
    }
    Err(format!(
        "`tvm-tirx-config --libdir` printed {lib_dir}, which holds no {}: build TIRx first \
         (`uv pip install -e .`), or set TVM_TIRX_LIBRARY_PATH to the directory of the library",
        names.join(" or ")
    ))
}

fn main() {
    // Cargo sets RUSTDOC for every build script, so only DOCS_RS marks a documentation build.
    let docs_only = env::var_os("DOCS_RS").is_some();
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=PATH");
    // `tvm-tirx-config` puts this directory first when it is set.
    println!("cargo:rerun-if-env-changed=TVM_TIRX_LIBRARY_PATH");
    println!("cargo:rerun-if-changed=build.rs");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let found = lib_dir("tvm-ffi-config").and_then(|ffi_lib_dir| {
        let tirx_lib_dir = lib_dir("tvm-tirx-config")?;
        check_compiler_library(&tirx_lib_dir, &target_os)?;
        Ok((ffi_lib_dir, tirx_lib_dir))
    });
    let (ffi_lib_dir, tirx_lib_dir) = match found {
        Ok(dirs) => dirs,
        Err(reason) if docs_only => {
            println!(
                "cargo:warning={reason}; not linking libtvm_tirx_compiler for a documentation build"
            );
            return;
        }
        Err(reason) => panic!("{reason}. To build documentation only, set DOCS_RS=1."),
    };

    // The crate references the library only through its link anchor (see `src/lib.rs`).
    println!("cargo:rustc-link-search=native={tirx_lib_dir}");
    println!("cargo:rustc-link-lib=dylib=tvm_tirx_compiler");

    let loader_var = match target_os.as_str() {
        "windows" => "PATH",
        "macos" => "DYLD_LIBRARY_PATH",
        _ => "LD_LIBRARY_PATH",
    };
    // The value emitted below starts with the variable's value in this script's environment, so
    // the script must run again when that changes; on Windows the variable is PATH, declared above.
    if loader_var != "PATH" {
        println!("cargo:rerun-if-env-changed={loader_var}");
    }
    let sep = if loader_var == "PATH" { ";" } else { ":" };
    let mut dirs = vec![ffi_lib_dir, tirx_lib_dir];
    if let Ok(prev) = env::var(loader_var) {
        if !prev.is_empty() {
            dirs.insert(0, prev);
        }
    }
    println!("cargo:rustc-env={loader_var}={}", dirs.join(sep));
}
