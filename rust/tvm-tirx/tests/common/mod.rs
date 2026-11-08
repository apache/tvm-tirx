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

//! Helpers shared by the test binaries, one per top-level module of the crate.

#![allow(dead_code)]

use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::mem::size_of;
use std::sync::{Mutex, MutexGuard, Once};

use tvm_tirx::ir::{BaseFunc, Expr, GlobalInfo, GlobalVar, IRModule, MissingTypeObj, PrimTypeObj};
use tvm_tirx::tvm_ffi::tvm_ffi_sys::{TVMFFIGetTypeInfo, TVMFFITypeInfo};
use tvm_tirx::tvm_ffi::{
    Any, AnyCompatible, AnyView, Array, Function, Map, Module, ObjectCore, ObjectRefCore, Result,
    String as FfiString, StructuralOptions,
};

/// Run the tests of a binary one at a time: registering an operator, a memory scope or the layout
/// axes of a backend grows a native registry, which other threads must not read meanwhile, and
/// two threads that register the same layout axes collide.
pub fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Load the LLVM backend library, which registers the `codegen.llvm_*` functions and the `llvm`
/// target build, as `import tvm_tirx` does. It sits beside `libtvm_tirx_compiler`, whose directory
/// the build script puts on the loader path; a build with `TVM_TIRX_LLVM_NATIVE_ONLY` names it
/// `tvm_tirx_llvm_backend_native`.
pub fn load_llvm_backend() {
    static LOAD: Once = Once::new();
    LOAD.call_once(|| {
        let mut errors = Vec::new();
        for name in ["tvm_tirx_llvm_backend", "tvm_tirx_llvm_backend_native"] {
            match Module::load_from_file(format!("{DLL_PREFIX}{name}{DLL_SUFFIX}")) {
                // Unloading the library would unregister its functions.
                Ok(backend) => return std::mem::forget(backend),
                Err(err) => errors.push(err.to_string()),
            }
        }
        panic!("could not load the LLVM backend: {}", errors.join("\n"));
    });
}

pub fn global(name: &str) -> Function {
    Function::get_global(name).unwrap()
}

pub fn structural_equal<L: AnyCompatible, R: AnyCompatible>(lhs: &L, rhs: &R) -> bool {
    tvm_ffi::structural_equal(lhs, rhs).unwrap()
}

/// The native function `name` on `args`.
pub fn native(name: &str, args: &[AnyView]) -> Result<Any> {
    global(name).call_packed(args)
}

/// Check that `rust`, built by a constructor written in Rust, agrees with `native`, built by the C++
/// constructor it follows: nodes of the same class that are structurally equal, whose types are
/// equal too, and are the same node when C++ shares it (a `PrimType` of the same dtype, the
/// `MissingType` singleton); or errors of the same kind and message.
pub fn agree<T: AnyCompatible>(what: &str, rust: Result<T>, native: Result<Any>) {
    agree_up_to(false, what, rust, native)
}

/// [`agree`] for a fresh variable, which equals another only up to renaming (structural equality
/// that maps free variables).
pub fn agree_fresh<T: AnyCompatible>(what: &str, rust: Result<T>, native: Result<Any>) {
    agree_up_to(true, what, rust, native)
}

fn agree_up_to<T: AnyCompatible>(
    map_free_vars: bool,
    what: &str,
    rust: Result<T>,
    native: Result<Any>,
) {
    match (rust, native) {
        (Ok(rust), Ok(native)) => {
            let rust = Any::from(rust);
            assert_eq!(rust.type_index(), native.type_index(), "{what}: the node classes differ");
            assert!(
                any_structural_equal(&rust, &native, map_free_vars),
                "{what}: the nodes differ"
            );
            if let (Ok(rust), Ok(native)) = (Expr::try_from(rust), Expr::try_from(native)) {
                assert!(structural_equal(&rust.ty, &native.ty), "{what}: the types differ");
                let shared = native.ty.downcast_ref::<PrimTypeObj>().is_some()
                    || native.ty.downcast_ref::<MissingTypeObj>().is_some();
                if shared {
                    assert!(rust.ty.same_as(&native.ty), "{what}: the shared type is not shared");
                }
            }
        }
        (Err(rust), Err(native)) => {
            assert_eq!(rust.kind(), native.kind(), "{what}: the error kinds differ");
            assert_eq!(rust.message(), native.message(), "{what}: the error messages differ");
        }
        (Ok(_), Err(native)) => panic!("{what}: only the native constructor fails: {native}"),
        (Err(rust), Ok(_)) => panic!("{what}: only the Rust constructor fails: {rust}"),
    }
}

/// Check that both constructors fail with the same kind of error. For an argument the native
/// function cannot convert, such as an expression that is not a `PrimExpr`, its message names the
/// function signature instead of the constructor's check.
pub fn agree_kind<T: AnyCompatible>(what: &str, rust: Result<T>, native: Result<Any>) {
    let rust = rust.err().unwrap_or_else(|| panic!("{what}: the Rust constructor succeeds"));
    let native = native.err().unwrap_or_else(|| panic!("{what}: the native constructor succeeds"));
    assert_eq!(rust.kind(), native.kind(), "{what}: the error kinds differ: {rust} vs. {native}");
}

fn any_structural_equal(lhs: &Any, rhs: &Any, map_free_vars: bool) -> bool {
    let options = StructuralOptions { map_free_vars, ..StructuralOptions::default() };
    tvm_ffi::structural_equal_with(lhs, rhs, options).unwrap()
}

pub fn empty_module() -> IRModule {
    global("ir.IRModule")
        .call_tuple((
            Map::<GlobalVar, BaseFunc>::new(),
            (),
            Map::<FfiString, Array<GlobalInfo>>::new(),
        ))
        .unwrap()
        .try_into()
        .unwrap()
}

pub fn runtime_type_info<N: ObjectCore>() -> &'static TVMFFITypeInfo {
    let pointer = unsafe { TVMFFIGetTypeInfo(N::type_index()) };
    assert!(!pointer.is_null(), "missing type info for {}", N::TYPE_KEY);
    unsafe { &*pointer }
}

/// The Rust struct agrees with the registry on key, depth, parent, and (when complete) size.
pub fn assert_type_contract<N: ObjectCore, P: ObjectCore>(complete: bool) {
    let info = runtime_type_info::<N>();
    // SAFETY: the type table holds its keys as UTF-8 for the life of the process.
    assert_eq!(unsafe { info.type_key.as_str() }, N::TYPE_KEY);
    assert_eq!(info.type_depth, N::TYPE_DEPTH, "depth of {}", N::TYPE_KEY);
    assert_eq!(N::TYPE_DEPTH, P::TYPE_DEPTH + 1, "parent of {}", N::TYPE_KEY);
    let parent = unsafe { *info.type_acenstors.add(P::TYPE_DEPTH as usize) };
    assert_eq!(unsafe { (*parent).type_index }, P::type_index(), "parent of {}", N::TYPE_KEY);
    if complete {
        let metadata = unsafe { info.metadata.as_ref() }
            .unwrap_or_else(|| panic!("{} has no metadata", N::TYPE_KEY));
        assert_eq!(metadata.total_size as usize, size_of::<N>(), "size of {}", N::TYPE_KEY);
    }
}

#[allow(unused_macros)]
macro_rules! assert_type_contracts {
    ($complete:expr; $($node:ty : $parent:ty),* $(,)?) => {
        $($crate::common::assert_type_contract::<$node, $parent>($complete);)*
    };
}
#[allow(unused_imports)]
pub(crate) use assert_type_contracts;
