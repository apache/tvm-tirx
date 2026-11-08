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

//! The TIRx compiler in Rust: its IR objects over TVM FFI, and the definition of new types,
//! operators and passes that extend it.
//!
//! The modules:
//!
//! - [`ir`]: the IR shared by every dialect, with the primitive expressions in [`ir::prim`] and
//!   the pass infrastructure in [`ir::transform`].
//! - [`tirx`]: tensor programs: functions, tensor types, layouts, tile instructions, passes and
//!   analyses.
//! - [`sym`]: symbolic integer analysis.
//! - [`target`]: compilation targets.
//! - [`backend`]: backend objects, such as the attributes of CUDA instructions.
//! - [`runtime`]: runtime objects and queries, and tensor allocation.
//! - [`codegen`]: compiling IR modules into runtime modules.
//! - [`script`]: printing IR objects as TVMScript.
//!
//! # Using the crate
//!
//! The crate links `libtvm_tirx_compiler`, which the build script finds through
//! `tvm-tirx-config --libdir`; `tvm-tirx-config` and `tvm-ffi-config` must be on `PATH` when the
//! crate is built. Many of its APIs call global functions of that library, as
//! [`target::Target::new`] calls `target.Target`, and any global function can also be called
//! through [`tvm_ffi::Function::get_global`]. A program that calls them only through `tvm_ffi`,
//! without an API of this crate, must link the library itself
//! (`tvm-tirx-config --libs compiler`).
//!
//! ```
//! use tvm_tirx::target::Target;
//! use tvm_tirx::tvm_ffi::{Function, String};
//!
//! let target = Target::new("llvm")?;
//! assert_eq!(target.kind.name()?, "llvm");
//!
//! let same = Function::get_global("target.Target")?.call_tuple((String::from("llvm"),))?;
//! assert_eq!(Target::try_from(same)?.kind.name()?, "llvm");
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```
//!
//! A handle such as [`ir::Expr`] is a reference-counted pointer to its object, [`ir::ExprObj`], so
//! cloning it does not copy the IR. A constructor `new` checks its arguments and fills in defaults,
//! and returns a `Result` when a check can fail; `from_fields` stores the fields as given.
//!
//! # Defining operators and types
//!
//! Operators and types are defined in a [`tvm_ffi::static_init_block!`], which runs when the
//! program or library loads: an operator by an [`ir::OpDef`] chain, a type by
//! [`tvm_ffi::rs_object`] and [`ir::register_type`].
//!
//! ```standalone_crate
//! use tvm_tirx::ir::op_attr::CALL_EFFECT_KIND;
//! use tvm_tirx::ir::sig::{self, IntExpr};
//! use tvm_tirx::ir::{register_type, CallEffectKind, CallObj, Op, OpDef, Type};
//! use tvm_tirx::tvm_ffi::{self, Result};
//! # use tvm_tirx::ir::TypeObj;
//! # use tvm_tirx::tvm_ffi::{rs_object, ObjectRefCore};
//! # #[rs_object(type_key = "my_ext.PairType", final, immutable)]
//! # pub struct PairTypeObj { base_: TypeObj, pub first: Type, pub second: Type }
//! # #[derive(Clone, tvm_ffi::derive::ObjectRef)]
//! # #[object_ref(parent = Type)]
//! # pub struct PairType {
//! #     data_: tvm_ffi::ObjectArc<PairTypeObj>,
//! # }
//!
//! fn index_type(call: &CallObj) -> Result<Type> {
//!     Ok(call.args[0].ty.clone())
//! }
//!
//! tvm_ffi::static_init_block! {
//!     OpDef::new("my_ext.clamp", "Clamp an index to the range `[0, bound)`.")
//!         .signature((
//!             sig::arg::<IntExpr>("index", "The index."),
//!             sig::arg::<IntExpr>("bound", "The exclusive upper bound."),
//!         ))
//!         .set_infer_type(index_type)
//!         .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
//!     register_type::<PairType>()?;
//! }
//!
//! let clamp = Op::get("my_ext.clamp")?;
//! # Ok::<(), tvm_ffi::Error>(())
//! ```
//!
//! A type extends [`ir::TypeObj`]; its fields are reflected, so Python and the TVMScript printer
//! see them. The generated code names `tvm_ffi`, which `use tvm_tirx::tvm_ffi` brings into scope.
//!
//! ```standalone_crate
//! use tvm_tirx::ir::{PrimType, Type, TypeObj, UNKNOWN_LOC};
//! use tvm_tirx::tvm_ffi::{self, rs_object, IntoArg, ObjectRefCore};
//!
//! #[rs_object(type_key = "my_ext.PairType", final, immutable)]
//! pub struct PairTypeObj {
//!     base_: TypeObj,
//!     pub first: Type,
//!     pub second: Type,
//!     #[reflect(seq_hash_ignore)]
//!     pub note: tvm_ffi::String,
//! }
//!
//! #[derive(Clone, tvm_ffi::derive::ObjectRef)]
//! #[object_ref(parent = Type)]
//! pub struct PairType {
//!     data_: tvm_ffi::ObjectArc<PairTypeObj>,
//! }
//!
//! impl PairType {
//!     pub fn new(first: impl IntoArg<Type>, second: impl IntoArg<Type>) -> Self {
//!         let (first, second) = (first.into_owned(), second.into_owned());
//!         Self::from_object(PairTypeObj::new(TypeObj::new(UNKNOWN_LOC.into()), first, second, ""))
//!     }
//! }
//!
//! fn main() {
//!     let pair = PairType::new(PrimType::int(32), PrimType::float(32));
//!     let ty: Type = pair.into();
//!     assert!(ty.downcast_ref::<PairTypeObj>().is_some());
//! }
//! ```
//!
//! # The library Python loads
//!
//! The crate is also built as `libtvm_tirx_rust`, which `import tvm_tirx` loads, so Python uses the
//! operators and types the crate defines as it uses the native ones. A `cdylib` of operators of its
//! own, such as `examples/define_ir.rs`, registers them when Python loads it; a crate it depends on
//! but never names is not linked in, so name it, as `use my_ops as _;` does.
//!
//! # Generated code
//!
//! The code between `// tvm-ffi-stubgen(begin)` and `// tvm-ffi-stubgen(end)` is generated, and
//! the rest is hand-written. Regenerate it with the `tvm-ffi-stubgen` of `3rdparty/tvm-ffi` after
//! rebuilding `libtvm_tirx_compiler` (`--check` only reports stale blocks):
//!
//! ```bash
//! tvm-ffi-stubgen rust/tvm-tirx/src --target rust --dlls build/lib/libtvm_tirx_compiler.so
//! ```
//!
//! # Tests
//!
//! Build `libtvm_tirx_compiler` first. The tests of `src/<path>.rs` are in `tests/<path>.rs`:
//!
//! ```bash
//! uv run --no-sync cargo test --manifest-path rust/Cargo.toml
//! ```

#![warn(missing_docs)]

pub mod backend;
pub mod codegen;
pub mod ir;
mod r#macro;
pub mod runtime;
pub mod script;
pub mod sym;
pub mod target;
pub mod tirx;

pub use tvm_ffi;

// The errors of the static init blocks of every library built on the crate, which
// `import tvm_tirx` raises for `libtvm_tirx_rust`: `__tvm_ffi_tirx_register_ir`, the TVM FFI
// function `tirx_register_ir` of the library as `tvm_ffi.load_module` loads it. The module keeps
// the symbol, which `#[no_mangle]` exports from wherever it is, out of the crate's documentation.
#[doc(hidden)]
pub mod exports {
    tvm_ffi::tvm_ffi_dll_export_typed_func!(tirx_register_ir, tvm_ffi::run_static_init_blocks);
}

// The crate reaches `libtvm_tirx_compiler` through the FFI registry, so a linker that drops
// unreferenced libraries (`--as-needed`) would drop the library the build script links, and its
// registrations would never run. Each TIRx shared library exports a `<library>_link_anchor`
// function that does nothing, defined in the `link_anchor.cc` of its sources (for the compiler,
// `src/ir/link_anchor.cc`) and declared by whoever references it. The static below holds the
// anchor's address: a relocation against the symbol in the object of this module, which costs
// nothing at run time and is what makes the linker keep the library.
//
// A program pulls in only the objects of the crate it references, so the static keeps the
// library only when the object of this module is pulled. It is, because the objects of the crate
// reference one another densely and `sym::analyzer` references `ExitOnDrop::new` below. The test
// binaries rely on the same mechanism and fail at their first registry lookup if that changes. A
// program that uses the registry only through `tvm_ffi` itself, without calling a wrapper, must
// keep the library by other means, such as the linker arguments `tvm-tirx-config --libs compiler`
// prints.
unsafe extern "C" {
    fn tvm_tirx_compiler_link_anchor() -> i32;
}

#[used]
static TVM_TIRX_COMPILER_LINK_ANCHOR: unsafe extern "C" fn() -> i32 = tvm_tirx_compiler_link_anchor;

/// Makes a scope guard neither `Send` nor `Sync`: the native scopes keep their state per thread,
/// or in an object that is not synchronized, so a scope must end on the thread that opened it.
pub(crate) type ThreadBound = std::marker::PhantomData<*const ()>;

/// Calls a native exit function when dropped: the end of a scope a native enter function opened.
pub(crate) struct ExitOnDrop(tvm_ffi::Function, ThreadBound);

impl ExitOnDrop {
    pub(crate) fn new(exit: tvm_ffi::Function) -> Self {
        Self(exit, ThreadBound::default())
    }
}

impl Drop for ExitOnDrop {
    fn drop(&mut self) {
        // A destructor cannot report the error, and the native exit functions do not fail.
        let _ = self.0.call_packed(&[]);
    }
}
