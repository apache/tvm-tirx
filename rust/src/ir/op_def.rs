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

//! Registering operators from Rust: C++ `OpDef`, with its signature descriptors ([`sig`]) and
//! attribute keys ([`op_attr`]); the [`register_ops!`](crate::register_ops) blocks that define
//! operators, as C++ defines them in a `TVM_FFI_STATIC_INIT_BLOCK`; and the handles that keep
//! operators in `static`s.
//!
//! Every hook an operator gets here is native, as the C++ ones are: a Rust function compiled into
//! the C ABI adapter that C++ `NativeFunctionView` calls through a function pointer, without
//! packing arguments. A signature generates its validator from its argument types, as C++
//! `OpDef::signature` does from `sig::arg<T>`; each signature is its own compiled function.
//!
//! The [`register_ops!`](crate::register_ops) and [`register_types!`](crate::register_types)
//! blocks are gathered at link time, wherever in the program or library they are, and run when it
//! loads, as a C++ static init block does: once Python loads a library built on this crate, by any
//! means, it sees its operators and types as it sees those of a C++ library. A block that several
//! libraries of a process have, as each links its own copy of a crate they share, runs once.

use std::any::TypeId;
use std::cell::RefCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Once, OnceLock};

use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFIObject, TVMFFITypeIndex};
use tvm_ffi::{
    Any, AnyView, Array, Error, Function, ObjectArc, ObjectRefCore, Result, String, INTERNAL_ERROR,
    TYPE_ERROR, VALUE_ERROR,
};

use super::attrs::Attrs;
use super::expr::{CallObj, Expr, Var};
use super::op::{Op, OpObj};
use super::r#type::Type;
use super::stmt::RegionStmtObj;
use super::support;
use sig::{Checks, Specs};

/// C++ `OpDef`: register a new operator from Rust.
///
/// Each step updates the registry at once, as in C++. After a failed step the rest are skipped,
/// and [`OpDef::op`] returns the error. The registry does not synchronize registration with
/// readers, so register an operator before building IR with it, as a
/// [`register_ops!`](crate::register_ops) block does. An operator and its hooks live for the rest
/// of the process.
///
/// ```no_run
/// use tvm_tirx::ir::op_attr::CALL_EFFECT_KIND;
/// use tvm_tirx::ir::sig::{self, IntExpr};
/// use tvm_tirx::ir::{CallEffectKind, CallObj, OpDef, Type};
/// use tvm_tirx::tvm_ffi::Result;
///
/// fn first_arg_type(call: &CallObj) -> Result<Type> {
///     Ok(call.args.get(0)?.ty.clone())
/// }
///
/// let op = OpDef::new("my_ext.clamp", "Clamp an index to a bound.")
///     .signature((sig::arg::<IntExpr>("index", ""), sig::arg::<IntExpr>("bound", "")))
///     .set_infer_type(first_arg_type)
///     .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure)
///     .op()?;
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub struct OpDef(Result<Op>);

impl OpDef {
    /// Register a new operator (`ir.RegisterOp`), with `doc` as its documentation, less the
    /// indentation its lines share and the blank lines around it, so that a string of several
    /// lines can be indented with the code. A name already registered is an error.
    pub fn new(name: &str, doc: &str) -> Self {
        let registered = tvm_ffi::cached_global_func!("ir.RegisterOp")
            .call_tuple((String::from(name), String::from(doc_text(doc))));
        Self(registered.and_then(|_| Op::get(name)))
    }

    /// The completed operator, or the error of the step that failed. Request its TVMScript
    /// builder through `script.expose_script_op` after its signature and attributes are ready.
    /// An absent receiver is a no-op; an installed receiver's error is returned.
    pub fn op(self) -> Result<Op> {
        let op = self.0?;
        if let Ok(expose) = tvm_ffi::Function::get_global("script.expose_script_op") {
            expose.call_tuple((op.name.clone(),))?;
        }
        Ok(op)
    }

    /// C++ `OpDef::signature`: replace the argument descriptions and the attrs type, and the
    /// validator generated from them, which checks a Call's argument counts, argument types and
    /// attrs. `specs` is a descriptor of [`sig`], or a tuple of them in order, as C++ takes them.
    /// A custom validator keeps precedence. Without [`sig::call_attrs`] the operator has no attrs
    /// type, and `Call.attrs` is unconstrained.
    pub fn signature<S: Specs>(self, specs: S) -> Self {
        let mut description = sig::Description::default();
        specs.describe(&mut description);
        let lists = description.into_lists().map(Array::new);
        let args: Vec<AnyView> = lists.iter().map(AnyView::from).collect();
        let def = self.invoke("_set_signature", &args).then(|op| {
            if !op.validator_is_custom() {
                let hook: Hook = validate_signature::<S::Checks>;
                op.write_validator(Any::from(hook as *mut c_void), false);
            }
            Ok(())
        });
        match <S::Checks as Checks>::ATTRS_TYPE_KEY {
            Some(key) => def.attrs_type_key(key),
            None => def,
        }
    }

    /// Append a value argument's description, without changing what the validator checks.
    pub fn add_arg(self, name: &str, doc: &str) -> Self {
        let (name, doc) = (String::from(name), String::from(doc));
        self.invoke("add_arg", &[AnyView::from(&name), AnyView::from(&doc)])
    }

    /// Append a type argument's description, without changing what the validator checks.
    pub fn add_ty_arg(self, name: &str, doc: &str) -> Self {
        let (name, doc) = (String::from(name), String::from(doc));
        self.invoke("add_ty_arg", &[AnyView::from(&name), AnyView::from(&doc)])
    }

    /// The type of `Call.attrs`, by its registered type key.
    pub fn attrs_type_key(self, key: &str) -> Self {
        let key = String::from(key);
        self.invoke("set_attrs_type_key", &[AnyView::from(&key)])
    }

    /// Register the attribute `key` (C++ `set_attr(key, value)`); a value already registered for
    /// this operator is an error.
    pub fn set_attr<T: Into<Any>>(self, key: AttrKey<T>, value: impl Into<T>) -> Self {
        let value: T = value.into();
        self.attr(key.name, value.into(), false)
    }

    /// Register the attribute `key`, replacing a value already registered for this operator
    /// (C++ `set_attr(key, value, true)`).
    pub fn replace_attr<T: Into<Any>>(self, key: AttrKey<T>, value: impl Into<T>) -> Self {
        let value: T = value.into();
        self.attr(key.name, value.into(), true)
    }

    /// Remove this operator's value of the attribute `key`.
    pub fn reset_attr<T>(self, key: AttrKey<T>) -> Self {
        let key = String::from(key.name);
        self.invoke("reset_attr", &[AnyView::from(&key)])
    }

    /// `FInferType`: the result type of a Call that does not give one, which the `Call`
    /// constructor and `Call::reinfer_type` ask for; [`op_attr::FIXED_RETURN_TYPE`] takes
    /// precedence. `infer` is a function or a closure that captures nothing.
    pub fn set_infer_type<F>(self, infer: F) -> Self
    where
        F: Fn(&CallObj) -> Result<Type> + 'static,
    {
        let _ = infer;
        let hook: Hook = infer_type_abi::<F>;
        self.attr("FInferType", Any::from(hook as *mut c_void), false)
    }

    /// A custom validator, run by `Call::validate`, which reports an invalid Call as an error. It
    /// replaces the one a signature generated, and a later signature keeps it; a second custom
    /// validator is an error. `validate` is a function or a closure that captures nothing.
    pub fn set_validator<F>(self, validate: F) -> Self
    where
        F: Fn(&CallObj) -> Result<()> + 'static,
    {
        let _ = validate;
        self.then(|op| {
            if op.validator_is_custom() {
                let message = format!("Validator of {} is already registered", op.name.as_str());
                return Err(Error::new(VALUE_ERROR, &message, ""));
            }
            let hook: Hook = validate_abi::<F>;
            op.write_validator(Any::from(hook as *mut c_void), true);
            Ok(())
        })
    }

    /// `FRegionGetBodyParams`, which makes the operator a region operation: the variables a
    /// region binds in its body, from its Call (the operation, operands and attributes), which
    /// the `RegionStmt` constructor checks the body parameters against. A region that binds none
    /// returns none. `body_params` is a function or a closure that captures nothing.
    pub fn set_region_body_params<F>(self, body_params: F) -> Self
    where
        F: Fn(&CallObj) -> Result<Array<Var>> + 'static,
    {
        let _ = body_params;
        let hook: Hook = region_body_params_abi::<F>;
        self.attr("FRegionGetBodyParams", Any::from(hook as *mut c_void), false)
    }

    /// `FRegionValidate`: a check of a complete region statement, run by the `RegionStmt`
    /// constructor after the body parameters. `validate` is a function or a closure that
    /// captures nothing.
    pub fn set_region_validate<F>(self, validate: F) -> Self
    where
        F: Fn(&RegionStmtObj) -> Result<()> + 'static,
    {
        let _ = validate;
        let hook: RegionHook = region_validate_abi::<F>;
        self.attr("FRegionValidate", Any::from(hook as *mut c_void), false)
    }

    fn attr(self, key: &str, value: Any, replace: bool) -> Self {
        let key = String::from(key);
        self.invoke(
            "_set_attr",
            &[AnyView::from(&key), AnyView::from(&value), AnyView::from(&replace)],
        )
    }

    /// Call the `ir.Op` method `method` on the operator with `args`.
    fn invoke(self, method: &str, args: &[AnyView]) -> Self {
        self.then(|op| {
            let packed: Vec<AnyView> =
                std::iter::once(AnyView::from(op)).chain(args.iter().copied()).collect();
            tvm_ffi::Function::from_type_key_method("ir.Op", method)?.call_packed(&packed)?;
            Ok(())
        })
    }

    fn then(self, step: impl FnOnce(&Op) -> Result<()>) -> Self {
        Self(self.0.and_then(|op| step(&op).map(|_| op)))
    }
}

/// C++ `ffi::reflection::AttrKey<T>`: the name of an operator attribute and the type of its
/// values, which [`OpDef::set_attr`] takes. [`op_attr`] and
/// [`tirx::op_attr`](crate::tirx::op_attr) have the keys C++ declares, and `AttrKey::new` names
/// another.
pub struct AttrKey<T> {
    name: &'static str,
    value: PhantomData<fn() -> T>,
}

impl<T> AttrKey<T> {
    pub const fn new(name: &'static str) -> Self {
        Self { name, value: PhantomData }
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }
}

impl<T> Clone for AttrKey<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for AttrKey<T> {}

/// The operator attributes of the shared IR (C++ `tvm::op_attr`). The hooks, `FInferType` and
/// the region ones, have setters of their own on [`OpDef`].
pub mod op_attr {
    use super::{AttrKey, CallEffectKind};
    use crate::ir::r#type::Type;

    /// The runtime effect of a call (C++ `kCallEffectKind`, `TCallEffectKind`).
    pub const CALL_EFFECT_KIND: AttrKey<CallEffectKind> = AttrKey::new("TCallEffectKind");
    /// The result type of every call, independent of its operands, attributes and type
    /// arguments, which takes precedence over `FInferType` (C++ `kFixedReturnType`,
    /// `TFixedReturnType`).
    pub const FIXED_RETURN_TYPE: AttrKey<Type> = AttrKey::new("TFixedReturnType");
}

/// The runtime effect of an operator call (C++ `CallEffectKind`), the value of
/// [`op_attr::CALL_EFFECT_KIND`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i64)]
pub enum CallEffectKind {
    /// An annotation, such as `likely`, which can be translated to the identity.
    ExprAnnotation = 0,
    /// A pure function, which does not interact with any external state.
    Pure = 1,
    /// A function that may read from states, such as memory.
    ReadState = 2,
    /// A function that may read from and write to states, such as memory; also
    /// [`CallEffectKind::Opaque`].
    UpdateState = 3,
    /// A special intrinsic that annotates call arguments, valid only as a direct argument to a
    /// call.
    SpecialCallArg = 4,
    /// Opaque information embedded in the expression, which cannot be generated as code.
    EmbedInfo = 5,
    /// A function that changes the control flow.
    ControlJump = 6,
}

impl CallEffectKind {
    /// An opaque function, about which nothing can be assumed: C++ `kOpaque`, which is
    /// [`CallEffectKind::UpdateState`].
    #[allow(non_upper_case_globals)]
    pub const Opaque: Self = Self::UpdateState;
}

impl From<CallEffectKind> for Any {
    fn from(kind: CallEffectKind) -> Self {
        Any::from(kind as i64)
    }
}

/// A handle to an operator, kept in a `static` as C++ keeps one in `const Op& xxx_op()`. The
/// operator is looked up by name once per thread; [`OpHandle::get`] returns it and
/// [`OpHandle::is`] compares with it. The [`register_ops!`](crate::register_ops) blocks run when
/// their program or library loads; where the loader does not run them, the first lookup of an
/// operator that is not registered runs them.
///
/// ```no_run
/// use tvm_tirx::ir::sig::{self, IntExpr};
/// use tvm_tirx::ir::{Call, CallObj, Expr, OpDef, OpHandle, PrimType, Type, Var, UNKNOWN_LOC};
/// use tvm_tirx::tvm_ffi::Result;
///
/// fn first_arg_type(call: &CallObj) -> Result<Type> {
///     Ok(call.args.get(0)?.ty.clone())
/// }
///
/// tvm_tirx::register_ops! {
///     OpDef::new("my_ext.clamp", "Clamp an index to a bound.")
///         .signature((sig::arg::<IntExpr>("index", ""), sig::arg::<IntExpr>("bound", "")))
///         .set_infer_type(first_arg_type);
/// }
///
/// static CLAMP: OpHandle = OpHandle::new("my_ext.clamp");
/// static LAUNCH_THREAD: OpHandle = OpHandle::new("tirx.launch_thread");
///
/// let index = Var::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let bound = Var::new("n", PrimType::int(32), UNKNOWN_LOC);
/// let args = [Expr::from(&index), Expr::from(&bound)];
/// let clamped = Call::new(None, CLAMP.get()?, args, None, Vec::<Type>::new(), UNKNOWN_LOC)?;
/// assert!(CLAMP.is(&clamped.op) && !LAUNCH_THREAD.is(&clamped.op));
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub struct OpHandle {
    name: &'static str,
    /// This handle's slot in each thread's cache of fetched operators.
    slot: OnceLock<usize>,
    /// The operator once fetched, for identity tests: operators are registry singletons that
    /// live for the process, and `is` is the hot path of every pass that looks at `Call::op`.
    object: AtomicPtr<TVMFFIObject>,
}

/// The number of slots handed out to handles.
static SLOTS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The operators this thread fetched, by handle slot.
    static FETCHED: RefCell<Vec<Option<Op>>> = const { RefCell::new(Vec::new()) };
}

impl OpHandle {
    /// The operator `name`, registered in C++, in Python or by a
    /// [`register_ops!`](crate::register_ops) block.
    pub const fn new(name: &'static str) -> Self {
        Self { name, slot: OnceLock::new(), object: AtomicPtr::new(std::ptr::null_mut()) }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The operator. One that is not registered runs the blocks first; if it is still missing, the
    /// error says so, followed by the failures of the blocks, if any.
    pub fn get(&self) -> Result<Op> {
        self.with(Op::clone)
    }

    /// Whether `value`, such as a Call's `op`, is this operator (C++ `value.same_as(op)`). No
    /// value is an operator that cannot be fetched, such as one this build does not register.
    pub fn is(&self, value: &impl ObjectRefCore) -> bool {
        let cached = self.object.load(Ordering::Acquire);
        if !cached.is_null() {
            // SAFETY: the pointer is only compared, never dereferenced.
            let pointer = unsafe { ObjectArc::as_raw(ObjectRefCore::data(value)) };
            return pointer.cast::<TVMFFIObject>().cast_mut() == cached;
        }
        self.with(|op| {
            // Keep one reference so the pointer stays valid for the process (a second thread
            // racing here leaks one more, which is harmless for a registry singleton).
            // SAFETY: the raw pointer is never turned back into an `ObjectArc`.
            let pointer = unsafe { ObjectArc::into_raw(Op::into_data(op.clone())) };
            self.object.store(pointer.cast::<TVMFFIObject>().cast_mut(), Ordering::Release);
            value.same_as(op)
        })
        .unwrap_or(false)
    }

    /// `use_op` applied to the operator, fetched into this thread's slot the first time.
    fn with<R>(&self, use_op: impl FnOnce(&Op) -> R) -> Result<R> {
        let slot = *self.slot.get_or_init(|| SLOTS.fetch_add(1, Ordering::Relaxed));
        let cached = FETCHED.with(|fetched| fetched.borrow().get(slot).cloned().flatten());
        let op = match cached {
            Some(op) => op,
            None => {
                let op = Op::get(self.name).or_else(|missing| {
                    let registered = register_defined_ir();
                    Op::get(self.name).map_err(|_| match registered {
                        Ok(()) => missing,
                        Err(failed) => {
                            let message = format!("{}\n{}", missing.message(), failed.message());
                            Error::new(missing.kind(), &message, "")
                        }
                    })
                })?;
                FETCHED.with(|fetched| {
                    let mut fetched = fetched.borrow_mut();
                    if fetched.len() <= slot {
                        fetched.resize(slot + 1, None);
                    }
                    fetched[slot] = Some(op.clone());
                });
                op
            }
        };
        Ok(use_op(&op))
    }
}

/// A [`register_ops!`](crate::register_ops) or [`register_types!`](crate::register_types) block,
/// gathered at link time.
#[doc(hidden)]
pub struct Registration {
    /// The macro of the block, and its module and line.
    kind: &'static str,
    module: &'static str,
    line: u32,
    /// The version of the block's crate and a hash of the block's text, which with the module and
    /// line identify the block in every library that links the crate.
    version: &'static str,
    text_hash: u64,
    register: fn() -> Result<()>,
    /// The outcome of the block, which runs once: the error's text if it failed.
    outcome: OnceLock<std::result::Result<(), std::string::String>>,
}

/// The blocks of [`register_ops!`](crate::register_ops) and
/// [`register_types!`](crate::register_types), gathered at link time from the whole program or
/// library.
#[doc(hidden)]
#[linkme::distributed_slice]
pub static REGISTRATIONS: [Registration];

impl Registration {
    #[doc(hidden)]
    pub const fn __new(
        kind: &'static str,
        module: &'static str,
        line: u32,
        version: &'static str,
        text: &str,
        register: fn() -> Result<()>,
    ) -> Self {
        let text_hash = text_hash(text);
        Self { kind, module, line, version, text_hash, register, outcome: OnceLock::new() }
    }

    /// The identity of the block in the process, `module:line@version#hash`: the same in every
    /// library that links the block's crate, and another one if the crate version or the block's
    /// text differs.
    #[doc(hidden)]
    pub fn identity(&self) -> std::string::String {
        format!("{}:{}@{}#{:016x}", self.module, self.line, self.version, self.text_hash)
    }

    /// The name of the global function that marks the block registered in the process,
    /// `ir.registered_block.<identity>`. The libraries of a process share the global function
    /// table of TVM FFI, and their statics are their own.
    #[doc(hidden)]
    pub fn marker(&self) -> std::string::String {
        format!("ir.registered_block.{}", self.identity())
    }

    /// Run the block if it has not run. The caller that runs a failing block gets its error;
    /// later ones get an error that names the block.
    #[doc(hidden)]
    pub fn register(&self) -> Result<()> {
        let mut failure = None;
        let outcome = self.outcome.get_or_init(|| {
            self.run().map_err(|error| {
                let text = format!("{}: {}", error.kind(), error.message());
                failure = Some(error);
                text
            })
        });
        match (failure, outcome) {
            (Some(error), _) => Err(error),
            (None, Ok(())) => Ok(()),
            (None, Err(text)) => {
                let message = format!("The {} block at {} failed: {text}", self.kind, self.place());
                Err(Error::new(INTERNAL_ERROR, &message, ""))
            }
        }
    }

    /// Run the block unless another library of the process has run it. Every library links its
    /// own copy of the crates it is built from, so the same block can be in several, and what it
    /// defines is registered once, with the hooks of the first library that runs it. The block
    /// is marked once it succeeds, so that one that fails fails in every library.
    fn run(&self) -> Result<()> {
        let marker = self.marker();
        if Function::get_global(&marker).is_err() {
            (self.register)()?;
            Function::register_global(&marker, Function::from_packed(|_| Ok(Any::default())))?;
        }
        Ok(())
    }

    /// Where the block is, for an error.
    fn place(&self) -> std::string::String {
        format!("{}:{}", self.module, self.line)
    }
}

/// The FNV-1a hash of `text` without its whitespace, which does not depend on how the compiler
/// spaces the tokens of a block it stringifies.
const fn text_hash(text: &str) -> u64 {
    let bytes = text.as_bytes();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_whitespace() {
            hash = (hash ^ bytes[i] as u64).wrapping_mul(0x0100_0000_01b3);
        }
        i += 1;
    }
    hash
}

/// Run every [`register_ops!`](crate::register_ops) and [`register_types!`](crate::register_types)
/// block of this program or library that has not run, in the order of their modules and lines;
/// running again does nothing. Every block is tried, and the error lists those that failed. A
/// block that another library of the process has registered, from the same version of its crate
/// and the same text, is skipped: libraries that link the same crate register its operators and
/// types once.
///
/// The blocks run when their program or library loads, so this reports the errors of those that
/// failed. Every library built on this crate exports it as the TVM FFI function
/// `tirx_register_ir`, which `import tvm_tirx` calls on `libtvm_tirx_rust` to raise them. Where
/// the loader does not run the blocks, the first lookup through an [`OpHandle`] of an operator
/// that is not registered calls it.
pub fn register_defined_ir() -> Result<()> {
    let mut blocks: Vec<&Registration> = REGISTRATIONS.iter().collect();
    blocks.sort_by_key(|block| (block.module, block.line));
    let mut failures: Vec<(&Registration, Error)> =
        blocks.into_iter().filter_map(|block| block.register().err().map(|e| (block, e))).collect();
    if failures.len() <= 1 {
        return failures.pop().map_or(Ok(()), |(_, error)| Err(error));
    }
    let mut kinds: Vec<&str> = failures.iter().map(|(block, _)| block.kind).collect();
    kinds.sort();
    kinds.dedup();
    let mut message = format!("{} {} blocks failed:", failures.len(), kinds.join(" and "));
    for (block, error) in &failures {
        message.push_str(&format!("\n  {}: {}: {}", block.place(), error.kind(), error.message()));
    }
    Err(Error::new(failures[0].1.kind(), &message, ""))
}

/// The constructor of every [`register_ops!`](crate::register_ops) and
/// [`register_types!`](crate::register_types) block, which the loader runs when the program or
/// library loads: the first one keeps the library loaded and runs every block of it. An error
/// cannot reach a caller here, so it is printed; [`register_defined_ir`] reports it again.
#[doc(hidden)]
pub extern "C" fn __register_at_load() {
    static LOADED: Once = Once::new();
    LOADED.call_once(|| {
        let library = keep_loaded().unwrap_or_else(|| "register_ops!".to_string());
        match catch_unwind(register_defined_ir) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => eprintln!(
                "{library}: registering operators and types failed: {}: {}",
                error.kind(),
                error.message()
            ),
            Err(_) => eprintln!("{library}: registering operators and types panicked"),
        }
    });
}

/// Keep the library that holds this code loaded for the rest of the process, as the operators and
/// types it registers keep its hooks: unloading it, as `tvm_ffi.load_module` does when Python drops the
/// module, would leave them dangling. Returns the path of the library.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn keep_loaded() -> Option<std::string::String> {
    use std::ffi::{c_char, c_int, CStr};

    #[repr(C)]
    struct DlInfo {
        fname: *const c_char,
        fbase: *mut c_void,
        sname: *const c_char,
        saddr: *mut c_void,
    }
    extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    }
    // RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE.
    #[cfg(target_os = "linux")]
    const FLAGS: c_int = 0x1 | 0x4 | 0x1000;
    #[cfg(target_os = "macos")]
    const FLAGS: c_int = 0x1 | 0x10 | 0x80;

    let null = std::ptr::null_mut();
    let mut info = DlInfo { fname: null as _, fbase: null, sname: null as _, saddr: null };
    // SAFETY: `dladdr` fills `info` for an address of this library, whose path stays valid while
    // the library is loaded; `dlopen` of a loaded library only marks it, and its handle is kept.
    unsafe {
        if dladdr(keep_loaded as *const c_void, &mut info) == 0 || info.fname.is_null() {
            return None;
        }
        dlopen(info.fname, FLAGS);
        Some(CStr::from_ptr(info.fname).to_string_lossy().into_owned())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn keep_loaded() -> Option<std::string::String> {
    None
}

/// The text of `doc` without the indentation its lines share and without the blank lines around
/// it. Text that is not indented is unchanged.
fn doc_text(doc: &str) -> std::string::String {
    let indent = doc
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);
    let lines: Vec<&str> =
        doc.lines().map(|line| line.get(indent..).unwrap_or("").trim_end()).collect();
    lines.join("\n").trim().to_string()
}

/// Define operators as C++ does in a `TVM_FFI_STATIC_INIT_BLOCK`: each item is an [`OpDef`]
/// chain, which the block runs in order, stopping at the first that fails.
///
/// ```no_run
/// use tvm_tirx::ir::op_attr::{CALL_EFFECT_KIND, FIXED_RETURN_TYPE};
/// use tvm_tirx::ir::sig::{self, IntExpr};
/// use tvm_tirx::ir::{CallEffectKind, CallObj, Op, OpDef, PrimType, Type};
/// use tvm_tirx::tvm_ffi::Result;
///
/// fn first_arg_type(call: &CallObj) -> Result<Type> {
///     Ok(call.args.get(0)?.ty.clone())
/// }
///
/// tvm_tirx::register_ops! {
///     OpDef::new("my_ext.clamp", "Clamp an index to a bound.")
///         .signature((sig::arg::<IntExpr>("index", ""), sig::arg::<IntExpr>("bound", "")))
///         .set_infer_type(first_arg_type)
///         .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
///
///     OpDef::new("my_ext.mark", "
///         Mark a point in the program.
///
///         The documentation is dedented.
///     ")
///         .set_attr(FIXED_RETURN_TYPE, PrimType::void())
///         .set_attr(CALL_EFFECT_KIND, CallEffectKind::Opaque);
/// }
///
/// // Registered when the program loaded.
/// assert_eq!(Op::get("my_ext.clamp")?.doc.as_str(), "Clamp an index to a bound.");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
///
/// As a C++ static init block, a block runs when its program or library loads: it adds a
/// constructor, which the loader runs, and the first constructor runs every block of the program
/// or library with [`register_defined_ir`]. Python sees the operators of a library once it loads
/// it, by any means, and the library stays loaded for the process, as the operators keep its
/// hooks. An error cannot reach a caller when the library loads: it is printed, and
/// [`register_defined_ir`] (`tirx_register_ir`) and the handles report it again. Where the
/// loader does not run the constructor, an [`OpHandle`] runs the blocks on the first lookup of an
/// operator that is not registered.
///
/// A block runs once per process. Every library built on the crate links its own copy of the
/// crates it uses, so a block of a crate that several libraries link is in each of them; the first
/// library that runs it registers its operators, and the others skip it. A block of another
/// version of the crate, or whose text differs, is another block, and registering its operators
/// again is an error. A change to a hook's body alone does not make another block: rebuild every
/// library that links a crate after changing it.
#[macro_export]
macro_rules! register_ops {
    ($($def:expr);* $(;)?) => {
        $crate::__register_block!("register_ops!", stringify!($($def);*), || {
            $($crate::ir::OpDef::op($def)?;)*
            Ok(())
        });
    };
}

/// The registration of a [`register_ops!`](crate::register_ops) or
/// [`register_types!`](crate::register_types) block: gathered at link time, and run by a
/// constructor when its program or library loads.
#[doc(hidden)]
#[macro_export]
macro_rules! __register_block {
    ($kind:literal, $text:expr, $register:expr) => {
        const _: () = {
            #[$crate::__private::linkme::distributed_slice($crate::ir::REGISTRATIONS)]
            #[linkme(crate = $crate::__private::linkme)]
            static REGISTRATION: $crate::ir::Registration = $crate::ir::Registration::__new(
                $kind,
                module_path!(),
                line!(),
                match option_env!("CARGO_PKG_VERSION") {
                    Some(version) => version,
                    None => "",
                },
                $text,
                $register,
            );
            #[used]
            #[cfg_attr(target_os = "linux", link_section = ".init_array")]
            #[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
            #[cfg_attr(windows, link_section = ".CRT$XCU")]
            static REGISTER_AT_LOAD: extern "C" fn() = $crate::ir::__register_at_load;
        };
    };
}

/// The C ABI of a native operator hook, `NativeFunctionView<R(const CallNode*)>`: it returns
/// the result, or the error, in an owned `TVMFFIAny`.
type Hook = extern "C" fn(*const CallObj) -> TVMFFIAny;

/// The C ABI of a native hook on a region statement, `NativeFunctionView<R(const
/// RegionStmtNode*)>`.
type RegionHook = extern "C" fn(*const RegionStmtObj) -> TVMFFIAny;

extern "C" fn validate_abi<F: Fn(&CallObj) -> Result<()>>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn infer_type_abi<F: Fn(&CallObj) -> Result<Type>>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn validate_signature<L: Checks>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, sig::validate::<L>)
}

extern "C" fn region_body_params_abi<F: Fn(&CallObj) -> Result<Array<Var>>>(
    call: *const CallObj,
) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn region_validate_abi<F: Fn(&RegionStmtObj) -> Result<()>>(
    region: *const RegionStmtObj,
) -> TVMFFIAny {
    call_hook(region, zero_sized::<F>())
}

/// The only value of `F`, a function or a closure that captures nothing.
fn zero_sized<F>() -> F {
    const {
        assert!(
            std::mem::size_of::<F>() == 0,
            "a native operator hook is a function or a closure that captures nothing"
        )
    };
    // SAFETY: a zero-sized type has no bytes to initialize.
    unsafe { std::mem::zeroed() }
}

/// Run `hook` on the node C++ passes (the Call or region being typed or checked) and return its
/// result as the hook ABI does. A panic, which must not unwind into C++, becomes an error.
fn call_hook<N, R: Into<Any>>(node: *const N, hook: impl Fn(&N) -> Result<R>) -> TVMFFIAny {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: C++ passes the node being typed or checked, alive for the call.
        match unsafe { node.as_ref() } {
            Some(node) => hook(node).map(Into::into),
            None => Err(Error::new(TYPE_ERROR, "Invalid Op Call", "")),
        }
    }));
    let value = match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => Any::from(error),
        Err(payload) => Any::from(tvm_ffi::function_internal::panic_to_error(payload)),
    };
    // SAFETY: the C++ caller takes ownership of the value.
    unsafe { Any::into_raw_ffi_any(value) }
}

/// Signature descriptors, for [`OpDef::signature`] (C++ `sig::arg<T>` and friends): the arguments
/// an operator takes, with their documentation and the types a generated validator checks.
///
/// [`arg`](sig::arg) and [`ty_arg`](sig::ty_arg) are the fixed value and type arguments, in
/// order, [`var_args`](sig::var_args) and [`var_ty_args`](sig::var_ty_args) a variadic tail of
/// each, and [`call_attrs`](sig::call_attrs) the type of `Call.attrs`. As an argument type, a node
/// type, such as [`StringImm`](super::StringImm) or [`Var`], accepts an instance of it, as C++
/// `TypeTraits<T>::CheckAnyStrict` does, and [`Expr`] or [`Type`] accepts any.
/// [`PrimExpr`](sig::PrimExpr), [`IntExpr`](sig::IntExpr) and [`PrimVar`](sig::PrimVar) are the
/// C++ view types, which also check the expression's type, as is
/// [`TensorVar`](crate::tirx::TensorVar). Implement [`ArgType`](sig::ArgType) for another check.
pub mod sig {
    use super::*;
    use crate::ir::expr::ExprObj;
    use crate::ir::r#type::PrimTypeObj;
    use tvm_ffi::type_traits::AnyCompatible;
    use tvm_ffi::ObjectCore;

    /// What a signature checks an argument against (C++ `TypeTraits<T>`).
    pub trait ArgType: 'static {
        /// Whether `value`, which is not None, is accepted (C++ `CheckAnyStrict`).
        fn check(value: &TVMFFIAny) -> bool;
        /// The expected type, for an error (C++ `TypeStr`).
        fn type_str() -> std::string::String;
        /// The rejected value, for an error (C++ `GetMismatchTypeInfo`): its type key.
        fn mismatch(value: &TVMFFIAny) -> std::string::String {
            support::type_key(value.type_index)
        }
    }

    impl<T: ObjectRefCore + AnyCompatible + 'static> ArgType for T {
        fn check(value: &TVMFFIAny) -> bool {
            // SAFETY: `value` is a live FFI value.
            unsafe { T::check_any_strict(value) }
        }

        fn type_str() -> std::string::String {
            T::type_str()
        }
    }

    /// C++ `PrimExpr`: an expression whose type is a `PrimType`.
    pub struct PrimExpr;

    impl ArgType for PrimExpr {
        fn check(value: &TVMFFIAny) -> bool {
            prim_dtype(value).is_some()
        }

        fn type_str() -> std::string::String {
            "ir.PrimExpr".into()
        }

        fn mismatch(value: &TVMFFIAny) -> std::string::String {
            typed_expr_mismatch(value)
        }
    }

    /// C++ `IntExpr`: a `PrimExpr` of a scalar signed or unsigned integer type.
    pub struct IntExpr;

    impl ArgType for IntExpr {
        fn check(value: &TVMFFIAny) -> bool {
            prim_dtype(value).is_some_and(|dtype| {
                dtype.lanes == 1 && matches!(dtype.code, support::INT | support::UINT)
            })
        }

        fn type_str() -> std::string::String {
            "ir.IntExpr".into()
        }

        fn mismatch(value: &TVMFFIAny) -> std::string::String {
            typed_expr_mismatch(value)
        }
    }

    /// C++ `PrimVar`: a `Var`, not a subclass, whose type is a `PrimType`.
    pub struct PrimVar;

    impl ArgType for PrimVar {
        fn check(value: &TVMFFIAny) -> bool {
            var_of_type::<PrimTypeObj>(value)
        }

        fn type_str() -> std::string::String {
            crate::ir::expr::VarObj::TYPE_KEY.into()
        }
    }

    /// Whether `value` is a `Var`, not a subclass, whose type is an `N`.
    pub fn var_of_type<N: ObjectCore>(value: &TVMFFIAny) -> bool {
        value.type_index == crate::ir::expr::VarObj::type_index()
            && expr(value).is_some_and(|var| var.ty.as_node::<N>().is_some())
    }

    /// The expression `value` holds, if it holds one.
    fn expr(value: &TVMFFIAny) -> Option<&ExprObj> {
        let is_object = value.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32;
        // SAFETY: an object value of an `ir.Expr` type points to an `ExprObj`.
        unsafe {
            (is_object && Expr::check_any_strict(value))
                .then(|| &*(value.data_union.v_obj as *const ExprObj))
        }
    }

    fn prim_dtype(value: &TVMFFIAny) -> Option<tvm_ffi::DLDataType> {
        Some(expr(value)?.ty.as_node::<PrimTypeObj>()?.dtype)
    }

    /// C++ `TypeTraits<TypedExpr<T>>::GetMismatchTypeInfo`: an expression with its type.
    fn typed_expr_mismatch(value: &TVMFFIAny) -> std::string::String {
        let Some(expr) = expr(value) else { return support::type_key(value.type_index) };
        let ty = match expr.ty.as_node::<PrimTypeObj>() {
            Some(prim) => support::dtype_name(prim.dtype),
            None => support::type_key(AnyView::from(&expr.ty).type_index()),
        };
        format!("{}[ty={ty}]", support::type_key(value.type_index))
    }

    /// A required value argument of type `T` (C++ `sig::arg<T>`); [`Expr`] accepts any.
    pub fn arg<T: ArgType>(name: &str, doc: &str) -> Arg<T> {
        Arg { name: String::from(name), doc: String::from(doc), ty: PhantomData }
    }

    /// Zero or more further value arguments of type `T` (C++ `sig::var_args<T>`).
    pub fn var_args<T: ArgType>(name: &str, doc: &str) -> VarArgs<T> {
        VarArgs { name: String::from(name), doc: String::from(doc), ty: PhantomData }
    }

    /// A required type argument of type `T` (C++ `sig::ty_arg<T>`); [`Type`] accepts any.
    pub fn ty_arg<T: ArgType>(name: &str, doc: &str) -> TyArg<T> {
        TyArg { name: String::from(name), doc: String::from(doc), ty: PhantomData }
    }

    /// Zero or more further type arguments of type `T` (C++ `sig::var_ty_args<T>`).
    pub fn var_ty_args<T: ArgType>(name: &str, doc: &str) -> VarTyArgs<T> {
        VarTyArgs { name: String::from(name), doc: String::from(doc), ty: PhantomData }
    }

    /// `Call.attrs` holds a `T`, which is also the operator's attrs type (C++
    /// `sig::call_attrs<T>`).
    pub fn call_attrs<T>() -> CallAttrs<T>
    where
        T: ObjectRefCore + Into<Attrs> + 'static,
    {
        CallAttrs { ty: PhantomData }
    }

    /// A required value argument of type `T`, from [`arg`].
    pub struct Arg<T> {
        name: String,
        doc: String,
        ty: PhantomData<fn() -> T>,
    }

    /// The variadic value tail, of type `T`, from [`var_args`].
    pub struct VarArgs<T> {
        name: String,
        doc: String,
        ty: PhantomData<fn() -> T>,
    }

    /// A required type argument of type `T`, from [`ty_arg`].
    pub struct TyArg<T> {
        name: String,
        doc: String,
        ty: PhantomData<fn() -> T>,
    }

    /// The variadic type-argument tail, of type `T`, from [`var_ty_args`].
    pub struct VarTyArgs<T> {
        name: String,
        doc: String,
        ty: PhantomData<fn() -> T>,
    }

    /// `Call.attrs`, of type `T`, from [`call_attrs`].
    pub struct CallAttrs<T> {
        ty: PhantomData<fn() -> T>,
    }

    /// The argument descriptions a signature gives its operator.
    #[doc(hidden)]
    #[derive(Default)]
    pub struct Description {
        arg_names: Vec<String>,
        arg_docs: Vec<String>,
        ty_arg_names: Vec<String>,
        ty_arg_docs: Vec<String>,
        /// Empty, or the tail's name and documentation.
        var_args: Vec<String>,
        var_ty_args: Vec<String>,
    }

    impl Description {
        /// The lists `ir.Op._set_signature` takes.
        pub(super) fn into_lists(self) -> [Vec<String>; 6] {
            let Self { arg_names, arg_docs, ty_arg_names, ty_arg_docs, var_args, var_ty_args } =
                self;
            [arg_names, arg_docs, ty_arg_names, ty_arg_docs, var_args, var_ty_args]
        }
    }

    /// One descriptor of a signature: what it adds to the description, and what it checks.
    #[doc(hidden)]
    pub trait Descriptor: 'static {
        const ARGS: usize = 0;
        const TY_ARGS: usize = 0;
        const VAR_ARGS: usize = 0;
        const VAR_TY_ARGS: usize = 0;
        const CALL_ATTRS: usize = 0;
        /// The type key of the attrs type, if the descriptor names one.
        const ATTRS_TYPE_KEY: Option<&'static str> = None;

        fn describe(self, description: &mut Description);

        /// Check what this describes of `call`. `fixed` counts the signature's fixed value and
        /// type arguments, where its tails start, and `at` those declared before this one.
        fn check(
            call: &CallObj,
            op: &OpObj,
            fixed: (usize, usize),
            at: (usize, usize),
        ) -> Result<()>;
    }

    impl<T: ArgType> Descriptor for Arg<T> {
        const ARGS: usize = 1;

        fn describe(self, description: &mut Description) {
            description.arg_names.push(self.name);
            description.arg_docs.push(self.doc);
        }

        fn check(call: &CallObj, op: &OpObj, _: (usize, usize), at: (usize, usize)) -> Result<()> {
            check_value::<T, Expr>(op, false, support::array_cells(&call.args), at.0)
        }
    }

    impl<T: ArgType> Descriptor for VarArgs<T> {
        const VAR_ARGS: usize = 1;

        fn describe(self, description: &mut Description) {
            description.var_args = vec![self.name, self.doc];
        }

        fn check(
            call: &CallObj,
            op: &OpObj,
            fixed: (usize, usize),
            _: (usize, usize),
        ) -> Result<()> {
            let values = support::array_cells(&call.args);
            (fixed.0..values.len()).try_for_each(|i| check_value::<T, Expr>(op, false, values, i))
        }
    }

    impl<T: ArgType> Descriptor for TyArg<T> {
        const TY_ARGS: usize = 1;

        fn describe(self, description: &mut Description) {
            description.ty_arg_names.push(self.name);
            description.ty_arg_docs.push(self.doc);
        }

        fn check(call: &CallObj, op: &OpObj, _: (usize, usize), at: (usize, usize)) -> Result<()> {
            check_value::<T, Type>(op, true, support::array_cells(&call.ty_args), at.1)
        }
    }

    impl<T: ArgType> Descriptor for VarTyArgs<T> {
        const VAR_TY_ARGS: usize = 1;

        fn describe(self, description: &mut Description) {
            description.var_ty_args = vec![self.name, self.doc];
        }

        fn check(
            call: &CallObj,
            op: &OpObj,
            fixed: (usize, usize),
            _: (usize, usize),
        ) -> Result<()> {
            let values = support::array_cells(&call.ty_args);
            (fixed.1..values.len()).try_for_each(|i| check_value::<T, Type>(op, true, values, i))
        }
    }

    impl<T: ObjectRefCore + 'static> Descriptor for CallAttrs<T> {
        const CALL_ATTRS: usize = 1;
        const ATTRS_TYPE_KEY: Option<&'static str> = Some(T::ContainerType::TYPE_KEY);

        fn describe(self, _: &mut Description) {}

        fn check(call: &CallObj, op: &OpObj, _: (usize, usize), _: (usize, usize)) -> Result<()> {
            let actual = match &call.attrs {
                Some(attrs) if attrs.as_node::<T::ContainerType>().is_some() => return Ok(()),
                Some(attrs) => {
                    format!("`{}`", support::type_key(AnyView::from(attrs).type_index()))
                }
                None => "None".into(),
            };
            let message = format!(
                "Op `{}`: Call.attrs expected `{}`, got {actual}",
                op.name.as_str(),
                T::ContainerType::TYPE_KEY,
            );
            Err(Error::new(TYPE_ERROR, &message, ""))
        }
    }

    /// The checks of a signature, as a list of descriptors in declaration order.
    #[doc(hidden)]
    pub trait Checks: 'static {
        const ARGS: usize;
        const TY_ARGS: usize;
        const VAR_ARGS: usize;
        const VAR_TY_ARGS: usize;
        const CALL_ATTRS: usize;
        const ATTRS_TYPE_KEY: Option<&'static str>;

        /// Check the arguments of `call` these descriptors describe, in declaration order, as
        /// [`Descriptor::check`] does.
        fn check(
            call: &CallObj,
            op: &OpObj,
            fixed: (usize, usize),
            at: (usize, usize),
        ) -> Result<()>;
    }

    impl Checks for () {
        const ARGS: usize = 0;
        const TY_ARGS: usize = 0;
        const VAR_ARGS: usize = 0;
        const VAR_TY_ARGS: usize = 0;
        const CALL_ATTRS: usize = 0;
        const ATTRS_TYPE_KEY: Option<&'static str> = None;

        fn check(_: &CallObj, _: &OpObj, _: (usize, usize), _: (usize, usize)) -> Result<()> {
            Ok(())
        }
    }

    impl<D: Descriptor, L: Checks> Checks for (D, L) {
        const ARGS: usize = D::ARGS + L::ARGS;
        const TY_ARGS: usize = D::TY_ARGS + L::TY_ARGS;
        const VAR_ARGS: usize = D::VAR_ARGS + L::VAR_ARGS;
        const VAR_TY_ARGS: usize = D::VAR_TY_ARGS + L::VAR_TY_ARGS;
        const CALL_ATTRS: usize = D::CALL_ATTRS + L::CALL_ATTRS;
        const ATTRS_TYPE_KEY: Option<&'static str> = match D::ATTRS_TYPE_KEY {
            Some(key) => Some(key),
            None => L::ATTRS_TYPE_KEY,
        };

        fn check(
            call: &CallObj,
            op: &OpObj,
            fixed: (usize, usize),
            at: (usize, usize),
        ) -> Result<()> {
            D::check(call, op, fixed, at)?;
            L::check(call, op, fixed, (at.0 + D::ARGS, at.1 + D::TY_ARGS))
        }
    }

    /// What [`OpDef::signature`] takes: a descriptor, or a tuple of them in order; `()` declares
    /// no argument.
    pub trait Specs {
        #[doc(hidden)]
        type Checks: Checks;

        #[doc(hidden)]
        fn describe(self, description: &mut Description);
    }

    impl<D: Descriptor> Specs for D {
        type Checks = (D, ());

        fn describe(self, description: &mut Description) {
            Descriptor::describe(self, description);
        }
    }

    /// `Specs` of the tuples of descriptors, whose checks nest in declaration order.
    macro_rules! tuple_specs {
        (@nest) => { () };
        (@nest $head:ident $(, $rest:ident)*) => { ($head, tuple_specs!(@nest $($rest),*)) };
        ($($name:ident),*) => {
            impl<$($name: Descriptor),*> Specs for ($($name,)*) {
                type Checks = tuple_specs!(@nest $($name),*);

                #[allow(non_snake_case, unused_variables)]
                fn describe(self, description: &mut Description) {
                    let ($($name,)*) = self;
                    $(Descriptor::describe($name, description);)*
                }
            }
        };
    }

    tuple_specs!();
    tuple_specs!(A);
    tuple_specs!(A, B);
    tuple_specs!(A, B, C);
    tuple_specs!(A, B, C, D);
    tuple_specs!(A, B, C, D, E);
    tuple_specs!(A, B, C, D, E, F);
    tuple_specs!(A, B, C, D, E, F, G);
    tuple_specs!(A, B, C, D, E, F, G, H);
    tuple_specs!(A, B, C, D, E, F, G, H, I);
    tuple_specs!(A, B, C, D, E, F, G, H, I, J);
    tuple_specs!(A, B, C, D, E, F, G, H, I, J, K);
    tuple_specs!(A, B, C, D, E, F, G, H, I, J, K, L);

    /// C++ `ValidateSignature<Specs...>`: the argument counts, then each argument's type and the
    /// attrs, in declaration order.
    pub(super) fn validate<L: Checks>(call: &CallObj) -> Result<()> {
        const {
            assert!(
                L::VAR_ARGS <= 1 && L::VAR_TY_ARGS <= 1 && L::CALL_ATTRS <= 1,
                "a signature has at most one variadic tail of each kind and one call_attrs"
            )
        };
        let Some(op) = call.op.as_node::<OpObj>() else {
            return Err(Error::new(TYPE_ERROR, "Invalid Op Call", ""));
        };
        let count = |label: &str, expected: usize, tail: bool, unit: &str, got: usize| {
            let short = if tail { got < expected } else { got != expected };
            if !short {
                return Ok(());
            }
            let at_least = if tail { "at least " } else { "" };
            let plural = if expected == 1 { "" } else { "s" };
            let message = format!(
                "Op `{}`: {label} expected {at_least}{expected} {unit}{plural}, got {got}",
                op.name.as_str()
            );
            Err(Error::new(TYPE_ERROR, &message, ""))
        };
        count("Call.args", L::ARGS, L::VAR_ARGS == 1, "argument", call.args.len())?;
        count(
            "Call.ty_args",
            L::TY_ARGS,
            L::VAR_TY_ARGS == 1,
            "type argument",
            call.ty_args.len(),
        )?;
        L::check(call, op, (L::ARGS, L::TY_ARGS), (0, 0))
    }

    /// Check argument `index` against `T`. `Base` is the category's base type, which C++ does
    /// not check.
    fn check_value<T: ArgType, Base: 'static>(
        op: &OpObj,
        type_arg: bool,
        values: &[TVMFFIAny],
        index: usize,
    ) -> Result<()> {
        if TypeId::of::<T>() == TypeId::of::<Base>() {
            return Ok(());
        }
        let value = &values[index];
        if value.type_index != TVMFFITypeIndex::kTVMFFINone as i32 && T::check(value) {
            return Ok(());
        }
        let (fixed, tail) = match type_arg {
            false => (&op.args_info, &op.var_args_info),
            true => (&op.ty_args_info, &op.var_ty_args_info),
        };
        let name = match fixed.get(index) {
            Ok(info) => Some(info.name.clone()),
            Err(_) => tail.as_ref().map(|info| info.name.clone()),
        };
        let name = match name {
            Some(name) if !name.as_str().is_empty() => format!(" (`{}`)", name.as_str()),
            _ => std::string::String::new(),
        };
        let message = format!(
            "Op `{}`: `{}[{index}]`{name} expected `{}`, got `{}`.",
            op.name.as_str(),
            if type_arg { "Call.ty_args" } else { "Call.args" },
            T::type_str(),
            T::mismatch(value),
        );
        Err(Error::new(TYPE_ERROR, &message, ""))
    }
}
