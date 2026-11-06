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

//! Operator registration from Rust: C++ `OpDef`.
//!
//! [`OpDef`] registers an operator with its signature ([`sig`]) and attributes ([`op_attr`]) in a
//! [`tvm_ffi::static_init_block!`], as C++ does in a `TVM_FFI_STATIC_INIT_BLOCK`. Its hooks are
//! native functions that C++ calls without packing arguments.

use std::any::TypeId;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use tvm_ffi::{Any, AnyView, Array, Error, ObjectRefCore, Result, String, TYPE_ERROR, VALUE_ERROR};

use super::attrs::Attrs;
use super::expr::{CallObj, Expr, Var};
use super::op::{Op, OpObj};
use super::r#type::Type;
use super::stmt::RegionStmtObj;
use super::support;
use sig::{Checks, Specs};

/// A chain of steps that registers a new operator from Rust, C++ `OpDef`.
///
/// Each step updates the registry at once, and a failed step skips the rest. The chain completes
/// when it is dropped, panicking with the error of a failed step, or by [`OpDef::op`], which
/// returns it. Register an operator before building IR with it, as a
/// [`tvm_ffi::static_init_block!`] does.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::op_attr::CALL_EFFECT_KIND;
/// use tvm_tirx::ir::sig::{self, IntExpr};
/// use tvm_tirx::ir::{
///     Call, CallEffectKind, CallObj, Op, OpDef, PrimType, PrimVar, Type, Var, UNKNOWN_LOC,
/// };
/// use tvm_tirx::script::script;
/// use tvm_tirx::tvm_ffi::{self, Result};
///
/// fn first_arg_type(call: &CallObj) -> Result<Type> {
///     Ok(call.args.get(0)?.ty.clone())
/// }
///
/// tvm_ffi::static_init_block! {
///     OpDef::new("my_ext.clamp", "Clamp an index to a bound.")
///         .signature((sig::arg::<IntExpr>("index", ""), sig::arg::<IntExpr>("bound", "")))
///         .set_infer_type(first_arg_type)
///         .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
/// }
///
/// let clamp = Op::get("my_ext.clamp")?;
/// let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let call = Call::new(None, clamp, [&i, &i], None, Vec::<Type>::new(), UNKNOWN_LOC)?;
/// assert_eq!(script(&call.ty)?, "T.int32");
/// call.validate()?;
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub struct OpDef(Option<Result<Op>>);

impl OpDef {
    /// Register a new operator `name` and start its chain (`ir.RegisterOp`).
    ///
    /// The indentation `doc` shares is removed. A name already registered fails the chain with an
    /// `AttributeError`.
    pub fn new(name: &str, doc: &str) -> Self {
        let registered = tvm_ffi::cached_global_func!("ir.RegisterOp")
            .call_tuple((String::from(name), String::from(doc_text(doc))));
        Self(Some(registered.and_then(|_| Op::get(name))))
    }

    /// Complete the chain now, instead of when it is dropped, and return the operator.
    ///
    /// # Errors
    ///
    /// Returns the error of the step that failed, which dropping the chain would panic with, or of
    /// the `ir.expose_op` receiver.
    pub fn op(mut self) -> Result<Op> {
        complete(self.take())
    }

    /// Set the signature of the operator to `specs`, a descriptor of [`sig`] or a tuple of them in
    /// order, as C++ `OpDef::signature` does.
    ///
    /// The validator generated from it checks a Call's argument counts, argument types and attrs,
    /// unless a custom validator is set.
    pub fn signature<S: Specs>(self, specs: S) -> Self {
        let mut description = sig::Description::default();
        specs.describe(&mut description);
        let lists = description.into_lists().map(Array::from_vec);
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

    /// Append the description of the value argument `name`, without changing what the validator
    /// checks.
    pub fn add_arg(self, name: &str, doc: &str) -> Self {
        let (name, doc) = (String::from(name), String::from(doc));
        self.invoke("add_arg", &[AnyView::from(&name), AnyView::from(&doc)])
    }

    /// Append the description of the type argument `name`, without changing what the validator
    /// checks.
    pub fn add_ty_arg(self, name: &str, doc: &str) -> Self {
        let (name, doc) = (String::from(name), String::from(doc));
        self.invoke("add_ty_arg", &[AnyView::from(&name), AnyView::from(&doc)])
    }

    /// Set the type of `Call.attrs` to the type registered as `key`; an unregistered `key` fails
    /// the chain with an `InternalError`.
    pub fn attrs_type_key(self, key: &str) -> Self {
        let key = String::from(key);
        self.invoke("set_attrs_type_key", &[AnyView::from(&key)])
    }

    /// Register the value of the attribute `key` for the operator, as C++ `set_attr(key, value)`
    /// does; a value already registered fails the chain with a `ValueError`.
    pub fn set_attr<T: Into<Any>>(self, key: AttrKey<T>, value: impl Into<T>) -> Self {
        let value: T = value.into();
        self.attr(key.name, value.into(), false)
    }

    /// Register the value of the attribute `key` for the operator, replacing a value already
    /// registered, as C++ `set_attr(key, value, true)` does.
    pub fn replace_attr<T: Into<Any>>(self, key: AttrKey<T>, value: impl Into<T>) -> Self {
        let value: T = value.into();
        self.attr(key.name, value.into(), true)
    }

    /// Remove the operator's value of the attribute `key`; a missing value is ignored, and live
    /// [`OpAttrMap`](crate::ir::OpAttrMap)s see the removal.
    pub fn reset_attr<T>(self, key: AttrKey<T>) -> Self {
        let key = String::from(key.name);
        self.invoke("reset_attr", &[AnyView::from(&key)])
    }

    /// Set the type inference of the operator, its `FInferType`, to `infer`, a function or a
    /// closure that captures nothing.
    ///
    /// It gives the result type of a Call that has none; [`op_attr::FIXED_RETURN_TYPE`] takes
    /// precedence. A second one fails the chain with a `ValueError`.
    pub fn set_infer_type<F>(self, infer: F) -> Self
    where
        F: Fn(&CallObj) -> Result<Type> + 'static,
    {
        let _ = infer;
        let hook: Hook = infer_type_abi::<F>;
        self.attr("FInferType", Any::from(hook as *mut c_void), false)
    }

    /// Set a custom validator of the operator, `validate`, a function or a closure that captures
    /// nothing.
    ///
    /// [`Call::validate`](crate::ir::Call::validate) runs it instead of the generated one. A second
    /// one fails the chain with a `ValueError`.
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

    /// Set the body parameters of the operator's regions, its `FRegionGetBodyParams`, to
    /// `body_params`, a function or a closure that captures nothing; this makes the operator a
    /// region operation.
    ///
    /// The `RegionStmt` constructor checks a region's body parameters against it. A second hook
    /// fails the chain with a `ValueError`.
    pub fn set_region_body_params<F>(self, body_params: F) -> Self
    where
        F: Fn(&CallObj) -> Result<Array<Var>> + 'static,
    {
        let _ = body_params;
        let hook: Hook = region_body_params_abi::<F>;
        self.attr("FRegionGetBodyParams", Any::from(hook as *mut c_void), false)
    }

    /// Set the check of the operator's regions, its `FRegionValidate`, to `validate`, a function or
    /// a closure that captures nothing.
    ///
    /// The `RegionStmt` constructor runs it on a complete region statement. A second check fails
    /// the chain with a `ValueError`.
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
            tvm_ffi::reflection::get_method("ir.Op", method)?.call_packed(&packed)?;
            Ok(())
        })
    }

    fn then(mut self, step: impl FnOnce(&Op) -> Result<()>) -> Self {
        Self(Some(self.take().and_then(|op| step(&op).map(|_| op))))
    }

    /// The state of the chain, which leaves nothing for `drop` to complete.
    fn take(&mut self) -> Result<Op> {
        self.0.take().expect("an `OpDef` is completed once")
    }
}

impl Drop for OpDef {
    fn drop(&mut self) {
        // Dropped while a panic unwinds, the chain did not finish, and a second panic would abort.
        if std::thread::panicking() {
            return;
        }
        if let Some(Err(error)) = self.0.take().map(complete) {
            panic!("{}: {}", error.kind(), error.message());
        }
    }
}

/// Request the APIs of a defined operator in registered destinations through `ir.expose_op`,
/// after its signature and attributes are ready. An absent receiver is a no-op; an installed
/// receiver's error is returned.
fn complete(state: Result<Op>) -> Result<Op> {
    let op = state?;
    if let Ok(expose) = tvm_ffi::Function::get_global("ir.expose_op") {
        expose.call_tuple((op.name.clone(),))?;
    }
    Ok(op)
}

/// The key of an operator attribute, C++ `ffi::reflection::AttrKey<T>`: its name and the type `T`
/// of its values.
///
/// [`op_attr`] and [`tirx::op_attr`](crate::tirx::op_attr) have the keys C++ declares;
/// [`AttrKey::new`] names another.
pub struct AttrKey<T> {
    name: &'static str,
    value: PhantomData<fn() -> T>,
}

impl<T> AttrKey<T> {
    /// Create the key of the attribute `name`, whose values are `T`.
    pub const fn new(name: &'static str) -> Self {
        Self { name, value: PhantomData }
    }

    /// Return the name of the attribute, which [`Op::attr_map`] reads the column of.
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

/// The keys of the operator attributes of the shared IR, C++ `tvm::op_attr`.
pub mod op_attr {
    use super::{AttrKey, CallEffectKind};
    use crate::ir::r#type::Type;

    /// The runtime effect of a call (C++ `kCallEffectKind`, `TCallEffectKind`).
    pub const CALL_EFFECT_KIND: AttrKey<CallEffectKind> = AttrKey::new("TCallEffectKind");
    /// The result type of every call, which takes precedence over `FInferType` (C++
    /// `kFixedReturnType`, `TFixedReturnType`).
    pub const FIXED_RETURN_TYPE: AttrKey<Type> = AttrKey::new("TFixedReturnType");
}

/// The runtime effect of an operator call, C++ `CallEffectKind`: the value of
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
        Err(payload) => Any::from(tvm_ffi::Error::from_panic(payload)),
    };
    // The C++ caller takes ownership of the value.
    Any::into_raw_ffi_any(value)
}

/// Signature descriptors for [`OpDef::signature`], C++ `sig::arg<T>` and friends.
///
/// [`arg`](sig::arg) and [`ty_arg`](sig::ty_arg) are fixed value and type arguments,
/// [`var_args`](sig::var_args) and [`var_ty_args`](sig::var_ty_args) variadic tails, and
/// [`call_attrs`](sig::call_attrs) the type of `Call.attrs`. A node type accepts an instance of it,
/// and the typed views such as [`PrimExpr`](super::PrimExpr) and [`PrimVar`](super::PrimVar)
/// also check its type; implement [`ArgType`](sig::ArgType) for another check.
pub mod sig {
    use super::*;
    use crate::ir::expr::ExprObj;
    use crate::ir::r#type::PrimTypeObj;
    use tvm_ffi::type_traits::AnyCompatible;
    use tvm_ffi::ObjectCore;

    /// A check of an argument against a type, C++ `TypeTraits<T>`.
    ///
    /// Every managed reference that is [`AnyCompatible`] implements it with its strict check;
    /// implement it for another check, as [`TensorVar`](crate::tirx::TensorVar) does.
    pub trait ArgType: 'static {
        /// Return whether `value`, which is not None, is accepted, as C++ `CheckAnyStrict` does.
        fn check(value: &TVMFFIAny) -> bool;
        /// Return the expected type, for an error, as C++ `TypeStr` does.
        fn type_str() -> std::string::String;
        /// Describe a rejected value, for an error, as C++ `GetMismatchTypeInfo` does.
        ///
        /// By default, the description is the type key of `value`.
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

        fn mismatch(value: &TVMFFIAny) -> std::string::String {
            T::get_mismatch_type_info(value)
        }
    }

    /// The typed views, which are argument types as they are: [`PrimVar`] checks as C++
    /// `PrimVar` does.
    pub use crate::ir::expr::{IntExpr, PrimExpr, PrimVar};

    /// Return whether `value` is a [`Var`], not a subclass, whose type is an `N`, such as
    /// `PrimTypeObj`: the check of [`PrimVar`] and of other variable argument types.
    pub fn var_of_type<N: ObjectCore>(value: &TVMFFIAny) -> bool {
        value.type_index == crate::ir::expr::VarObj::type_index()
            && expr(value).is_some_and(|var| var.ty.downcast_ref::<N>().is_some())
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

    /// Describe a required value argument `name` of type `T`, C++ `sig::arg<T>`; [`Expr`] accepts
    /// any.
    pub fn arg<T: ArgType>(name: &str, doc: &str) -> Arg<T> {
        Arg { name: String::from(name), doc: String::from(doc), ty: PhantomData }
    }

    /// Describe zero or more further value arguments of type `T`, C++ `sig::var_args<T>`; a
    /// signature has at most one variadic value tail.
    pub fn var_args<T: ArgType>(name: &str, doc: &str) -> VarArgs<T> {
        VarArgs { name: String::from(name), doc: String::from(doc), ty: PhantomData }
    }

    /// Describe a required type argument `name` of type `T`, C++ `sig::ty_arg<T>`; [`Type`] accepts
    /// any.
    pub fn ty_arg<T: ArgType>(name: &str, doc: &str) -> TyArg<T> {
        TyArg { name: String::from(name), doc: String::from(doc), ty: PhantomData }
    }

    /// Describe zero or more further type arguments of type `T`, C++ `sig::var_ty_args<T>`; a
    /// signature has at most one variadic type-argument tail.
    pub fn var_ty_args<T: ArgType>(name: &str, doc: &str) -> VarTyArgs<T> {
        VarTyArgs { name: String::from(name), doc: String::from(doc), ty: PhantomData }
    }

    /// Describe the attributes of a Call: `Call.attrs` holds a `T`, which is also the operator's
    /// attrs type, C++ `sig::call_attrs<T>`; a signature has at most one.
    pub fn call_attrs<T>() -> CallAttrs<T>
    where
        T: ObjectRefCore + Into<Attrs> + 'static,
    {
        CallAttrs { ty: PhantomData }
    }

    /// A descriptor of a required value argument of type `T`, made by [`arg`].
    pub struct Arg<T> {
        name: String,
        doc: String,
        ty: PhantomData<fn() -> T>,
    }

    /// A descriptor of a variadic value tail of type `T`, made by [`var_args`].
    pub struct VarArgs<T> {
        name: String,
        doc: String,
        ty: PhantomData<fn() -> T>,
    }

    /// A descriptor of a required type argument of type `T`, made by [`ty_arg`].
    pub struct TyArg<T> {
        name: String,
        doc: String,
        ty: PhantomData<fn() -> T>,
    }

    /// A descriptor of a variadic type-argument tail of type `T`, made by [`var_ty_args`].
    pub struct VarTyArgs<T> {
        name: String,
        doc: String,
        ty: PhantomData<fn() -> T>,
    }

    /// A descriptor of `Call.attrs` of type `T`, made by [`call_attrs`].
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
                Some(attrs) if attrs.downcast_ref::<T::ContainerType>().is_some() => return Ok(()),
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

    /// The descriptors that [`OpDef::signature`] takes: a descriptor, or a tuple of them in order.
    ///
    /// `()` declares no argument.
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
        let Some(op) = call.op.downcast_ref::<OpObj>() else {
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
