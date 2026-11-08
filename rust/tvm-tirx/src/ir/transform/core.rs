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

//! Passes. A [`Pass`] transforms an [`IRModule`] under the current [`PassContext`], which decides
//! the passes that run and the instruments that observe them; [`ModulePass`] wraps a function and
//! [`Sequential`] runs passes in order.

// tvm-ffi-stubgen(missing-docs): allow
// `custom-new` reserves `new` for the checked `transform.PassContext`; the generated `from_fields`
// allocates the fields as given.
// tvm-ffi-stubgen(custom-new): transform.PassContext
// tvm-ffi-stubgen(field): transform.PassInfo.opt_level -> i32
// tvm-ffi-stubgen(field): transform.PassContext.opt_level -> i32
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::module::IRModule
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): tvm_ffi::IntoArg
// tvm-ffi-stubgen(import-object): std::borrow::Borrow

// tvm-ffi-stubgen(begin): import-section
use crate::ir::module::IRModule;
use crate::ir::support;
use std::borrow::Borrow;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::IntoArg;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::cached_field_getter;
// tvm-ffi-stubgen(end)

/// A transformation of an [`IRModule`], with a name and an optimization level ([`PassInfo`]).
///
/// [`run`](Self::run) applies it under the current [`PassContext`].
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::transform::{ModulePass, Pass, PassContext, PassInfo, PassInstrument};
/// use tvm_tirx::ir::transform::Sequential;
/// use tvm_tirx::ir::{DictAttrs, IRModule, IntImm};
/// use tvm_tirx::tvm_ffi::Map;
///
/// let mark = |name: &'static str| {
///     let pass = move |module: IRModule, _ctx| module.with_attr(name, IntImm::bool(true));
///     ModulePass::new(pass, PassInfo::new(0, name))
/// };
///
/// let info = PassInfo::new(0, "pipeline");
/// let pipeline = Pass::from(Sequential::new([mark("a"), mark("b")], info));
/// let ctx = PassContext::new(2, [""; 0], ["b"], Vec::<PassInstrument>::new(), None)?;
/// let _scope = ctx.enter()?;
/// let empty = IRModule::new(Map::new(), DictAttrs::new(Map::new()), Map::new())?;
/// let module = pipeline.run(empty)?;
/// assert!(module.get_attr("a")?.is_some() && module.get_attr("b")?.is_none());
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/transform.Pass
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Pass {
    data_: ObjectArc<PassObj>,
}

/// The object behind [`Pass`].
///
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "transform.Pass")]
pub struct PassObj {
    base_: Object,
}
// tvm-ffi-stubgen(end)

impl Pass {
    /// Return the name and optimization level of the pass (`transform.Info`).
    pub fn info(&self) -> Result<PassInfo> {
        tvm_ffi::cached_global_func!("transform.Info").call_tuple((self,))?.try_into()
    }

    /// Run the pass on `module`, which it consumes, under the current pass context
    /// (`transform.RunPass`).
    ///
    /// An instrument's `should_run` hook may skip the pass, unless the context requires it.
    pub fn run(&self, module: impl IntoArg<IRModule>) -> Result<IRModule> {
        let module = tvm_ffi::rvalue_ref::RValueRef::new(module.into_owned());
        tvm_ffi::cached_global_func!("transform.RunPass")
            .call_packed(&[tvm_ffi::AnyView::from(self), tvm_ffi::AnyView::from(&module)])?
            .try_into()
    }
}

/// The name `name` and optimization level `opt_level` of a pass, by which a [`PassContext`]
/// requires, disables or skips it.
// tvm-ffi-stubgen(begin): object/transform.PassInfo
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PassInfo {
    data_: ObjectArc<PassInfoObj>,
}

/// The object behind [`PassInfo`].
///
/// Complete: reflected fields fill [24, 48) exactly (alignment padding [28, 32)).
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "transform.PassInfo", final, immutable)]
pub struct PassInfoObj {
    base_: Object,
    pub opt_level: i32,
    pub name: String,
}

const _: () = {
    assert!(::core::mem::size_of::<PassInfoObj>() == 48);
    assert!(::core::mem::align_of::<PassInfoObj>() == 8);
};

#[allow(missing_docs)]
impl PassInfoObj {
    pub(crate) fn new(opt_level: i32, name: String) -> Self {
        let base_ = Object::new();
        Self { base_, opt_level, name }
    }
}

#[allow(missing_docs)]
impl PassInfo {
    /// Lossless complete-field allocation.
    pub fn new(opt_level: i32, name: impl Into<String>) -> Self {
        let obj = PassInfoObj::new(opt_level, name.into());
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// `PassInfo::new` above has the semantics of the native `transform.PassInfo`, which only stores
// its fields.

/// A pass that applies a function to the whole module.
// tvm-ffi-stubgen(begin): object/transform.ModulePass
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ModulePass {
    data_: ObjectArc<ModulePassObj>,
}

/// The object behind [`ModulePass`].
///
/// Opaque: parent 'transform.Pass' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "transform.ModulePass", final, immutable)]
pub struct ModulePassObj {
    base_: PassObj,
}

impl Deref for ModulePassObj {
    type Target = PassObj;
    fn deref(&self) -> &PassObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl ModulePassObj {
    pub fn pass_info(&self) -> Result<PassInfo> {
        cached_field_getter!(ModulePassObj, "pass_info").get(self)
    }
}

tvm_ffi::impl_object_ref_upcast!(ModulePass => Pass);
// tvm-ffi-stubgen(end)

impl ModulePass {
    /// Create a pass that applies `pass_func` to the module it runs on, which `pass_func` receives
    /// by value with the pass context (`transform.MakeModulePass`).
    ///
    /// The pass may run on any thread, so `pass_func` is `Send + Sync`.
    pub fn new(
        pass_func: impl Fn(IRModule, PassContext) -> Result<IRModule> + Send + Sync + 'static,
        pass_info: impl IntoArg<PassInfo>,
    ) -> Self {
        let pass_func = tvm_ffi::Function::from_typed(
            move |module: tvm_ffi::rvalue_ref::RValueRef<IRModule>, ctx: PassContext| {
                pass_func(module.into_inner(), ctx)
            },
        );
        support::infallible(
            tvm_ffi::cached_global_func!("transform.MakeModulePass")
                .call_tuple((&pass_func, pass_info.into_arg().borrow()))
                .and_then(|any| any.try_into()),
        )
    }
}

/// A pass that runs a list of passes in order, skipping the ones the current [`PassContext`] does
/// not enable.
// tvm-ffi-stubgen(begin): object/transform.Sequential
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Sequential {
    data_: ObjectArc<SequentialObj>,
}

/// The object behind [`Sequential`].
///
/// Opaque: parent 'transform.Pass' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "transform.Sequential", final, immutable)]
pub struct SequentialObj {
    base_: PassObj,
}

impl Deref for SequentialObj {
    type Target = PassObj;
    fn deref(&self) -> &PassObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl SequentialObj {
    pub fn pass_info(&self) -> Result<PassInfo> {
        cached_field_getter!(SequentialObj, "pass_info").get(self)
    }

    pub fn passes(&self) -> Result<Array<Pass>> {
        cached_field_getter!(SequentialObj, "passes").get(self)
    }
}

tvm_ffi::impl_object_ref_upcast!(Sequential => Pass);
// tvm-ffi-stubgen(end)

impl Sequential {
    /// Create a pass named by `pass_info` that runs `passes` in order; a pipeline with only a name
    /// takes `PassInfo::new(0, name)`.
    ///
    /// It skips the passes the current pass context disables, and those whose level exceeds the
    /// context's `opt_level` unless the context requires them.
    pub fn new(
        passes: impl IntoIterator<Item = impl IntoArg<Pass>>,
        pass_info: impl IntoArg<PassInfo>,
    ) -> Self {
        let passes = passes.into_iter().map(IntoArg::into_owned).collect::<Array<Pass>>();
        let pass_info = pass_info.into_arg();
        let pass_info: &PassInfo = pass_info.borrow();
        support::infallible(
            tvm_ffi::cached_global_func!("transform.Sequential")
                .call_tuple((&passes, pass_info.opt_level, &pass_info.name))
                .and_then(|any| any.try_into()),
        )
    }
}

/// The settings passes run under: the optimization level `opt_level`, the passes in
/// `required_pass` and `disabled_pass`, the configuration options `config` and the
/// `instruments`.
///
/// [`enter`](Self::enter) makes it the current context of the thread.
// tvm-ffi-stubgen(begin): object/transform.PassContext
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PassContext {
    data_: ObjectArc<PassContextObj>,
}

/// The object behind [`PassContext`].
///
/// Complete: reflected fields fill [24, 64) exactly (alignment padding [28, 32)).
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "transform.PassContext", final)]
pub struct PassContextObj {
    base_: Object,
    pub opt_level: i32,
    pub required_pass: Array<String>,
    pub disabled_pass: Array<String>,
    pub config: Map<String, Any>,
    pub instruments: Array<PassInstrument>,
}

const _: () = {
    assert!(::core::mem::size_of::<PassContextObj>() == 64);
    assert!(::core::mem::align_of::<PassContextObj>() == 8);
};

#[allow(missing_docs)]
impl PassContextObj {
    pub(crate) fn new(
        opt_level: i32,
        required_pass: Array<String>,
        disabled_pass: Array<String>,
        config: Map<String, Any>,
        instruments: Array<PassInstrument>,
    ) -> Self {
        let base_ = Object::new();
        Self { base_, opt_level, required_pass, disabled_pass, config, instruments }
    }
}

#[allow(missing_docs)]
impl PassContext {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        opt_level: i32,
        required_pass: impl Into<Array<String>>,
        disabled_pass: impl Into<Array<String>>,
        config: impl Into<Map<String, Any>>,
        instruments: impl Into<Array<PassInstrument>>,
    ) -> Self {
        let obj = PassContextObj::new(
            opt_level,
            required_pass.into(),
            disabled_pass.into(),
            config.into(),
            instruments.into(),
        );
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl PassContext {
    /// Create a pass context with `config` checked and converted to the registered option types;
    /// `None` is the empty configuration (`transform.PassContext`).
    ///
    /// A pass in `disabled_pass`, or above `opt_level` and not in `required_pass`, is skipped.
    ///
    /// # Errors
    ///
    /// Returns an `AttributeError` if `config` has an option that is not registered or a `None`
    /// value, or the error of converting a value to its option type.
    pub fn new(
        opt_level: i32,
        required_pass: impl IntoIterator<Item = impl AsRef<str>>,
        disabled_pass: impl IntoIterator<Item = impl AsRef<str>>,
        instruments: impl IntoIterator<Item = impl IntoArg<PassInstrument>>,
        config: impl Into<Option<Map<String, Any>>>,
    ) -> Result<Self> {
        let required_pass =
            required_pass.into_iter().map(|name| String::from(name.as_ref())).collect::<Array<_>>();
        let disabled_pass =
            disabled_pass.into_iter().map(|name| String::from(name.as_ref())).collect::<Array<_>>();
        let instruments =
            instruments.into_iter().map(IntoArg::into_owned).collect::<Array<PassInstrument>>();
        tvm_ffi::cached_global_func!("transform.PassContext")
            .call_tuple((opt_level, &required_pass, &disabled_pass, &instruments, config.into()))?
            .try_into()
    }

    /// Return the innermost pass context in effect on this thread, or the default one when there is
    /// none (`transform.GetCurrentPassContext`).
    pub fn current() -> Result<Self> {
        tvm_ffi::cached_global_func!("transform.GetCurrentPassContext").call_tuple(())?.try_into()
    }

    /// Make this the current pass context until the returned [`PassContextScope`] is dropped
    /// (`transform.EnterPassContext`, `transform.ExitPassContext`).
    ///
    /// # Errors
    ///
    /// Returns the error of an instrument's `enter_pass_ctx` hook; the instruments entered before
    /// it are exited, and the context's instruments are cleared.
    pub fn enter(&self) -> Result<PassContextScope> {
        tvm_ffi::cached_global_func!("transform.EnterPassContext").call_tuple((self,))?;
        Ok(PassContextScope(self.clone(), crate::ThreadBound::default()))
    }

    /// Replace the instruments with `instruments`, exiting the old ones and entering the new ones
    /// (`transform.OverrideInstruments`).
    ///
    /// # Errors
    ///
    /// Returns the error of an old instrument's `exit_pass_ctx` hook or a new one's
    /// `enter_pass_ctx` hook, after which the context has no instruments.
    pub fn override_instruments(
        &self,
        instruments: impl IntoIterator<Item = impl IntoArg<PassInstrument>>,
    ) -> Result<()> {
        let instruments =
            instruments.into_iter().map(IntoArg::into_owned).collect::<Array<PassInstrument>>();
        tvm_ffi::cached_global_func!("transform.OverrideInstruments")
            .call_tuple((self, &instruments))?;
        Ok(())
    }

    /// Return the registered configuration options, each with its metadata such as its `type`
    /// (`transform.ListConfigs`).
    pub fn list_configs() -> Result<Map<String, Map<String, String>>> {
        tvm_ffi::cached_global_func!("transform.ListConfigs").call_tuple(())?.try_into()
    }
}

/// A scope in which a pass context is the current one, returned by [`PassContext::enter`].
///
/// The entered contexts are a stack of the thread: a scope stays on its thread, and the innermost
/// scope is dropped first.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::transform::{PassContext, PassInstrument};
///
/// let outer = PassContext::new(3, [""; 0], [""; 0], Vec::<PassInstrument>::new(), None)?;
/// let inner = PassContext::new(1, [""; 0], [""; 0], Vec::<PassInstrument>::new(), None)?;
/// let outer_scope = outer.enter()?;
/// let inner_scope = inner.enter()?;
/// assert_eq!(PassContext::current()?.opt_level, 1);
///
/// drop(inner_scope);
/// assert_eq!(PassContext::current()?.opt_level, 3);
/// drop(outer_scope);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
///
/// A scope cannot move to another thread:
///
/// ```compile_fail
/// fn send<T: Send>() {}
/// send::<tvm_tirx::ir::transform::PassContextScope>();
/// ```
#[must_use = "the pass context is exited when the scope is dropped"]
pub struct PassContextScope(PassContext, crate::ThreadBound);

impl Drop for PassContextScope {
    fn drop(&mut self) {
        // A destructor cannot report the error, and exiting the innermost context does not fail.
        let _ = tvm_ffi::cached_global_func!("transform.ExitPassContext").call_tuple((&self.0,));
    }
}

/// The base of instruments, which a [`PassContext`] calls around each pass it runs, such as to
/// trace or time them.
// tvm-ffi-stubgen(begin): object/transform.PassInstrument
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PassInstrument {
    data_: ObjectArc<PassInstrumentObj>,
}

/// The object behind [`PassInstrument`].
///
/// Opaque: bytes [40, 48) of [24, 48) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "transform.PassInstrument", immutable)]
pub struct PassInstrumentObj {
    base_: Object,
}

#[allow(missing_docs)]
impl PassInstrumentObj {
    pub fn name(&self) -> Result<String> {
        cached_field_getter!(PassInstrumentObj, "name").get(self)
    }
}
// tvm-ffi-stubgen(end)

/// Create a pass that logs `header` and the module, which it returns unchanged
/// (`transform.PrintIR`).
pub fn print_ir(header: &str) -> Result<Pass> {
    tvm_ffi::cached_global_func!("transform.PrintIR")
        .call_tuple((String::from(header),))?
        .try_into()
}
