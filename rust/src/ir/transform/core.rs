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
#![allow(dead_code, unused_imports)]

//! Passes and the context they run in.
//!
//! The objects `python/tvm_tirx/ir/transform/core.py` defines. A [`Pass`] carries its
//! [`PassInfo`] (name and optimization level); [`ModulePass`] and [`Sequential`] are passes over a
//! module. A [`PassContext`] holds the optimization level, the required and disabled passes, the
//! configuration and the instruments of a run; [`PassContext::enter`] makes it current.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::module::IRModule
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::module::IRModule;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/transform.PassInfo
/// Complete: reflected fields fill [24, 48) exactly (alignment padding [28, 32)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "transform.PassInfo"]
#[type_final]
pub struct PassInfoObj {
    base: Object,
    pub opt_level: i32,
    pub name: String,
}

const _: () = {
    assert!(::core::mem::size_of::<PassInfoObj>() == 48);
    assert!(::core::mem::align_of::<PassInfoObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PassInfo {
    base: ObjectArc<PassInfoObj>,
}

impl Deref for PassInfo {
    type Target = PassInfoObj;
    fn deref(&self) -> &PassInfoObj {
        &self.base
    }
}

impl PassInfoObj {
    pub(crate) fn new(opt_level: i32, name: String) -> Self {
        let base = Object::new();
        Self { base, opt_level, name }
    }
}

impl PassInfo {
    /// Lossless complete-field allocation.
    pub fn new(opt_level: i32, name: impl Into<String>) -> Self {
        let obj = PassInfoObj::new(opt_level, name.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/transform.PassInstrument
/// Opaque: bytes [40, 48) of [24, 48) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "transform.PassInstrument"]
pub struct PassInstrumentObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PassInstrument {
    base: ObjectArc<PassInstrumentObj>,
}

impl Deref for PassInstrument {
    type Target = PassInstrumentObj;
    fn deref(&self) -> &PassInstrumentObj {
        &self.base
    }
}

impl PassInstrumentObj {
    pub fn name(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "name")?.get(self)
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/transform.PassContext
/// Complete: reflected fields fill [24, 64) exactly (alignment padding [28, 32)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "transform.PassContext"]
#[type_final]
pub struct PassContextObj {
    base: Object,
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

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PassContext {
    base: ObjectArc<PassContextObj>,
}

impl Deref for PassContext {
    type Target = PassContextObj;
    fn deref(&self) -> &PassContextObj {
        &self.base
    }
}

impl PassContextObj {
    pub(crate) fn new(
        opt_level: i32,
        required_pass: Array<String>,
        disabled_pass: Array<String>,
        config: Map<String, Any>,
        instruments: Array<PassInstrument>,
    ) -> Self {
        let base = Object::new();
        Self { base, opt_level, required_pass, disabled_pass, config, instruments }
    }
}

impl PassContext {
    /// Lossless complete-field allocation.
    pub fn new(
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
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/transform.Pass
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "transform.Pass"]
pub struct PassObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Pass {
    base: ObjectArc<PassObj>,
}

impl Deref for Pass {
    type Target = PassObj;
    fn deref(&self) -> &PassObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/transform.ModulePass
/// Opaque: parent 'transform.Pass' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "transform.ModulePass"]
#[type_final]
pub struct ModulePassObj {
    base: PassObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ModulePass {
    base: ObjectArc<ModulePassObj>,
}

impl Deref for ModulePass {
    type Target = ModulePassObj;
    fn deref(&self) -> &ModulePassObj {
        &self.base
    }
}

impl Deref for ModulePassObj {
    type Target = PassObj;
    fn deref(&self) -> &PassObj {
        &self.base
    }
}

impl ModulePassObj {
    pub fn pass_info(&self) -> Result<PassInfo> {
        FieldGetter::new(Self::type_index(), "pass_info")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(ModulePass => Pass);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/transform.Sequential
/// Opaque: parent 'transform.Pass' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "transform.Sequential"]
#[type_final]
pub struct SequentialObj {
    base: PassObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Sequential {
    base: ObjectArc<SequentialObj>,
}

impl Deref for Sequential {
    type Target = SequentialObj;
    fn deref(&self) -> &SequentialObj {
        &self.base
    }
}

impl Deref for SequentialObj {
    type Target = PassObj;
    fn deref(&self) -> &PassObj {
        &self.base
    }
}

impl SequentialObj {
    pub fn pass_info(&self) -> Result<PassInfo> {
        FieldGetter::new(Self::type_index(), "pass_info")?.get(self)
    }

    pub fn passes(&self) -> Result<Array<Pass>> {
        FieldGetter::new(Self::type_index(), "passes")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(Sequential => Pass);
// tvm-ffi-stubgen(end)

// `PassInfo::new` above has the semantics of the native `transform.PassInfo`, which only stores
// its fields.

/// A pass context in effect while it is alive: [`PassContext::enter`].
///
/// The entered pass contexts are a stack of the thread, so a scope stays on the thread that
/// opened it, and the scopes end in the reverse order they were opened: drop the innermost one
/// first.
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

impl PassContext {
    /// A pass context with its `config` checked and converted to the registered option types;
    /// `None` is the empty configuration (`transform.PassContext`).
    pub fn create(
        opt_level: i32,
        required_pass: impl IntoIterator<Item = impl AsRef<str>>,
        disabled_pass: impl IntoIterator<Item = impl AsRef<str>>,
        instruments: impl IntoIterator<Item = impl Into<PassInstrument>>,
        config: Option<&Map<String, Any>>,
    ) -> Result<Self> {
        let required_pass =
            required_pass.into_iter().map(|name| String::from(name.as_ref())).collect::<Array<_>>();
        let disabled_pass =
            disabled_pass.into_iter().map(|name| String::from(name.as_ref())).collect::<Array<_>>();
        let instruments =
            instruments.into_iter().map(Into::into).collect::<Array<PassInstrument>>();
        tvm_ffi::cached_global_func!("transform.PassContext")
            .call_tuple((opt_level, &required_pass, &disabled_pass, &instruments, config.cloned()))?
            .try_into()
    }

    /// The innermost pass context in effect, or the default one
    /// (`transform.GetCurrentPassContext`).
    pub fn current() -> Result<Self> {
        tvm_ffi::cached_global_func!("transform.GetCurrentPassContext").call_tuple(())?.try_into()
    }

    /// Make this the current pass context until the returned scope is dropped
    /// (`transform.EnterPassContext`, `transform.ExitPassContext`).
    pub fn enter(&self) -> Result<PassContextScope> {
        tvm_ffi::cached_global_func!("transform.EnterPassContext").call_tuple((self,))?;
        Ok(PassContextScope(self.clone(), crate::ThreadBound::default()))
    }

    /// Replace the instruments, exiting the old ones and entering the new ones
    /// (`transform.OverrideInstruments`).
    pub fn override_instruments(
        &self,
        instruments: impl IntoIterator<Item = impl Into<PassInstrument>>,
    ) -> Result<()> {
        let instruments =
            instruments.into_iter().map(Into::into).collect::<Array<PassInstrument>>();
        tvm_ffi::cached_global_func!("transform.OverrideInstruments")
            .call_tuple((self, &instruments))?;
        Ok(())
    }

    /// The registered configuration options, each with its metadata such as its `type`
    /// (`transform.ListConfigs`).
    pub fn list_configs() -> Result<Map<String, Map<String, String>>> {
        tvm_ffi::cached_global_func!("transform.ListConfigs").call_tuple(())?.try_into()
    }
}

impl Pass {
    /// The name and optimization level of the pass (`transform.Info`).
    pub fn info(&self) -> Result<PassInfo> {
        tvm_ffi::cached_global_func!("transform.Info").call_tuple((self,))?.try_into()
    }

    /// Run the pass on `module`, which it consumes, under the current pass context
    /// (`transform.RunPass`).
    pub fn run(&self, module: impl Into<IRModule>) -> Result<IRModule> {
        let module = tvm_ffi::RValueRef::new(module.into());
        tvm_ffi::cached_global_func!("transform.RunPass")
            .call_packed(&[tvm_ffi::AnyView::from(self), tvm_ffi::AnyView::from(&module)])?
            .try_into()
    }
}

impl ModulePass {
    /// The pass that applies `pass_func` to the module it runs on, which `pass_func` receives
    /// by value (`transform.MakeModulePass`).
    ///
    /// The pass may run on any thread, so `pass_func` is `Send + Sync`.
    pub fn new(
        pass_func: impl Fn(IRModule, PassContext) -> Result<IRModule> + Send + Sync + 'static,
        pass_info: impl Into<PassInfo>,
    ) -> Self {
        let pass_func = tvm_ffi::Function::from_typed(
            move |module: tvm_ffi::RValueRef<IRModule>, ctx: PassContext| {
                pass_func(module.into_inner(), ctx)
            },
        );
        support::infallible(
            tvm_ffi::cached_global_func!("transform.MakeModulePass")
                .call_tuple((&pass_func, &pass_info.into()))
                .and_then(|any| any.try_into()),
        )
    }
}

impl Sequential {
    /// The pass that runs `passes` in order, skipping those the current pass context disables or
    /// whose level exceeds its `opt_level` (`transform.Sequential`).
    pub fn new(
        passes: impl IntoIterator<Item = impl Into<Pass>>,
        opt_level: i32,
        name: &str,
    ) -> Self {
        let passes = passes.into_iter().map(Into::into).collect::<Array<Pass>>();
        support::infallible(
            tvm_ffi::cached_global_func!("transform.Sequential")
                .call_tuple((&passes, opt_level, String::from(name)))
                .and_then(|any| any.try_into()),
        )
    }
}

/// The pass that logs `header` and the module, which it returns unchanged (`transform.PrintIR`).
pub fn print_ir(header: &str) -> Result<Pass> {
    tvm_ffi::cached_global_func!("transform.PrintIR")
        .call_tuple((String::from(header),))?
        .try_into()
}
