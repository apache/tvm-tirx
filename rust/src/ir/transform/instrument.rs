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

//! Pass instrumentation.
//!
//! A [`BasePassInstrument`] is a [`PassInstrument`] whose hooks are callbacks, such as Rust
//! closures ([`BasePassInstrument::new`]).
//!
//! # Examples
//!
//! ```
//! use std::sync::{Arc, Mutex};
//! use tvm_tirx::ir::transform::{BasePassInstrument, ModulePass, Pass, PassContext, PassInfo};
//! use tvm_tirx::ir::transform::PassInstrumentHooks;
//! use tvm_tirx::ir::IRModule;
//! use tvm_tirx::tvm_ffi::Map;
//!
//! let log = Arc::new(Mutex::new(Vec::new()));
//! let sink = log.clone();
//! let hooks = PassInstrumentHooks {
//!     run_before_pass: Some(Box::new(move |_module, info| {
//!         Ok(sink.lock().unwrap().push(info.name.as_str().to_owned()))
//!     })),
//!     ..Default::default()
//! };
//! let tracer = BasePassInstrument::new("tracer", hooks);
//!
//! let _scope = PassContext::create(2, [""; 0], [""; 0], [tracer], None)?.enter()?;
//! let noop = Pass::from(ModulePass::new(|module, _ctx| Ok(module), PassInfo::new(0, "noop")));
//! noop.run(IRModule::from_functions(&Map::new(), None, &Map::new())?)?;
//! assert_eq!(*log.lock().unwrap(), ["noop"]);
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::module::IRModule
// tvm-ffi-stubgen(import-object): crate::ir::transform::core::PassInfo
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::Function
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::module::IRModule;
use crate::ir::support;
use crate::ir::transform::core::PassInfo;
use crate::ir::transform::core::PassInstrument;
use crate::ir::transform::core::PassInstrumentObj;
use std::ops::Deref;
use tvm_ffi::AnyView;
use tvm_ffi::Function;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/transform.BasePassInstrument
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "transform.BasePassInstrument"]
pub struct BasePassInstrumentObj {
    base: PassInstrumentObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BasePassInstrument {
    base: ObjectArc<BasePassInstrumentObj>,
}

impl Deref for BasePassInstrument {
    type Target = BasePassInstrumentObj;
    fn deref(&self) -> &BasePassInstrumentObj {
        &self.base
    }
}

impl Deref for BasePassInstrumentObj {
    type Target = PassInstrumentObj;
    fn deref(&self) -> &PassInstrumentObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(BasePassInstrument => PassInstrument);
// tvm-ffi-stubgen(end)

/// A hook of a [`BasePassInstrument`], called when a pass context is entered or exited.
///
/// It may be called from any thread, so it is `Send + Sync`; an error fails the entry or exit.
pub type PassContextHook = Box<dyn Fn() -> Result<()> + Send + Sync>;

/// A hook of a [`BasePassInstrument`], called with a module and the information of the pass to run
/// on it; `Send + Sync` like a [`PassContextHook`].
///
/// An error fails the pass run.
pub type PassHook<R> = Box<dyn Fn(IRModule, PassInfo) -> Result<R> + Send + Sync>;

/// The hooks of a [`BasePassInstrument`].
///
/// A hook left `None` does nothing, so `..Default::default()` fills in the ones not needed.
#[derive(Default)]
pub struct PassInstrumentHooks {
    /// Called when a pass context with the instrument is entered.
    pub enter_pass_ctx: Option<PassContextHook>,
    /// Called when a pass context with the instrument is exited.
    pub exit_pass_ctx: Option<PassContextHook>,
    /// Called before a pass runs on the module, to decide whether to run it; a pass the context
    /// requires runs without asking.
    pub should_run: Option<PassHook<bool>>,
    /// Called before a pass runs on the module.
    pub run_before_pass: Option<PassHook<()>>,
    /// Called after a pass ran, with the module it returned.
    pub run_after_pass: Option<PassHook<()>>,
}

impl BasePassInstrument {
    /// Create the instrument `name` that calls `hooks` (`transform.PassInstrument`).
    pub fn new(name: &str, hooks: PassInstrumentHooks) -> Self {
        let PassInstrumentHooks {
            enter_pass_ctx,
            exit_pass_ctx,
            should_run,
            run_before_pass,
            run_after_pass,
        } = hooks;
        let enter_pass_ctx = enter_pass_ctx.map(Function::from_typed);
        let exit_pass_ctx = exit_pass_ctx.map(Function::from_typed);
        let should_run = should_run.map(Function::from_typed);
        let run_before_pass = run_before_pass.map(Function::from_typed);
        let run_after_pass = run_after_pass.map(Function::from_typed);
        support::infallible(
            tvm_ffi::cached_global_func!("transform.PassInstrument")
                .call_packed(&[
                    AnyView::from(&String::from(name)),
                    AnyView::from(&enter_pass_ctx),
                    AnyView::from(&exit_pass_ctx),
                    AnyView::from(&should_run),
                    AnyView::from(&run_before_pass),
                    AnyView::from(&run_after_pass),
                ])
                .and_then(|any| any.try_into()),
        )
    }
}

/// Create an instrument that times the passes, for [`render_pass_timing`]: Python's
/// `PassTimingInstrument()` (`transform.instrument.MakePassTimingInstrument`).
///
pub fn pass_timing_instrument() -> Result<BasePassInstrument> {
    tvm_ffi::cached_global_func!("transform.instrument.MakePassTimingInstrument")
        .call_tuple(())?
        .try_into()
}

/// Render the pass times that a [`pass_timing_instrument`] recorded, as Python's
/// `PassTimingInstrument.render()` does (`transform.instrument.RenderTimePassProfiles`).
///
/// The result has a line per pass timed on this thread, and is empty when none was.
///
/// # Errors
///
/// Returns an `InternalError` if a pass is still running, as when a pass calls it.
pub fn render_pass_timing() -> Result<String> {
    tvm_ffi::cached_global_func!("transform.instrument.RenderTimePassProfiles")
        .call_tuple(())?
        .try_into()
}
