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

//! Function passes.
//!
//! A [`FunctionPass`] is a [`Pass`] that transforms each TIRx function of a module.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::transform::{Pass, PassInfo};
//! use tvm_tirx::tirx::transform::FunctionPass;
//! # use tvm_tirx::ir::{BaseFunc, DictAttrs, Evaluate, GlobalVar, IRModule, MissingType, SeqStmt};
//! # use tvm_tirx::ir::{Var, UNKNOWN_LOC};
//! # use tvm_tirx::tirx::Function;
//! # use tvm_tirx::tvm_ffi::Map;
//! # let body = SeqStmt::new([Evaluate::new(1, UNKNOWN_LOC)], UNKNOWN_LOC);
//! # let attrs = DictAttrs::new(Map::new());
//! # let main = Function::new(Vec::<Var>::new(), &body, MissingType::new(), attrs, UNKNOWN_LOC);
//! # let functions = [(GlobalVar::new("main", UNKNOWN_LOC), BaseFunc::from(&main))];
//! # let module = IRModule::from_functions(&functions.into_iter().collect(), None, &Map::new())?;
//!
//! let pass = FunctionPass::new(
//!     |func, _module, _ctx| Ok((!func.params.is_empty()).then_some(func)),
//!     PassInfo::new(0, "my_ext.RemoveNullary"),
//! );
//! let module = Pass::from(&pass).run(module)?;
//! assert!(!module.contains("main")?);
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::module::IRModule
// tvm-ffi-stubgen(import-object): crate::ir::transform::core::PassContext
// tvm-ffi-stubgen(import-object): crate::tirx::function::Function
// tvm-ffi-stubgen(import-object): tvm_ffi::RValueRef
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::module::IRModule;
use crate::ir::support;
use crate::ir::transform::core::Pass;
use crate::ir::transform::core::PassContext;
use crate::ir::transform::core::PassInfo;
use crate::ir::transform::core::PassObj;
use crate::tirx::function::Function;
use std::ops::Deref;
use tvm_ffi::FieldGetter;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::RValueRef;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.FunctionPass
/// Opaque: parent 'transform.Pass' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.FunctionPass"]
#[type_final]
pub struct FunctionPassObj {
    base: PassObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FunctionPass {
    base: ObjectArc<FunctionPassObj>,
}

impl Deref for FunctionPass {
    type Target = FunctionPassObj;
    fn deref(&self) -> &FunctionPassObj {
        &self.base
    }
}

impl Deref for FunctionPassObj {
    type Target = PassObj;
    fn deref(&self) -> &PassObj {
        &self.base
    }
}

impl FunctionPassObj {
    pub fn pass_info(&self) -> Result<PassInfo> {
        FieldGetter::new(Self::type_index(), "pass_info")?.get(self)
    }
}

tvm_ffi::unsafe_impl_object_upcast!(FunctionPass => Pass);
// tvm-ffi-stubgen(end)

impl FunctionPass {
    /// Create the pass described by `pass_info` that replaces each TIRx function of a module by
    /// what `pass_func` returns for it, the module and the pass context, or removes it for `None`
    /// (`tirx.transform.CreateFunctionPass`).
    ///
    /// `pass_func` owns the function, so it can update it in place when it holds the only
    /// reference.
    pub fn new(
        pass_func: impl Fn(Function, IRModule, PassContext) -> Result<Option<Function>>
            + Send
            + Sync
            + 'static,
        pass_info: impl Into<PassInfo>,
    ) -> Self {
        let pass_func = tvm_ffi::Function::from_typed(
            move |func: RValueRef<Function>, module: IRModule, ctx: PassContext| {
                pass_func(func.into_inner(), module, ctx)
            },
        );
        support::infallible(
            tvm_ffi::cached_global_func!("tirx.transform.CreateFunctionPass")
                .call_tuple((&pass_func, &pass_info.into()))
                .and_then(|any| any.try_into()),
        )
    }
}
