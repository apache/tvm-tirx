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

//! The attributes of tile instruction calls.
//!
//! The objects `python/tvm_tirx/tirx/tile/instruction.py` registers. Their type keys are
//! `tirx.tensor.*`. [`ScopeAttrs`] holds the thread scope of an instruction, and [`TrnAttrs`] the
//! scope and options of a Trainium (`trn`) instruction. [`configure_tensor_instruction`] installs
//! the validator of a tile instruction operator.

// Every object registered under `tirx.tensor` has its block in this file.
// tvm-ffi-stubgen(prefix): tirx.tensor

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::expr::Call
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::expr::TensorRegion
// tvm-ffi-stubgen(import-object): crate::ir::op::Op
// tvm-ffi-stubgen(import-object): tvm_ffi::Result

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::attrs::AttrsObj;
use crate::ir::expr::Call;
use crate::ir::expr::Expr;
use crate::ir::expr::TensorRegion;
use crate::ir::op::Op;
use crate::tirx::thread_scope::ThreadScope;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
use tvm_ffi::Optional;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.tensor.ScopeAttrs
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.tensor.ScopeAttrs"]
#[type_final]
pub struct ScopeAttrsObj {
    base: AttrsObj,
    pub scope: ThreadScope,
}

const _: () = {
    assert!(::core::mem::size_of::<ScopeAttrsObj>() == 32);
    assert!(::core::mem::align_of::<ScopeAttrsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ScopeAttrs {
    base: ObjectArc<ScopeAttrsObj>,
}

impl Deref for ScopeAttrs {
    type Target = ScopeAttrsObj;
    fn deref(&self) -> &ScopeAttrsObj {
        &self.base
    }
}

impl Deref for ScopeAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl ScopeAttrsObj {
    pub(crate) fn new(scope: ThreadScope) -> Self {
        let base = AttrsObj::new();
        Self { base, scope }
    }
}

impl ScopeAttrs {
    /// Lossless complete-field allocation.
    pub fn new(scope: impl Into<ThreadScope>) -> Self {
        let obj = ScopeAttrsObj::new(scope.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(ScopeAttrs => Attrs);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.tensor.TrnAttrs
/// Complete: reflected fields fill [24, 128) exactly (alignment padding [122, 128)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.tensor.TrnAttrs"]
#[type_final]
pub struct TrnAttrsObj {
    base: AttrsObj,
    pub scope: ThreadScope,
    pub max_inst_size: Optional<i64>,
    pub opcode: String,
    pub reduce_op: String,
    pub op0: String,
    pub op1: String,
    pub axes: Array<i64>,
    pub negate: bool,
    pub reverse1: bool,
}

const _: () = {
    assert!(::core::mem::size_of::<TrnAttrsObj>() == 128);
    assert!(::core::mem::align_of::<TrnAttrsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TrnAttrs {
    base: ObjectArc<TrnAttrsObj>,
}

impl Deref for TrnAttrs {
    type Target = TrnAttrsObj;
    fn deref(&self) -> &TrnAttrsObj {
        &self.base
    }
}

impl Deref for TrnAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl TrnAttrsObj {
    pub(crate) fn new(
        scope: ThreadScope,
        max_inst_size: Optional<i64>,
        opcode: String,
        reduce_op: String,
        op0: String,
        op1: String,
        axes: Array<i64>,
        negate: bool,
        reverse1: bool,
    ) -> Self {
        let base = AttrsObj::new();
        Self { base, scope, max_inst_size, opcode, reduce_op, op0, op1, axes, negate, reverse1 }
    }
}

impl TrnAttrs {
    /// Lossless complete-field allocation.
    pub fn new(
        scope: impl Into<ThreadScope>,
        max_inst_size: impl Into<Optional<i64>>,
        opcode: impl Into<String>,
        reduce_op: impl Into<String>,
        op0: impl Into<String>,
        op1: impl Into<String>,
        axes: impl Into<Array<i64>>,
        negate: bool,
        reverse1: bool,
    ) -> Self {
        let obj = TrnAttrsObj::new(
            scope.into(),
            max_inst_size.into(),
            opcode.into(),
            reduce_op.into(),
            op0.into(),
            op1.into(),
            axes.into(),
            negate,
            reverse1,
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TrnAttrs => Attrs);
// tvm-ffi-stubgen(end)

/// The tensor region `value` covers: a `TensorLoad` converted, a `TensorRegion` as is
/// (`tirx.AsTensorRegion`).
pub fn as_tensor_region(value: impl Into<Expr>) -> Result<TensorRegion> {
    tvm_ffi::cached_global_func!("tirx.AsTensorRegion").call_tuple((&value.into(),))?.try_into()
}

/// Validate the calls of the tile instruction `op` with `validator`, after the checks of its
/// operand count, void type and registered attributes (`tirx.ConfigureTensorInstruction`).
///
/// An operator takes a custom validator once; `validator` reports an invalid call as an error.
/// The operator registry keeps it for the process and calls it from any thread, so it is
/// `Send + Sync`.
pub fn configure_tensor_instruction(
    op: impl Into<Op>,
    validator: impl Fn(Call) -> Result<()> + Send + Sync + 'static,
) -> Result<()> {
    let validator = tvm_ffi::Function::from_typed(validator);
    tvm_ffi::cached_global_func!("tirx.ConfigureTensorInstruction")
        .call_tuple((&op.into(), &validator))?;
    Ok(())
}
