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
// The generated constructors take one argument per reflected field.
#![allow(clippy::too_many_arguments)]

//! The attributes of tile instruction calls.
//!
//! [`ScopeAttrs`] holds the thread scope of an instruction and [`TrnAttrs`] the scope and options
//! of a Trainium (`trn`) instruction; their type keys are `tirx.tensor.*`.

// tvm-ffi-stubgen(missing-docs): allow
// Every object registered under `tirx.tensor` has its block in this file.
// tvm-ffi-stubgen(prefix): tirx.tensor

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::expr::Call
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::expr::TensorRegion
// tvm-ffi-stubgen(import-object): crate::ir::op::Op
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): std::borrow::Borrow

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::attrs::AttrsObj;
use crate::ir::expr::Call;
use crate::ir::expr::Expr;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::TensorRegion;
use crate::ir::op::Op;
use crate::tirx::thread_scope::ThreadScope;
use std::borrow::Borrow;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
use tvm_ffi::Optional;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.tensor.ScopeAttrs
/// Complete: reflected fields fill [24, 32) exactly.
#[allow(missing_docs)]
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

#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ScopeAttrs {
    base: ObjectArc<ScopeAttrsObj>,
}

impl Deref for ScopeAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

#[allow(missing_docs)]
impl ScopeAttrsObj {
    pub(crate) fn new(scope: ThreadScope) -> Self {
        let base = AttrsObj::new();
        Self { base, scope }
    }
}

#[allow(missing_docs)]
impl ScopeAttrs {
    /// Lossless complete-field allocation.
    pub fn new(scope: impl Into<ThreadScope>) -> Self {
        let obj = ScopeAttrsObj::new(scope.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ScopeAttrsObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(ScopeAttrs => Attrs);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.tensor.TrnAttrs
/// Complete: reflected fields fill [24, 128) exactly (alignment padding [122, 128)).
#[allow(missing_docs)]
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

#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TrnAttrs {
    base: ObjectArc<TrnAttrsObj>,
}

impl Deref for TrnAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

#[allow(missing_docs)]
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

#[allow(missing_docs)]
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

unsafe impl tvm_ffi::object::ImmutableObject for TrnAttrsObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(TrnAttrs => Attrs);
// tvm-ffi-stubgen(end)

/// Return the tensor region `value` covers: a `TensorLoad` converted, a [`TensorRegion`] as is
/// (`tirx.AsTensorRegion`).
///
/// # Errors
///
/// Returns an error if `value` is neither.
pub fn as_tensor_region(value: impl IntoExpr) -> Result<TensorRegion> {
    tvm_ffi::cached_global_func!("tirx.AsTensorRegion")
        .call_tuple((value.into_expr()?.borrow(),))?
        .try_into()
}

/// Validate the calls of the tile instruction `op` with `validator`, after the checks of its
/// operand count, void type and registered attributes (`tirx.ConfigureTensorInstruction`).
///
/// An operator takes one custom validator, which reports an invalid call as an error.
///
/// # Errors
///
/// Returns an error if `op` already has a custom validator.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{Attrs, Call, OpDef, PrimType, Type, UNKNOWN_LOC};
/// use tvm_tirx::tirx::tile::{configure_tensor_instruction, ScopeAttrs, ScopeAttrsObj};
/// use tvm_tirx::tirx::ThreadScope;
/// use tvm_tirx::tvm_ffi::{ensure, ObjectRefCore, Result, VALUE_ERROR};
///
/// let op = OpDef::new("my_ext.tile_fill", "Fill a tile.")
///     .add_arg("value", "The fill value.")
///     .attrs_type_key("tirx.tensor.ScopeAttrs")
///     .op()?;
/// configure_tensor_instruction(&op, |call| {
///     let attrs = call.attrs.as_ref().and_then(|attrs| attrs.downcast_ref::<ScopeAttrsObj>());
///     let at_warp = attrs.unwrap().scope.same_as(&ThreadScope::get("cuda.warp")?);
///     ensure!(at_warp, VALUE_ERROR, "my_ext.tile_fill runs per warp");
///     Ok(())
/// })?;
///
/// # let call_at = |scope| -> Result<Call> {
/// #     let attrs = Attrs::from(ScopeAttrs::new(ThreadScope::get(scope)?));
/// #     let void = Type::from(PrimType::void());
/// #     Call::new(Some(&void), &op, [0], Some(&attrs), Vec::<Type>::new(), UNKNOWN_LOC)
/// # };
/// assert!(call_at("cuda.warp")?.validate().is_ok());
/// assert!(call_at("cuda.cta")?.validate().is_err());
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub fn configure_tensor_instruction(
    op: impl Into<Op>,
    validator: impl Fn(Call) -> Result<()> + Send + Sync + 'static,
) -> Result<()> {
    let validator = tvm_ffi::Function::from_typed(validator);
    tvm_ffi::cached_global_func!("tirx.ConfigureTensorInstruction")
        .call_tuple((&op.into(), &validator))?;
    Ok(())
}
