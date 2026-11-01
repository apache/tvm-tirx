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

//! The TIRx passes and their configurations.
//!
//! The passes `python/tvm_tirx/tirx/transform/transform.py` defines, each a function that returns
//! the [`Pass`], and their configurations: [`RemoveNoOpConfig`] for [`remove_no_op`], and
//! [`UnrollLoopConfig`] for [`unroll_loop`].

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::transform::core::Pass
// tvm-ffi-stubgen(import-object): crate::target::Target
// tvm-ffi-stubgen(import-object): crate::tirx::function::Function
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String

// tvm-ffi-stubgen(begin): import-section
use crate::ir::transform::core::Pass;
use crate::target::Target;
use crate::tirx::function::Function;
use std::ops::Deref;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.transform.RemoveNoOpConfig
/// Complete: reflected fields fill [24, 40) exactly (alignment padding [33, 40)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.transform.RemoveNoOpConfig"]
#[type_final]
pub struct RemoveNoOpConfigObj {
    base: Object,
    pub max_simplification_steps: i64,
    pub ignore_profiler_call: bool,
}

const _: () = {
    assert!(::core::mem::size_of::<RemoveNoOpConfigObj>() == 40);
    assert!(::core::mem::align_of::<RemoveNoOpConfigObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RemoveNoOpConfig {
    base: ObjectArc<RemoveNoOpConfigObj>,
}

impl Deref for RemoveNoOpConfig {
    type Target = RemoveNoOpConfigObj;
    fn deref(&self) -> &RemoveNoOpConfigObj {
        &self.base
    }
}

impl RemoveNoOpConfigObj {
    pub(crate) fn new(max_simplification_steps: i64, ignore_profiler_call: bool) -> Self {
        let base = Object::new();
        Self { base, max_simplification_steps, ignore_profiler_call }
    }
}

impl RemoveNoOpConfig {
    /// Lossless complete-field allocation.
    pub fn new(max_simplification_steps: i64, ignore_profiler_call: bool) -> Self {
        let obj = RemoveNoOpConfigObj::new(max_simplification_steps, ignore_profiler_call);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.transform.UnrollLoopConfig
/// Complete: reflected fields fill [24, 48) exactly (alignment padding [44, 48)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.transform.UnrollLoopConfig"]
#[type_final]
pub struct UnrollLoopConfigObj {
    base: Object,
    pub auto_max_step: i32,
    pub auto_max_depth: i32,
    pub auto_max_extent: i32,
    pub explicit_unroll: i32,
    pub unroll_local_access: i32,
}

const _: () = {
    assert!(::core::mem::size_of::<UnrollLoopConfigObj>() == 48);
    assert!(::core::mem::align_of::<UnrollLoopConfigObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct UnrollLoopConfig {
    base: ObjectArc<UnrollLoopConfigObj>,
}

impl Deref for UnrollLoopConfig {
    type Target = UnrollLoopConfigObj;
    fn deref(&self) -> &UnrollLoopConfigObj {
        &self.base
    }
}

impl UnrollLoopConfigObj {
    pub(crate) fn new(
        auto_max_step: i32,
        auto_max_depth: i32,
        auto_max_extent: i32,
        explicit_unroll: i32,
        unroll_local_access: i32,
    ) -> Self {
        let base = Object::new();
        Self {
            base,
            auto_max_step,
            auto_max_depth,
            auto_max_extent,
            explicit_unroll,
            unroll_local_access,
        }
    }
}

impl UnrollLoopConfig {
    /// Lossless complete-field allocation.
    pub fn new(
        auto_max_step: i32,
        auto_max_depth: i32,
        auto_max_extent: i32,
        explicit_unroll: i32,
        unroll_local_access: i32,
    ) -> Self {
        let obj = UnrollLoopConfigObj::new(
            auto_max_step,
            auto_max_depth,
            auto_max_extent,
            explicit_unroll,
            unroll_local_access,
        );
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

/// Infer and mark an unambiguous TIRx entry function (`tirx.transform.AnnotateEntryFunc`).
pub fn annotate_entry_func() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.AnnotateEntryFunc").call_tuple(())?.try_into()
}

/// Promote unsupported BF16 computations and internal temporaries to float32
/// (`tirx.transform.BF16ComputeLegalize`).
pub fn bf16_compute_legalize() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.BF16ComputeLegalize").call_tuple(())?.try_into()
}

/// Represent BF16 storage as uint16, keeping its bit patterns
/// (`tirx.transform.BF16StorageLegalize`).
pub fn bf16_storage_legalize() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.BF16StorageLegalize").call_tuple(())?.try_into()
}

/// Bind `target` to the functions and classify host and device helpers (`tirx.transform.BindTarget`).
pub fn bind_target(target: impl Into<Target>) -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.BindTarget")
        .call_tuple((&target.into(),))?
        .try_into()
}

/// Replace redundant computations by new variables (`tirx.transform.CommonSubexprElim`).
pub fn common_subexpr_elim() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.CommonSubexprElim").call_tuple(())?.try_into()
}

/// Give every variable definition in a module a distinct identity (`tirx.transform.ConvertSSA`).
pub fn convert_ssa() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.ConvertSSA").call_tuple(())?.try_into()
}

/// Wrap each function body in a `tirx.device_scope` region (`tirx.transform.DecorateDeviceScope`).
pub fn decorate_device_scope() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.DecorateDeviceScope").call_tuple(())?.try_into()
}

/// Promote unsupported FP8 computations and internal temporaries to `promote_dtype`, `float16` or
/// `float32` (`tirx.transform.FP8ComputeLegalize`).
pub fn fp8_compute_legalize(promote_dtype: &str) -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.FP8ComputeLegalize")
        .call_tuple((String::from(promote_dtype),))?
        .try_into()
}

/// Represent FP8 storage as uint8, keeping its bit patterns (`tirx.transform.FP8StorageLegalize`).
pub fn fp8_storage_legalize() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.FP8StorageLegalize").call_tuple(())?.try_into()
}

/// Remove the TIRx functions for which `fcond` is false (`tirx.transform.Filter`).
///
/// The pass may run on any thread, so `fcond` is `Send + Sync`.
pub fn filter(fcond: impl Fn(Function) -> Result<bool> + Send + Sync + 'static) -> Result<Pass> {
    let fcond = tvm_ffi::Function::from_typed(fcond);
    tvm_ffi::cached_global_func!("tirx.transform.Filter").call_tuple((&fcond,))?.try_into()
}

/// Flatten tensor allocations and accesses to one-dimensional storage
/// (`tirx.transform.FlattenTensor`).
pub fn flatten_tensor() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.FlattenTensor").call_tuple(())?.try_into()
}

/// Force index expressions to int32 without range proofs
/// (`tirx.transform.ForceNarrowIndexToInt32`).
pub fn force_narrow_index_to_int32() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.ForceNarrowIndexToInt32")
        .call_tuple(())?
        .try_into()
}

/// Hoist loop-invariant `if` statements out of eligible loops (`tirx.transform.HoistIf`).
pub fn hoist_if() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.HoistIf").call_tuple(())?.try_into()
}

/// Lower intrinsic calls by the rules registered for the function target
/// (`tirx.transform.LowerIntrin`).
pub fn lower_intrin() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.LowerIntrin").call_tuple(())?.try_into()
}

/// Lower runtime builtins to explicit stack storage and ABI operations
/// (`tirx.transform.LowerTVMBuiltin`).
pub fn lower_tvm_builtin() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.LowerTVMBuiltin").call_tuple(())?.try_into()
}

/// Apply tensor layouts to physical storage and accesses (`tirx.transform.LowerTensorLayouts`).
pub fn lower_tensor_layouts() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.LowerTensorLayouts").call_tuple(())?.try_into()
}

/// Lower `gpu_thread_allreduce` to target-specific communication
/// (`tirx.transform.LowerThreadAllreduce`).
pub fn lower_thread_allreduce() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.LowerThreadAllreduce").call_tuple(())?.try_into()
}

/// Expose exported functions through the TVM FFI packed-call ABI (`tirx.transform.MakePackedAPI`).
pub fn make_packed_api() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.MakePackedAPI").call_tuple(())?.try_into()
}

/// Narrow integer indices and loop variables to `target_bits` where their bounds prove it safe
/// (`tirx.transform.NarrowDataType`).
pub fn narrow_data_type(target_bits: i32) -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.NarrowDataType")
        .call_tuple((target_bits,))?
        .try_into()
}

/// Rewrite pointer and tensor element types to match their accesses
/// (`tirx.transform.PointerValueTypeRewrite`).
pub fn pointer_value_type_rewrite() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.PointerValueTypeRewrite")
        .call_tuple(())?
        .try_into()
}

/// Remove effect-free statements and provably empty control flow (`tirx.transform.RemoveNoOp`).
pub fn remove_no_op() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.RemoveNoOp").call_tuple(())?.try_into()
}

/// Rewrite `floordiv(floormod())` split patterns to `floormod(floordiv())`
/// (`tirx.transform.RenormalizeSplitPattern`).
pub fn renormalize_split_pattern() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.RenormalizeSplitPattern")
        .call_tuple(())?
        .try_into()
}

/// Replace assertion statements with no-ops (`tirx.transform.SkipAssert`).
pub fn skip_assert() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.SkipAssert").call_tuple(())?.try_into()
}

/// Annotate, split and lower host and device functions (`tirx.transform.SplitHostDevice`).
pub fn split_host_device() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.SplitHostDevice").call_tuple(())?.try_into()
}

/// Run statement-level arithmetic simplifications (`tirx.transform.StmtSimplify`).
pub fn stmt_simplify() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.StmtSimplify").call_tuple(())?.try_into()
}

/// Plan storage by moving allocations and reusing compatible storage
/// (`tirx.transform.StorageRewrite`).
pub fn storage_rewrite() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.StorageRewrite").call_tuple(())?.try_into()
}

/// Lower TIRx tile calls with the dispatch table of the active backend
/// (`tirx.transform.TileDispatch`).
pub fn tile_dispatch() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.TileDispatch").call_tuple(())?.try_into()
}

/// Unroll loops with constant extents marked `T.unroll` (`tirx.transform.UnrollLoop`).
pub fn unroll_loop() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.UnrollLoop").call_tuple(())?.try_into()
}

/// Lower vectorized loops to vector operations, or to serial loops unless `enable_vectorize`
/// (`tirx.transform.VectorizeLoop`).
pub fn vectorize_loop(enable_vectorize: bool) -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.VectorizeLoop")
        .call_tuple((enable_vectorize,))?
        .try_into()
}

/// Reject illegal host-side access to device memory (`tirx.transform.VerifyMemory`).
pub fn verify_memory() -> Result<Pass> {
    tvm_ffi::cached_global_func!("tirx.transform.VerifyMemory").call_tuple(())?.try_into()
}
