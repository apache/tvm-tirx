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

//! Tile instruction dispatch. A lowering dispatches each tile instruction call with a
//! [`DispatchContext`], which describes where the call runs and collects the statements and
//! allocations the dispatch requests.

// tvm-ffi-stubgen(missing-docs): allow
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::stmt::Stmt
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimVar
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(import-object): tvm_ffi::IntoArg
// tvm-ffi-stubgen(import-object): std::borrow::Borrow
// The axis extents and offsets are `PrimExpr`s, as C++ `DispatchContextObj` holds them.
// tvm-ffi-stubgen(field): tirx.DispatchContext.inter -> Map<String, Array<PrimExpr>>
// tvm-ffi-stubgen(field): tirx.DispatchContext.intra -> Map<String, Array<PrimExpr>>

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::PrimVar;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use crate::ir::stmt::Stmt;
use crate::target::Target;
use crate::tirx::thread_scope::ThreadScope;
use std::borrow::Borrow;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::IntoArg;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::cached_field_getter;
use tvm_ffi::object::ObjectRef;
// tvm-ffi-stubgen(end)

/// The context a tile instruction call is lowered in: the `target`, the `thread_scope` of the
/// call, the kernel's `launch_params`, the ranges of the loop variables in scope
/// (`var_range_map`), the axis views across and within teams (`inter`, `intra`) and the
/// `scope_kind`.
///
/// Through it a dispatch requests allocations and statements, and caches objects that later
/// dispatches of the same lowering reuse. A context with `alloc_only` set is used only to allocate
/// tensors.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{Evaluate, IntImm, PrimType, PrimVar, UNKNOWN_LOC};
/// use tvm_tirx::target::Target;
/// use tvm_tirx::tirx::tile::DispatchContext;
/// use tvm_tirx::tirx::ThreadScope;
/// use tvm_tirx::tvm_ffi::{EnumRef, Map};
///
/// let tx = PrimVar::new("tx", PrimType::int(32), UNKNOWN_LOC);
/// let ctx = DispatchContext::new(
///     Target::new("cuda")?,
///     ThreadScope::get("cuda.warp")?,
///     [("threadIdx.x", tx, 32)],
///     Map::new(), false, Map::new(), Map::new(), Map::new(), Map::new(),
///     "warp",
/// )?;
///
/// ctx.cache_set("my_ext.zero", &IntImm::int32(0))?;
/// assert!(ctx.cache_get("my_ext.zero")?.is_some());
/// ctx.add_init_stmt(Evaluate::new(1, UNKNOWN_LOC), false)?;
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/tirx.DispatchContext
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DispatchContext {
    data_: ObjectArc<DispatchContextObj>,
}

/// The object behind [`DispatchContext`].
///
/// Opaque: field 'launch_params' (Map[str, tuple[ir.Var, ir.Expr]]) has no native mirror. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.DispatchContext", final, immutable)]
pub struct DispatchContextObj {
    base_: Object,
}

#[allow(missing_docs)]
impl DispatchContextObj {
    pub fn target(&self) -> Result<Target> {
        cached_field_getter!(DispatchContextObj, "target").get(self)
    }

    pub fn thread_scope(&self) -> Result<ThreadScope> {
        cached_field_getter!(DispatchContextObj, "thread_scope").get(self)
    }

    pub fn launch_params(&self) -> Result<Any> {
        cached_field_getter!(DispatchContextObj, "launch_params").get_any(self)
    }

    pub fn var_range_map(&self) -> Result<Map<Var, Range>> {
        cached_field_getter!(DispatchContextObj, "var_range_map").get(self)
    }

    pub fn alloc_only(&self) -> Result<bool> {
        cached_field_getter!(DispatchContextObj, "alloc_only").get(self)
    }

    pub fn callbacks(&self) -> Result<Map<String, ObjectRef>> {
        cached_field_getter!(DispatchContextObj, "callbacks").get(self)
    }

    pub fn shared_state(&self) -> Result<Map<String, ObjectRef>> {
        cached_field_getter!(DispatchContextObj, "shared_state").get(self)
    }

    pub fn inter(&self) -> Result<Map<String, Array<PrimExpr>>> {
        cached_field_getter!(DispatchContextObj, "inter").get(self)
    }

    pub fn intra(&self) -> Result<Map<String, Array<PrimExpr>>> {
        cached_field_getter!(DispatchContextObj, "intra").get(self)
    }

    pub fn scope_kind(&self) -> Result<String> {
        cached_field_getter!(DispatchContextObj, "scope_kind").get(self)
    }
}
// tvm-ffi-stubgen(end)

impl DispatchContext {
    /// Create the context of a dispatch for `target` at `thread_scope` (`tirx.DispatchContext`).
    ///
    /// `launch_params` maps each launch parameter to its variable and extent; `inter` and `intra`
    /// map each axis to its `[extent, offset]` across and within teams.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend of `thread_scope` is not the target's kind, or `scope_kind`
    /// is neither empty nor the local name of `thread_scope`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: impl IntoArg<Target>,
        thread_scope: impl IntoArg<ThreadScope>,
        launch_params: impl IntoIterator<
            Item = (impl AsRef<str>, impl IntoArg<PrimVar>, impl IntoExpr<PrimExpr>),
        >,
        var_range_map: impl Into<Map<Var, Range>>,
        alloc_only: bool,
        callbacks: impl Into<Map<String, ObjectRef>>,
        shared_state: impl Into<Map<String, ObjectRef>>,
        inter: impl Into<Map<String, Array<PrimExpr>>>,
        intra: impl Into<Map<String, Array<PrimExpr>>>,
        scope_kind: &str,
    ) -> Result<Self> {
        let launch_params = launch_params
            .into_iter()
            .map(|(name, var, extent)| {
                let (var, extent) = (var.into_owned(), extent.into_owned_expr()?);
                Ok((
                    String::from(name.as_ref()),
                    Array::from_vec(vec![Any::from(var), Any::from(extent)]),
                ))
            })
            .collect::<Result<Map<String, Array<Any>>>>()?;
        let scope_kind = String::from(scope_kind);
        tvm_ffi::cached_global_func!("tirx.DispatchContext")
            .call_packed(&[
                AnyView::from(target.into_arg().borrow()),
                AnyView::from(thread_scope.into_arg().borrow()),
                AnyView::from(&launch_params),
                AnyView::from(&var_range_map.into()),
                AnyView::from(&alloc_only),
                AnyView::from(&callbacks.into()),
                AnyView::from(&shared_state.into()),
                AnyView::from(&inter.into()),
                AnyView::from(&intra.into()),
                AnyView::from(&scope_kind),
            ])?
            .try_into()
    }

    /// Request the allocation of the tensor variable `tensor`, placed at `allocated_addr` and
    /// annotated with `annotations`; only while `alloc_only`
    /// (`tirx.DispatchContextAddAllocTensor`).
    pub fn add_alloc_tensor(
        &self,
        tensor: impl IntoArg<Var>,
        allocated_addr: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        annotations: impl Into<Map<String, Any>>,
    ) -> Result<()> {
        let allocated_addr = allocated_addr
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddAllocTensor").call_tuple((
            self,
            tensor.into_arg().borrow(),
            &allocated_addr,
            &annotations.into(),
        ))?;
        Ok(())
    }

    /// Request `stmt` at the start of the host code if `host`, else of the kernel
    /// (`tirx.DispatchContextAddInitStmt`).
    pub fn add_init_stmt(&self, stmt: impl IntoArg<Stmt>, host: bool) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddInitStmt").call_tuple((
            self,
            stmt.into_arg().borrow(),
            host,
        ))?;
        Ok(())
    }

    /// Request `stmt` right after the definition of the tensor variable `tensor`
    /// (`tirx.DispatchContextAddPostTensorDefStmt`).
    pub fn add_post_tensor_def_stmt(
        &self,
        tensor: impl IntoArg<Var>,
        stmt: impl IntoArg<Stmt>,
    ) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddPostTensorDefStmt").call_tuple((
            self,
            tensor.into_arg().borrow(),
            stmt.into_arg().borrow(),
        ))?;
        Ok(())
    }

    /// Return the object cached under `key` across dispatches, if any
    /// (`tirx.DispatchContextSharedStateGet`).
    pub fn cache_get(&self, key: &str) -> Result<Option<ObjectRef>> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextSharedStateGet")
            .call_tuple((self, String::from(key)))?
            .try_into()
    }

    /// Cache `value` under `key` across dispatches (`tirx.DispatchContextSharedStateSet`).
    pub fn cache_set(&self, key: &str, value: &impl ObjectRefCore) -> Result<()> {
        let key = String::from(key);
        tvm_ffi::cached_global_func!("tirx.DispatchContextSharedStateSet").call_packed(&[
            AnyView::from(self),
            AnyView::from(&key),
            AnyView::from(value),
        ])?;
        Ok(())
    }
}
