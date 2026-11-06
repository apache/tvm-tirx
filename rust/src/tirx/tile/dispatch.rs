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

//! The context of tile instruction dispatch.
//!
//! A [`DispatchContext`] gives a lowering its target, thread scope, launch parameters and variable
//! ranges, and the state shared between dispatches.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{Evaluate, IntImm, PrimType, PrimVar, UNKNOWN_LOC};
//! use tvm_tirx::target::Target;
//! use tvm_tirx::tirx::tile::DispatchContext;
//! use tvm_tirx::tirx::ThreadScope;
//! use tvm_tirx::tvm_ffi::Map;
//!
//! let tx = PrimVar::new("tx", PrimType::int(32), UNKNOWN_LOC);
//! let ctx = DispatchContext::new(
//!     Target::new("cuda")?,
//!     ThreadScope::get("cuda.warp")?,
//!     [("threadIdx.x", tx, 32.into())],
//!     &Map::new(), false, &Map::new(), &Map::new(), &Map::new(), &Map::new(),
//!     "warp",
//! )?;
//!
//! ctx.cache_set("my_ext.zero", &IntImm::int32(0))?;
//! assert!(ctx.cache_get("my_ext.zero")?.is_some());
//! ctx.add_init_stmt(Evaluate::new(1, UNKNOWN_LOC), false)?;
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::stmt::Stmt
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimVar
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
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
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::object::ObjectRef;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.DispatchContext
/// Opaque: field 'launch_params' (Map[str, tuple[ir.Var, ir.Expr]]) has no native mirror. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.DispatchContext"]
#[type_final]
pub struct DispatchContextObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DispatchContext {
    base: ObjectArc<DispatchContextObj>,
}

impl Deref for DispatchContext {
    type Target = DispatchContextObj;
    fn deref(&self) -> &DispatchContextObj {
        &self.base
    }
}

impl DispatchContextObj {
    pub fn target(&self) -> Result<Target> {
        FieldGetter::new(Self::type_index(), "target")?.get(self)
    }

    pub fn thread_scope(&self) -> Result<ThreadScope> {
        FieldGetter::new(Self::type_index(), "thread_scope")?.get(self)
    }

    pub fn launch_params(&self) -> Result<Any> {
        FieldGetter::new(Self::type_index(), "launch_params")?.get_any(self)
    }

    pub fn var_range_map(&self) -> Result<Map<Var, Range>> {
        FieldGetter::new(Self::type_index(), "var_range_map")?.get(self)
    }

    pub fn alloc_only(&self) -> Result<bool> {
        FieldGetter::new(Self::type_index(), "alloc_only")?.get(self)
    }

    pub fn callbacks(&self) -> Result<Map<String, ObjectRef>> {
        FieldGetter::new(Self::type_index(), "callbacks")?.get(self)
    }

    pub fn shared_state(&self) -> Result<Map<String, ObjectRef>> {
        FieldGetter::new(Self::type_index(), "shared_state")?.get(self)
    }

    pub fn inter(&self) -> Result<Map<String, Array<PrimExpr>>> {
        FieldGetter::new(Self::type_index(), "inter")?.get(self)
    }

    pub fn intra(&self) -> Result<Map<String, Array<PrimExpr>>> {
        FieldGetter::new(Self::type_index(), "intra")?.get(self)
    }

    pub fn scope_kind(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "scope_kind")?.get(self)
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
        target: impl Into<Target>,
        thread_scope: impl Into<ThreadScope>,
        launch_params: impl IntoIterator<Item = (impl AsRef<str>, PrimVar, PrimExpr)>,
        var_range_map: &Map<Var, Range>,
        alloc_only: bool,
        callbacks: &Map<String, ObjectRef>,
        shared_state: &Map<String, ObjectRef>,
        inter: &Map<String, Array<PrimExpr>>,
        intra: &Map<String, Array<PrimExpr>>,
        scope_kind: &str,
    ) -> Result<Self> {
        let (target, thread_scope) = (target.into(), thread_scope.into());
        let launch_params = launch_params
            .into_iter()
            .map(|(name, var, extent)| {
                (String::from(name), Array::from_vec(vec![Any::from(var), Any::from(extent)]))
            })
            .collect::<Map<String, Array<Any>>>();
        let scope_kind = String::from(scope_kind);
        tvm_ffi::cached_global_func!("tirx.DispatchContext")
            .call_packed(&[
                AnyView::from(&target),
                AnyView::from(&thread_scope),
                AnyView::from(&launch_params),
                AnyView::from(var_range_map),
                AnyView::from(&alloc_only),
                AnyView::from(callbacks),
                AnyView::from(shared_state),
                AnyView::from(inter),
                AnyView::from(intra),
                AnyView::from(&scope_kind),
            ])?
            .try_into()
    }

    /// Request the allocation of the tensor variable `tensor`, placed at `allocated_addr` and
    /// annotated with `annotations`; only while `alloc_only`
    /// (`tirx.DispatchContextAddAllocTensor`).
    pub fn add_alloc_tensor(
        &self,
        tensor: impl Into<Var>,
        allocated_addr: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        annotations: &Map<String, Any>,
    ) -> Result<()> {
        let tensor = tensor.into();
        let allocated_addr = allocated_addr
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddAllocTensor").call_tuple((
            self,
            &tensor,
            &allocated_addr,
            annotations,
        ))?;
        Ok(())
    }

    /// Request `stmt` at the start of the host code if `host`, else of the kernel
    /// (`tirx.DispatchContextAddInitStmt`).
    pub fn add_init_stmt(&self, stmt: impl Into<Stmt>, host: bool) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddInitStmt").call_tuple((
            self,
            &stmt.into(),
            host,
        ))?;
        Ok(())
    }

    /// Request `stmt` right after the definition of the tensor variable `tensor`
    /// (`tirx.DispatchContextAddPostTensorDefStmt`).
    pub fn add_post_tensor_def_stmt(
        &self,
        tensor: impl Into<Var>,
        stmt: impl Into<Stmt>,
    ) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddPostTensorDefStmt").call_tuple((
            self,
            &tensor.into(),
            &stmt.into(),
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
    pub fn cache_set(
        &self,
        key: &str,
        value: &(impl ObjectRefCore + tvm_ffi::type_traits::AnyCompatible),
    ) -> Result<()> {
        let key = String::from(key);
        tvm_ffi::cached_global_func!("tirx.DispatchContextSharedStateSet").call_packed(&[
            AnyView::from(self),
            AnyView::from(&key),
            AnyView::from(value),
        ])?;
        Ok(())
    }
}
