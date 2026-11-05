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

//! The module: the unit of compilation.
//!
//! An [`IRModule`] maps global variables to functions, with the module attributes and module-level
//! information. Adding, updating or removing functions changes the module through every handle to
//! it; [`IRModule::shallow_copy`] makes one to change independently.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{BaseFunc, GlobalVar, IRModule, UNKNOWN_LOC};
//! use tvm_tirx::tvm_ffi::{Map, ObjectRefCore};
//! # use tvm_tirx::ir::{DictAttrs, PrimType, Return, SeqStmt, Var};
//! # use tvm_tirx::tirx::Function;
//! # let body = SeqStmt::new([Return::new(0, UNKNOWN_LOC)], UNKNOWN_LOC);
//! # let attrs = DictAttrs::new(Map::new());
//! # let func = Function::new(Vec::<Var>::new(), &body, PrimType::int(32), attrs, UNKNOWN_LOC);
//!
//! let main = GlobalVar::new("main", UNKNOWN_LOC);
//! let functions = Map::from_iter([(main.clone(), BaseFunc::from(func))]);
//! let module = IRModule::from_functions(&functions, None, &Map::new())?;
//!
//! assert!(module.lookup(&main)?.same_as(&module.lookup_name("main")?));
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::RValueRef
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): std::borrow::Borrow

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::Expr;
use crate::ir::expr::GlobalVar;
use crate::ir::expr::IntoExpr;
use crate::ir::function::BaseFunc;
use crate::ir::global_info::GlobalInfo;
use crate::ir::support;
use std::borrow::Borrow;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::RValueRef;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.IRModule
/// Complete: reflected fields fill [24, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.IRModule"]
#[type_final]
pub struct IRModuleObj {
    base: Object,
    pub functions: Map<GlobalVar, BaseFunc>,
    pub attrs: DictAttrs,
    pub global_infos: Map<String, Array<GlobalInfo>>,
    pub global_var_map: Map<String, GlobalVar>,
}

const _: () = {
    assert!(::core::mem::size_of::<IRModuleObj>() == 56);
    assert!(::core::mem::align_of::<IRModuleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IRModule {
    base: ObjectArc<IRModuleObj>,
}

impl Deref for IRModule {
    type Target = IRModuleObj;
    fn deref(&self) -> &IRModuleObj {
        &self.base
    }
}

impl IRModuleObj {
    pub(crate) fn new(
        functions: Map<GlobalVar, BaseFunc>,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
        global_var_map: Map<String, GlobalVar>,
    ) -> Self {
        let base = Object::new();
        Self { base, functions, attrs, global_infos, global_var_map }
    }
}

impl IRModule {
    /// Lossless complete-field allocation.
    pub fn new(
        functions: impl Into<Map<GlobalVar, BaseFunc>>,
        attrs: impl Into<DictAttrs>,
        global_infos: impl Into<Map<String, Array<GlobalInfo>>>,
        global_var_map: impl Into<Map<String, GlobalVar>>,
    ) -> Self {
        let obj = IRModuleObj::new(
            functions.into(),
            attrs.into(),
            global_infos.into(),
            global_var_map.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl IRModule {
    /// Create the module of `functions`, with the attributes `attrs` (empty if `None`) and the
    /// module-level information `global_infos` (C++ `IRModule(functions, attrs, global_infos)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if two global variables of `functions` have the same name.
    pub fn from_functions(
        functions: &Map<GlobalVar, BaseFunc>,
        attrs: Option<&DictAttrs>,
        global_infos: &Map<String, Array<GlobalInfo>>,
    ) -> Result<Self> {
        let attrs = match attrs {
            Some(attrs) => attrs.clone(),
            None => DictAttrs::new(Map::new()),
        };
        let mut names = std::collections::HashSet::new();
        let mut global_var_map = Vec::with_capacity(functions.len());
        for (var, _) in functions.iter() {
            let name = var.name_hint.clone();
            if !names.insert(name.as_str().to_string()) {
                let condition = "n->global_var_map_.count(kv.first->name_hint) == 0";
                let message = format!("Duplicate global function name {}", name.as_str());
                return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, &message));
            }
            global_var_map.push((name, var));
        }
        let global_var_map = global_var_map.into_iter().collect::<Map<String, GlobalVar>>();
        Ok(Self::new(functions.clone(), attrs, global_infos.clone(), global_var_map))
    }

    /// Create the module of `functions` and `expr`, a function bound to its `global_symbol`, or to
    /// `main` (made unique) when it has none (`ir.IRModuleFromExpr`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `expr` is not a function.
    pub fn from_expr(expr: impl IntoExpr, functions: &Map<GlobalVar, BaseFunc>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.IRModuleFromExpr")
            .call_tuple((expr.into_expr()?.borrow(), functions))?
            .try_into()
    }

    /// Return a copy of the module that shares its functions (`ir.IRModuleClone`).
    ///
    pub fn shallow_copy(&self) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.IRModuleClone").call_tuple((self,))?.try_into()
    }

    /// Add `func` as `var`, replacing an existing definition; the native side currently ignores
    /// `update` (`ir.IRModuleAdd`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if the module has another global variable of the same name.
    pub fn add(
        &self,
        var: impl Into<GlobalVar>,
        func: impl Into<BaseFunc>,
        update: bool,
    ) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleAdd").call_tuple((
            self,
            &var.into(),
            &func.into(),
            update,
        ))?;
        Ok(())
    }

    /// Remove the function named `name` (`ir.IRModuleRemove`).
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if the module has no global variable named `name`.
    pub fn remove(&self, name: &str) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleRemove").call_tuple((self, String::from(name)))?;
        Ok(())
    }

    /// Remove the function of `var`, if the module has one (`ir.IRModuleRemove`).
    pub fn remove_global_var(&self, var: impl Into<GlobalVar>) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleRemove").call_tuple((self, &var.into()))?;
        Ok(())
    }

    /// Return whether the module has a global variable named `name` (`ir.IRModuleContains`).
    pub fn contains(&self, name: &str) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.IRModuleContains")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// Return whether the module defines a function for `var` (`ir.IRModuleContains`).
    pub fn contains_global_var(&self, var: impl Into<GlobalVar>) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.IRModuleContains")
            .call_tuple((self, &var.into()))?
            .try_into()
    }

    /// Return whether the module has a global variable named `name`
    /// (`ir.IRModuleContainGlobalVar`).
    pub fn has_global_var(&self, name: &str) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.IRModuleContainGlobalVar")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// Return the global variable named `name` (`ir.IRModuleGetGlobalVar`).
    ///
    /// # Errors
    ///
    /// Returns a `ValueError`, which lists the names the module has, if there is none.
    pub fn get_global_var(&self, name: &str) -> Result<GlobalVar> {
        tvm_ffi::cached_global_func!("ir.IRModuleGetGlobalVar")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// Return the global variables of the module, sorted by name (`ir.IRModuleGetGlobalVars`).
    pub fn get_global_vars(&self) -> Result<Array<GlobalVar>> {
        tvm_ffi::cached_global_func!("ir.IRModuleGetGlobalVars").call_tuple((self,))?.try_into()
    }

    /// Return the function of `var` (`ir.IRModuleLookup`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if the module has no function for `var`.
    pub fn lookup(&self, var: impl Into<GlobalVar>) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.IRModuleLookup")
            .call_tuple((self, &var.into()))?
            .try_into()
    }

    /// Return the function named `name` (`ir.IRModuleLookupStr`).
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if the module has no global variable named `name`.
    pub fn lookup_name(&self, name: &str) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.IRModuleLookupStr")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// Add or replace the functions of `other` in the module (`ir.IRModuleUpdate`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if a function of `other` has the name of another global variable
    /// of the module.
    pub fn update(&self, other: &IRModule) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleUpdate").call_tuple((self, other))?;
        Ok(())
    }

    /// Set the function of `var` to `func` (`ir.IRModuleUpdateFunction`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if the module has another global variable of the same name.
    pub fn update_func(&self, var: impl Into<GlobalVar>, func: impl Into<BaseFunc>) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleUpdateFunction").call_tuple((
            self,
            &var.into(),
            &func.into(),
        ))?;
        Ok(())
    }

    /// Set the module-level information `name` to `global_info` (`ir.IRModuleUpdateGlobalInfo`).
    pub fn update_global_info(
        &self,
        name: &str,
        global_info: impl IntoIterator<Item = impl Into<GlobalInfo>>,
    ) -> Result<()> {
        let global_info = global_info.into_iter().map(Into::into).collect::<Array<GlobalInfo>>();
        tvm_ffi::cached_global_func!("ir.IRModuleUpdateGlobalInfo").call_tuple((
            self,
            String::from(name),
            &global_info,
        ))?;
        Ok(())
    }

    /// Return the attribute `key` of the module, or `None` if it is not set (`ir.IRModuleGetAttr`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the value is not an object, such as a `bool` or a short string,
    /// which is stored inline: the native function returns an `Optional<ObjectRef>`.
    pub fn get_attr(&self, key: &str) -> Result<Option<Any>> {
        let value = tvm_ffi::cached_global_func!("ir.IRModuleGetAttr")
            .call_tuple((self, String::from(key)))?;
        let is_none = value.type_index() == tvm_ffi::TypeIndex::kTVMFFINone as i32;
        Ok((!is_none).then_some(value))
    }

    /// Return the module with the attribute `key` set to `value` (`ir.IRModuleWithAttr`).
    ///
    /// The module is updated in place when this was its only handle, and copied otherwise.
    pub fn with_attr(self, key: &str, value: impl Into<Any>) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("ir.IRModuleWithAttr")
            .call_tuple((RValueRef::new(self), String::from(key), value.into()))?
            .try_into()
    }

    /// Return the module with the attributes of `attrs` set (`ir.IRModuleWithAttrs`).
    ///
    /// Consumes `self`, as [`with_attr`](Self::with_attr) does.
    pub fn with_attrs(self, attrs: &Map<String, Any>) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("ir.IRModuleWithAttrs")
            .call_tuple((RValueRef::new(self), attrs))?
            .try_into()
    }

    /// Return the module without the attribute `key` (`ir.IRModuleWithoutAttr`).
    ///
    /// Consumes `self`, as [`with_attr`](Self::with_attr) does.
    pub fn without_attr(self, key: &str) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("ir.IRModuleWithoutAttr")
            .call_tuple((RValueRef::new(self), String::from(key)))?
            .try_into()
    }
}
