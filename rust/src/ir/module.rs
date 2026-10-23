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
//! The objects `python/tvm_tirx/ir/module.py` defines. An [`IRModule`] maps global variables to
//! functions, and holds the module attributes, the module-level information and the names of its
//! global variables. [`IRModule::from_functions`] is written in Rust after the C++ constructor;
//! the other methods wrap the native `ir.IRModule*` functions.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::RValueRef
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::Expr;
use crate::ir::expr::GlobalVar;
use crate::ir::function::BaseFunc;
use crate::ir::global_info::GlobalInfo;
use crate::ir::support;
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
    /// The module of `functions`, with the attributes `attrs` (empty if `None`) and the
    /// module-level information `global_infos`; it indexes the functions by name, which must be
    /// distinct (C++ `IRModule(functions, attrs, global_infos)`).
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

    /// The module of `functions` and `expr`, a function bound to its global symbol or to `main`
    /// (`ir.IRModuleFromExpr`).
    pub fn from_expr(expr: impl Into<Expr>, functions: &Map<GlobalVar, BaseFunc>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.IRModuleFromExpr")
            .call_tuple((&expr.into(), functions))?
            .try_into()
    }

    /// A copy of the module that shares its functions (`ir.IRModuleClone`).
    pub fn shallow_copy(&self) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.IRModuleClone").call_tuple((self,))?.try_into()
    }

    /// Add `func` as `var`, replacing an existing definition; the native side currently ignores
    /// `update` (`ir.IRModuleAdd`).
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
    pub fn remove(&self, name: &str) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleRemove").call_tuple((self, String::from(name)))?;
        Ok(())
    }

    /// Remove the function of `var` (`ir.IRModuleRemove`).
    pub fn remove_global_var(&self, var: impl Into<GlobalVar>) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleRemove").call_tuple((self, &var.into()))?;
        Ok(())
    }

    /// Whether the module has a global variable named `name` (`ir.IRModuleContains`).
    pub fn contains(&self, name: &str) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.IRModuleContains")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// Whether the module defines a function for `var` (`ir.IRModuleContains`).
    pub fn contains_global_var(&self, var: impl Into<GlobalVar>) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.IRModuleContains")
            .call_tuple((self, &var.into()))?
            .try_into()
    }

    /// Whether the module has a global variable named `name` (`ir.IRModuleContainGlobalVar`).
    pub fn has_global_var(&self, name: &str) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.IRModuleContainGlobalVar")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// The global variable named `name`; fails if there is none (`ir.IRModuleGetGlobalVar`).
    pub fn get_global_var(&self, name: &str) -> Result<GlobalVar> {
        tvm_ffi::cached_global_func!("ir.IRModuleGetGlobalVar")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// The global variables of the module, sorted by name (`ir.IRModuleGetGlobalVars`).
    pub fn get_global_vars(&self) -> Result<Array<GlobalVar>> {
        tvm_ffi::cached_global_func!("ir.IRModuleGetGlobalVars").call_tuple((self,))?.try_into()
    }

    /// The function of `var`; fails if there is none (`ir.IRModuleLookup`).
    pub fn lookup(&self, var: impl Into<GlobalVar>) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.IRModuleLookup")
            .call_tuple((self, &var.into()))?
            .try_into()
    }

    /// The function named `name`; fails if there is none (`ir.IRModuleLookupStr`).
    pub fn lookup_name(&self, name: &str) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.IRModuleLookupStr")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// Add or replace the functions of `other` in the module (`ir.IRModuleUpdate`).
    pub fn update(&self, other: &IRModule) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleUpdate").call_tuple((self, other))?;
        Ok(())
    }

    /// Set the function of `var` to `func` (`ir.IRModuleUpdateFunction`).
    pub fn update_func(&self, var: impl Into<GlobalVar>, func: impl Into<BaseFunc>) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleUpdateFunction").call_tuple((
            self,
            &var.into(),
            &func.into(),
        ))?;
        Ok(())
    }

    /// Set the module-level information `name` to `global_info`
    /// (`ir.IRModuleUpdateGlobalInfo`).
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

    /// The attribute `key` of the module, if set (`ir.IRModuleGetAttr`).
    pub fn get_attr(&self, key: &str) -> Result<Option<Any>> {
        let value = tvm_ffi::cached_global_func!("ir.IRModuleGetAttr")
            .call_tuple((self, String::from(key)))?;
        let is_none = value.type_index() == tvm_ffi::TypeIndex::kTVMFFINone as i32;
        Ok((!is_none).then_some(value))
    }

    /// The module with the attribute `key` set to `value` (`ir.IRModuleWithAttr`).
    ///
    /// Consumes `self`: the module is updated in place when this was its only handle, and copied
    /// otherwise.
    pub fn with_attr(self, key: &str, value: impl Into<Any>) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("ir.IRModuleWithAttr")
            .call_tuple((RValueRef::new(self), String::from(key), value.into()))?
            .try_into()
    }

    /// The module with the attributes of `attrs` set (`ir.IRModuleWithAttrs`).
    ///
    /// Consumes `self`, as [`IRModule::with_attr`] does.
    pub fn with_attrs(self, attrs: &Map<String, Any>) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("ir.IRModuleWithAttrs")
            .call_tuple((RValueRef::new(self), attrs))?
            .try_into()
    }

    /// The module without the attribute `key` (`ir.IRModuleWithoutAttr`).
    ///
    /// Consumes `self`, as [`IRModule::with_attr`] does.
    pub fn without_attr(self, key: &str) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("ir.IRModuleWithoutAttr")
            .call_tuple((RValueRef::new(self), String::from(key)))?
            .try_into()
    }
}
