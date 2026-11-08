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

//! Modules. An [`IRModule`] is the unit of compilation, the input and output of passes: it maps
//! each [`GlobalVar`] to a function ([`BaseFunc`]), and carries module attributes and module-level
//! information.

// tvm-ffi-stubgen(file): mod(missing_docs = allow)
// `custom_new` reserves `new` for a hand-written constructor, which derives the global variable
// map; the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(object): ir.IRModule(custom_new)
// The name index is internal: C++ registers it with the trailing `_` of a private member.
// tvm-ffi-stubgen(field): ir.IRModule.global_var_map_(visibility = pub(crate))
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import): tvm_ffi::Any
// tvm-ffi-stubgen(import): tvm_ffi::rvalue_ref::RValueRef
// tvm-ffi-stubgen(import): tvm_ffi::Result
// tvm-ffi-stubgen(import): crate::ir::support
// tvm-ffi-stubgen(import): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import): std::borrow::Borrow
// tvm-ffi-stubgen(import): tvm_ffi::IntoArg

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::GlobalVar;
use crate::ir::expr::IntoExpr;
use crate::ir::function::BaseFunc;
use crate::ir::global_info::GlobalInfo;
use crate::ir::support;
use std::borrow::Borrow;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::IntoArg;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::rvalue_ref::RValueRef;
// tvm-ffi-stubgen(end)

/// A set of functions, each bound to a [`GlobalVar`] with a unique name, with the module
/// attributes `attrs` and the module-level information `global_infos`.
///
/// A module is mutable: adding, updating or removing functions changes it through every handle to
/// it, and [`shallow_copy`](Self::shallow_copy) makes one to change independently.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{BaseFunc, DictAttrs, GlobalVar, IRModule, UNKNOWN_LOC};
/// use tvm_tirx::tvm_ffi::{Map, ObjectRefCore};
/// # use tvm_tirx::ir::{PrimType, Return, SeqStmt, Var};
/// # use tvm_tirx::tirx::Function;
/// # let body = SeqStmt::new([Return::new(0, UNKNOWN_LOC)], UNKNOWN_LOC);
/// # let attrs = DictAttrs::new(Map::new());
/// # let func = Function::new(Vec::<Var>::new(), &body, PrimType::int(32), attrs, UNKNOWN_LOC);
///
/// let main = GlobalVar::new("main", UNKNOWN_LOC);
/// let functions = Map::from_iter([(main.clone(), BaseFunc::from(func))]);
/// let module = IRModule::new(&functions, DictAttrs::new(Map::new()), Map::new())?;
///
/// assert!(module.lookup(&main)?.same_as(&module.lookup_name("main")?));
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/ir.IRModule
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IRModule {
    data_: ObjectArc<IRModuleObj>,
}

/// The object behind [`IRModule`].
// Complete: reflected fields fill [24, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.IRModule", final)]
pub struct IRModuleObj {
    base_: Object,
    pub functions: Map<GlobalVar, BaseFunc>,
    pub attrs: DictAttrs,
    pub global_infos: Map<String, Array<GlobalInfo>>,
    pub(crate) global_var_map_: Map<String, GlobalVar>,
}

const _: () = {
    assert!(::core::mem::size_of::<IRModuleObj>() == 56);
    assert!(::core::mem::align_of::<IRModuleObj>() == 8);
};

#[allow(missing_docs)]
impl IRModuleObj {
    pub(crate) fn new(
        functions: Map<GlobalVar, BaseFunc>,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
        global_var_map_: Map<String, GlobalVar>,
    ) -> Self {
        let base_ = Object::new();
        Self { base_, functions, attrs, global_infos, global_var_map_ }
    }
}

#[allow(missing_docs)]
impl IRModule {
    /// Create a `IRModule` from its field values.
    pub fn from_fields(
        functions: impl Into<Map<GlobalVar, BaseFunc>>,
        attrs: impl IntoArg<DictAttrs>,
        global_infos: impl Into<Map<String, Array<GlobalInfo>>>,
        global_var_map_: impl Into<Map<String, GlobalVar>>,
    ) -> Self {
        let obj = IRModuleObj::new(
            functions.into(),
            attrs.into_owned(),
            global_infos.into(),
            global_var_map_.into(),
        );
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl IRModule {
    /// Create the module of `functions`, with the attributes `attrs` and the module-level
    /// information `global_infos`, keyed by name.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if two global variables of `functions` have the same name.
    pub fn new(
        functions: impl Into<Map<GlobalVar, BaseFunc>>,
        attrs: impl IntoArg<DictAttrs>,
        global_infos: impl Into<Map<String, Array<GlobalInfo>>>,
    ) -> Result<Self> {
        let functions = functions.into();
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
        Ok(Self::from_fields(functions, attrs, global_infos, global_var_map))
    }

    /// Create the module of `functions` and `expr`, a function bound to its `global_symbol`, or to
    /// `main` (made unique) when it has none (`ir.IRModuleFromExpr`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `expr` is not a function.
    pub fn from_expr(
        expr: impl IntoExpr,
        functions: impl Into<Map<GlobalVar, BaseFunc>>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.IRModuleFromExpr")
            .call_tuple((expr.into_expr()?.borrow(), &functions.into()))?
            .try_into()
    }

    /// Return a copy of the module that shares its functions (`ir.IRModuleClone`).
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
        var: impl IntoArg<GlobalVar>,
        func: impl IntoArg<BaseFunc>,
        update: bool,
    ) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleAdd").call_tuple((
            self,
            var.into_arg().borrow(),
            func.into_arg().borrow(),
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
    pub fn remove_global_var(&self, var: impl IntoArg<GlobalVar>) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleRemove")
            .call_tuple((self, var.into_arg().borrow()))?;
        Ok(())
    }

    /// Return whether the module has a global variable named `name` (`ir.IRModuleContains`).
    pub fn contains(&self, name: &str) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.IRModuleContains")
            .call_tuple((self, String::from(name)))?
            .try_into()
    }

    /// Return whether the module defines a function for `var` (`ir.IRModuleContains`).
    pub fn contains_global_var(&self, var: impl IntoArg<GlobalVar>) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.IRModuleContains")
            .call_tuple((self, var.into_arg().borrow()))?
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
    pub fn lookup(&self, var: impl IntoArg<GlobalVar>) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.IRModuleLookup")
            .call_tuple((self, var.into_arg().borrow()))?
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
    pub fn update(&self, other: impl IntoArg<IRModule>) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleUpdate")
            .call_tuple((self, other.into_arg().borrow()))?;
        Ok(())
    }

    /// Set the function of `var` to `func` (`ir.IRModuleUpdateFunction`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if the module has another global variable of the same name.
    pub fn update_func(
        &self,
        var: impl IntoArg<GlobalVar>,
        func: impl IntoArg<BaseFunc>,
    ) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.IRModuleUpdateFunction").call_tuple((
            self,
            var.into_arg().borrow(),
            func.into_arg().borrow(),
        ))?;
        Ok(())
    }

    /// Set the module-level information `name` to `global_info` (`ir.IRModuleUpdateGlobalInfo`).
    pub fn update_global_info(
        &self,
        name: &str,
        global_info: impl IntoIterator<Item = impl IntoArg<GlobalInfo>>,
    ) -> Result<()> {
        let global_info =
            global_info.into_iter().map(IntoArg::into_owned).collect::<Array<GlobalInfo>>();
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
        Ok((!value.is_none()).then_some(value))
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
    pub fn with_attrs(self, attrs: impl Into<Map<String, Any>>) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("ir.IRModuleWithAttrs")
            .call_tuple((RValueRef::new(self), &attrs.into()))?
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
