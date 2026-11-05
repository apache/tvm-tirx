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

//! Operators: the registry entry [`Op`], its [`ArgumentInfo`], and its attribute columns.
//!
//! [`Op::get`] looks up a registered operator, and [`Op::attr_map`] reads an attribute column as an
//! [`OpAttrMap`].
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{Call, CallEffectKind, Op, PrimType, Type, Var, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//!
//! let exp = Op::get("prim.exp")?;
//! let x = Var::new("x", PrimType::float(32), UNKNOWN_LOC);
//! let call = Call::new(None, exp.clone(), [&x], None, Vec::<Type>::new(), UNKNOWN_LOC)?;
//! assert_eq!(script(&call)?, "T.exp(x)");
//!
//! let effects = Op::attr_map::<i64>("TCallEffectKind")?.unwrap();
//! assert_eq!(effects.get(&exp)?, Some(CallEffectKind::Pure as i64));
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// `ir.Op` has three unreflected members, so its complete layout is hand-written below.
// tvm-ffi-stubgen(ty-map): ir.Op -> crate::ir::op::Op
// Names the hand-written `Op` binding uses: the import section keeps them even when no
// generated block does.
// tvm-ffi-stubgen(import-object): std::ops::Deref
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectArc
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::expr::ExprObj
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// The layout is hand-written (the block is `skip`ped): `attrs_type_index`, `validator`, and
// `validator_is_custom` are not reflected, so the generator can only emit an opaque view of
// `OpObj`. Keep the members in step with `OpObj` in `include/tvm/ir/op.h`.
/// The object of an operator registry entry (`ir.Op`).
///
/// Only the registry constructs one: get an operator with [`Op::get`].
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Op"]
#[type_final]
pub struct OpObj {
    base: ExprObj,
    /// The dense, process-local index of the operator into the attribute columns; never
    /// serialized.
    pub index: u32,
    /// The canonical name of the operator, such as `"prim.exp"`.
    pub name: String,
    /// The documentation of the operator.
    pub doc: String,
    /// The descriptions of the required value arguments, in order.
    pub args_info: Array<ArgumentInfo>,
    /// The descriptions of the required type arguments, in order.
    pub ty_args_info: Array<ArgumentInfo>,
    /// The description of the variadic value tail after [`args_info`](Self::args_info), if any.
    pub var_args_info: Option<ArgumentInfo>,
    /// The description of the variadic type-argument tail after
    /// [`ty_args_info`](Self::ty_args_info), if any.
    pub var_ty_args_info: Option<ArgumentInfo>,
    /// The type key of the operator's `Call.attrs`, or empty when it declares none.
    pub attrs_type_key: String,
    attrs_type_index: u32,
    validator: Any,
    validator_is_custom: bool,
}

const _: () = {
    assert!(::core::mem::size_of::<OpObj>() == 160);
    assert!(::core::mem::align_of::<OpObj>() == 8);
    assert!(::core::mem::offset_of!(OpObj, index) == 40);
    assert!(::core::mem::offset_of!(OpObj, name) == 48);
    assert!(::core::mem::offset_of!(OpObj, doc) == 64);
    assert!(::core::mem::offset_of!(OpObj, args_info) == 80);
    assert!(::core::mem::offset_of!(OpObj, ty_args_info) == 88);
    assert!(::core::mem::offset_of!(OpObj, var_args_info) == 96);
    assert!(::core::mem::offset_of!(OpObj, var_ty_args_info) == 104);
    assert!(::core::mem::offset_of!(OpObj, attrs_type_key) == 112);
    assert!(::core::mem::offset_of!(OpObj, attrs_type_index) == 128);
    assert!(::core::mem::offset_of!(OpObj, validator) == 136);
    assert!(::core::mem::offset_of!(OpObj, validator_is_custom) == 152);
};

/// A registered operator, the callee of a [`Call`](crate::ir::Call) to it.
///
/// The registry keeps one object per name, so `call.op.same_as(&op)` tests whether a call is to
/// `op`. Register a new operator with [`OpDef`](crate::ir::OpDef).
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Op {
    base: ObjectArc<OpObj>,
}

impl Deref for Op {
    type Target = OpObj;
    fn deref(&self) -> &OpObj {
        &self.base
    }
}

impl Deref for OpObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl OpObj {
    /// Return the runtime type index of [`attrs_type_key`](Self::attrs_type_key), or `0` when the
    /// operator declares no attrs type.
    pub fn attrs_type_index(&self) -> u32 {
        self.attrs_type_index
    }

    /// Return the validator that [`Call::validate`](crate::ir::Call::validate) runs: a native
    /// function pointer, a packed function, or `None`.
    pub fn validator(&self) -> &Any {
        &self.validator
    }

    /// Return whether the validator was set with
    /// [`OpDef::set_validator`](crate::ir::OpDef::set_validator) rather than generated from a
    /// signature.
    pub fn validator_is_custom(&self) -> bool {
        self.validator_is_custom
    }
}

impl Op {
    /// Look up the operator registered as `name`, a registry singleton as C++ `Op::Get` returns it
    /// (`ir.GetOp`).
    ///
    /// # Errors
    ///
    /// Returns an `AttributeError` if no operator is registered as `name`.
    pub fn get(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.GetOp").call_tuple((String::from(name),))?.try_into()
    }

    /// Return whether any operator registered the attribute `name`, as C++ `Op::HasAttrMap` does
    /// (`ir.ListOpAttrNames`).
    pub fn has_attr_map(name: &str) -> Result<bool> {
        Ok(Self::list_attr_names()?.iter().any(|column| column.as_str() == name))
    }

    /// Return the live attribute column `name`, as C++ `Op::GetAttrMap<T>` does, or `None` when no
    /// operator registered it (`ir.GetOpAttrColumn`).
    ///
    /// The map also sees the values registered after it was read.
    pub fn attr_map<T: tvm_ffi::type_traits::AnyCompatible>(
        name: &str,
    ) -> Result<Option<OpAttrMap<T>>> {
        Self::attr_column(name)
    }

    /// [`Op::attr_map`] for a column read entry by entry as raw values ([`OpAttrMap::get_any`]).
    pub(crate) fn attr_column<T>(name: &str) -> Result<Option<OpAttrMap<T>>> {
        let column =
            tvm_ffi::cached_global_func!("ir.GetOpAttrColumn").call_tuple((String::from(name),))?;
        if column.type_index() == tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(None);
        }
        if column.type_index() != tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex::kTVMFFIList as i32 {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::TYPE_ERROR,
                &format!("Op attribute column {name} is not an ffi.List"),
                "",
            ));
        }
        Ok(Some(OpAttrMap {
            column: column.try_into()?,
            name: String::from(name),
            _value: std::marker::PhantomData,
        }))
    }

    /// Return the names of the registered operators (`ir.ListOpNames`).
    pub fn list_op_names() -> Result<Array<String>> {
        tvm_ffi::cached_global_func!("ir.ListOpNames").call_tuple(())?.try_into()
    }

    /// Return the names of the attributes any operator registered (`ir.ListOpAttrNames`).
    pub fn list_attr_names() -> Result<Array<String>> {
        tvm_ffi::cached_global_func!("ir.ListOpAttrNames").call_tuple(())?.try_into()
    }

    /// Register a new operator `name` with the documentation `doc` and no signature or attributes
    /// (`ir.RegisterOp`).
    ///
    /// # Errors
    ///
    /// Returns an `AttributeError` if `name` is already registered.
    pub fn register(name: &str, doc: &str) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.RegisterOp")
            .call_tuple((String::from(name), String::from(doc)))?;
        Ok(())
    }

    /// Store `validator` as this operator's validator in place, as C++ `OpDef::set_validator` and
    /// `OpDef::signature` write the registry's node: an opaque pointer to a C ABI adapter, as
    /// `NativeFunctionView::FromNative` stores one.
    pub(crate) fn write_validator(&self, validator: Any, custom: bool) {
        // SAFETY: C++ `OpDef` writes these members of the registry's node in place too. No
        // reference into the node is live, and registration is not synchronized with readers.
        unsafe {
            let node = ObjectArc::as_raw(ObjectRefCore::data(self)) as *mut OpObj;
            let previous = std::ptr::replace(std::ptr::addr_of_mut!((*node).validator), validator);
            std::ptr::addr_of_mut!((*node).validator_is_custom).write(custom);
            drop(previous);
        }
    }
}

tvm_ffi::impl_object_upcast!(Op => Expr);

// tvm-ffi-stubgen(begin): object/ir.ArgumentInfo
/// Complete: reflected fields fill [24, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.ArgumentInfo"]
#[type_final]
pub struct ArgumentInfoObj {
    base: Object,
    pub name: String,
    pub doc: String,
}

const _: () = {
    assert!(::core::mem::size_of::<ArgumentInfoObj>() == 56);
    assert!(::core::mem::align_of::<ArgumentInfoObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ArgumentInfo {
    base: ObjectArc<ArgumentInfoObj>,
}

impl Deref for ArgumentInfo {
    type Target = ArgumentInfoObj;
    fn deref(&self) -> &ArgumentInfoObj {
        &self.base
    }
}

impl ArgumentInfoObj {
    pub(crate) fn new(name: String, doc: String) -> Self {
        let base = Object::new();
        Self { base, name, doc }
    }
}

impl ArgumentInfo {
    /// Lossless complete-field allocation.
    pub fn new(name: impl Into<String>, doc: impl Into<String>) -> Self {
        let obj = ArgumentInfoObj::new(name.into(), doc.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// The crate has no `ffi.List` binding, so the column is read through the `TVMFFISeqCell` that
// `ffi.List` shares with `ffi.Array` (`c_api.h`).
/// A live attribute column of the operator registry, C++ `OpAttrMap<T>`, returned by
/// [`Op::attr_map`].
///
/// It holds the registry's own column, so values registered later stay visible and the map can be
/// cached. A lookup reads the entry without an FFI call.
pub struct OpAttrMap<T> {
    column: tvm_ffi::object::ObjectRef,
    name: String,
    _value: std::marker::PhantomData<T>,
}

impl<T> OpAttrMap<T> {
    /// The entry of `op` in the column, unless it has none.
    fn entry(&self, op: &OpObj) -> Option<&tvm_ffi::tvm_ffi_sys::TVMFFIAny> {
        // SAFETY: `column` is an `ffi.List` (checked by `Op::attr_map`) that `self` keeps alive.
        // The cell is read on every lookup because growth may move the element buffer.
        let cell = unsafe {
            let data = tvm_ffi::object::ObjectRefCore::data(&self.column);
            &*(ObjectArc::as_raw(data) as *const support::SeqCell)
        };
        let index = op.index as usize;
        if index >= cell.size as usize {
            return None;
        }
        // SAFETY: `index` is within the `size` initialized entries.
        let entry = unsafe { &*cell.data.add(index) };
        (entry.type_index != tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex::kTVMFFINone as i32)
            .then_some(entry)
    }

    /// The value `op` registers in this column, of whatever type, or `None`.
    pub(crate) fn get_any(&self, op: &OpObj) -> Option<Any> {
        self.entry(op).map(|entry| {
            // SAFETY: the borrowed entry is copied, never released.
            let entry = std::mem::ManuallyDrop::new(unsafe { Any::from_raw_ffi_any(*entry) });
            (*entry).clone()
        })
    }
}

impl<T: tvm_ffi::type_traits::AnyCompatible> OpAttrMap<T> {
    /// Return the value registered for `op`, or `None` when it has none, as C++
    /// `OpAttrMap::get(op, None)` does.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the value is not a `T`.
    pub fn get(&self, op: &OpObj) -> Result<Option<T>> {
        let Some(entry) = self.entry(op) else { return Ok(None) };
        // SAFETY: `entry` is a live element of the column.
        match unsafe { T::try_cast_from_any_view(entry) } {
            Ok(value) => Ok(Some(value)),
            Err(()) => Err(tvm_ffi::Error::new(
                tvm_ffi::TYPE_ERROR,
                &format!(
                    "Op attribute {} of {} has unexpected type {}",
                    self.name.as_str(),
                    op.name.as_str(),
                    T::get_mismatch_type_info(entry)
                ),
                "",
            )),
        }
    }
}
