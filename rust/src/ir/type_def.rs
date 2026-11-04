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

//! Registering IR types from Rust: [`register_type`], which a [`tvm_ffi::static_init_block!`]
//! calls for an `rs_object` type, as C++ registers a type with `refl::ObjectDef` in a
//! `TVM_FFI_STATIC_INIT_BLOCK`.
//!
//! `rs_object` registers the fields of a type and, as C++ `refl::ObjectDef` does, a creator of an
//! object whose fields are undefined, which tvm-ffi fills through the reflection of the fields to
//! construct the type from Python (the `__ffi_init__` it generates) and to read it from a JSON
//! graph (`__ffi_new__`). [`register_type`] registers the type when its program or library loads
//! and exposes its constructor to Python.

use tvm_ffi::{Function, ObjectCore, ObjectRefCore, Result, String};

use super::support;

/// Register the IR type of the reference `R`, an `rs_object` type, as C++ registers one with
/// `refl::ObjectDef`: the type, with the `__ffi_new__` and `__ffi_init__` `rs_object` registers,
/// then, when the receiver is installed, a request for its constructor in registered destinations
/// (`ir.expose_type`); an absent receiver is a no-op, and an installed receiver's error is
/// returned. With an explicitly declared `my_ext` type API destination, Python then constructs the
/// type from its fields, `my_ext.Pair(T.int32, second=T.float32)`, as it constructs a C++ type, and
/// a JSON graph holds it. Import `tvm_tirx` before loading the library so exposure requests reach
/// Python. Import the extension module in Python scope, or supply it to text parsing through
/// imports, `extra_vars` or explicit parser namespace registration; type exposure does not add
/// parser defaults.
///
/// Call it in a [`tvm_ffi::static_init_block!`], with the [`OpDef`](crate::ir::OpDef) chains of
/// the operators, so that the type is registered when its program or library loads, as a C++ one
/// is; the type's struct, reference and `impl`s are written outside the block, as any other:
///
/// ```no_run
/// use tvm_tirx::ir::{register_type, Location, Type, TypeObj};
/// use tvm_tirx::tvm_ffi::{self, def_object_ref, rs_object, ObjectRefCore};
///
/// /// The type of a pair of values, printed as `my_ext.Pair(first, second)`.
/// #[rs_object("my_ext.PairType", final)]
/// pub struct PairTypeObj {
///     base: TypeObj,
///     pub first: Type,
///     #[reflect(kw_only)]
///     pub second: Type,
/// }
/// def_object_ref!(pub PairType, Type, PairTypeObj);
///
/// impl PairType {
///     pub fn new(first: impl Into<Type>, second: impl Into<Type>) -> Self {
///         PairTypeObj::new(TypeObj::new(Location::unknown()), first, second).into()
///     }
/// }
///
/// tvm_ffi::static_init_block! {
///     register_type::<PairType>()?;
/// }
/// ```
pub fn register_type<R: ObjectRefCore>() -> Result<()> {
    let type_index = R::ContainerType::type_index();
    if let Ok(expose) = Function::get_global("ir.expose_type") {
        expose.call_tuple((String::from(support::type_key(type_index).as_str()),))?;
    }
    Ok(())
}
