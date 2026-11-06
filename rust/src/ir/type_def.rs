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
//! calls for an `rs_object` type, as C++ registers a type with `refl::ObjectDef`.

use tvm_ffi::{Function, ObjectCore, ObjectRefCore, Result, String};

use super::support;

/// Register the IR type of the reference `R`, an `rs_object` type, as C++ registers one with
/// `refl::ObjectDef`.
///
/// Call it in a [`tvm_ffi::static_init_block!`], so that the type is registered when its program or
/// library loads. Python then constructs the type from its fields, as it constructs a C++ type, if
/// `tvm_tirx` is imported before the library loads.
///
/// # Errors
///
/// Returns the error of the installed `ir.expose_type` receiver.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{register_type, PrimType, Type, TypeObj, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::tvm_ffi::{self, rs_object};
///
/// #[rs_object("my_ext.PairType", final)]
/// pub struct PairTypeObj {
///     base: TypeObj,
///     pub first: Type,
///     #[reflect(kw_only)]
///     pub second: Type,
/// }
/// #[derive(Clone, tvm_ffi::derive::ObjectRef)]
/// #[object_ref(parent = Type)]
/// pub struct PairType {
///     data: tvm_ffi::ObjectArc<PairTypeObj>,
/// }
///
/// impl PairType {
///     pub fn new(first: impl Into<Type>, second: impl Into<Type>) -> Self {
///         Self::from_object(PairTypeObj::new(TypeObj::new(UNKNOWN_LOC.into()), first, second))
///     }
/// }
///
/// tvm_ffi::static_init_block! {
///     register_type::<PairType>()?;
/// }
///
/// fn main() -> tvm_ffi::Result<()> {
///     let p = Var::new("p", PairType::new(PrimType::int(32), PrimType::float(32)), UNKNOWN_LOC);
///     assert_eq!(script(&p.ty)?, "my_ext.Pair(T.int32, second=T.float32)");
///     Ok(())
/// }
/// ```
pub fn register_type<R: ObjectRefCore>() -> Result<()> {
    let type_index = R::ContainerType::type_index();
    if let Ok(expose) = Function::get_global("ir.expose_type") {
        expose.call_tuple((String::from(support::type_key(type_index).as_str()),))?;
    }
    Ok(())
}
