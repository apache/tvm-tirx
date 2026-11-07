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

//! Type registration. An IR type defined in Rust with `rs_object` calls [`register_type`] in a
//! [`tvm_ffi::static_init_block!`], so Python can construct it once its library loads.

use tvm_ffi::{Function, ObjectCore, ObjectRefCore, Result, String};

/// Expose the IR type of the reference `R`, an `rs_object` type, to Python.
///
/// Call it in a [`tvm_ffi::static_init_block!`], so that the type is registered when its program or
/// library loads. Python then constructs the type from its fields, like a built-in type, if
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
/// use tvm_tirx::tvm_ffi::{self, rs_object, IntoObjectRef};
///
/// #[rs_object("my_ext.PairType", final)]
/// pub struct PairTypeObj {
///     base_: TypeObj,
///     pub first: Type,
///     #[reflect(kw_only)]
///     pub second: Type,
/// }
/// #[derive(Clone, tvm_ffi::derive::ObjectRef)]
/// #[object_ref(parent = Type)]
/// pub struct PairType {
///     data_: tvm_ffi::ObjectArc<PairTypeObj>,
/// }
///
/// impl PairType {
///     pub fn new(first: impl IntoObjectRef<Type>, second: impl IntoObjectRef<Type>) -> Self {
///         let (first, second) = (first.into_owned_object_ref(), second.into_owned_object_ref());
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
    // The key of the registered type, which `type_index` registers first.
    let key = tvm_ffi::reflection::type_key(R::ContainerType::type_index()).unwrap_or_default();
    if let Ok(expose) = Function::get_global("ir.expose_type") {
        expose.call_tuple((String::from(key),))?;
    }
    Ok(())
}
