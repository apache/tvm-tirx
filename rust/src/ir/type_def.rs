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

//! Registering IR types from Rust: the [`register_types!`](crate::register_types) blocks that
//! define `rs_class` types, as C++ defines types with `refl::ObjectDef` in a
//! `TVM_FFI_STATIC_INIT_BLOCK`, and the [`Placeholder`] values of their fields.
//!
//! `rs_class` registers the fields of a type, but not a way to make one. C++ `refl::ObjectDef`
//! also registers a creator of an empty object, which tvm-ffi fills through the reflection of the
//! fields to construct the type from Python (the `__ffi_init__` it generates) and to read it from
//! a JSON graph. A Rust object cannot be empty, as its references are not null, so a block
//! registers `__ffi_new__`, which makes an object whose fields hold [`Placeholder`]s, and tvm-ffi
//! then sets each field as it sets those of a C++ object.

use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeRegisterAttr};
use tvm_ffi::{Any, AnyCompatible, Array, Function, ObjectCore, ObjectRefCore, Result, String};

use super::expr::{Expr, IntImm};
use super::location::Location;
use super::r#type::{MissingType, PrimType, Type, TypeObj};
use super::support;

/// A valid value of a field type, which an object of a [`register_types!`](crate::register_types)
/// block holds until reflection sets the field: `__ffi_new__` makes an object with one in every
/// field, and a field `__ffi_init__` does not take (`#[reflect(init_off)]`) keeps it. A block
/// implements it for the objects and references it defines, and this crate for the common field
/// types; implement it for a field type of your own crate.
pub trait Placeholder {
    /// The value.
    fn placeholder() -> Self;
}

macro_rules! default_placeholder {
    ($($ty:ty),*) => {
        $(impl Placeholder for $ty {
            fn placeholder() -> Self {
                Self::default()
            }
        })*
    };
}

default_placeholder!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl Placeholder for String {
    fn placeholder() -> Self {
        String::from("")
    }
}

impl<T: AnyCompatible + Clone + 'static> Placeholder for Array<T> {
    fn placeholder() -> Self {
        Array::new(vec![])
    }
}

impl<T> Placeholder for Option<T> {
    fn placeholder() -> Self {
        None
    }
}

impl Placeholder for Location {
    fn placeholder() -> Self {
        Location::unknown()
    }
}

impl Placeholder for TypeObj {
    fn placeholder() -> Self {
        TypeObj::new(Location::unknown())
    }
}

impl Placeholder for Type {
    fn placeholder() -> Self {
        MissingType::new().into()
    }
}

impl Placeholder for PrimType {
    fn placeholder() -> Self {
        PrimType::void()
    }
}

impl Placeholder for Expr {
    fn placeholder() -> Self {
        IntImm::new(Location::unknown(), PrimType::int(32), 0).into()
    }
}

/// Register the type of the reference `R` of a [`register_types!`](crate::register_types) block:
/// `__ffi_new__`, then the `__ffi_init__` tvm-ffi generates from the reflected fields, then, once
/// the Python receiver is available, request its TVMScript class (`script.expose_script_type`).
/// An absent receiver is a no-op; an installed receiver's error is returned.
#[doc(hidden)]
pub fn __register_type<R>() -> Result<()>
where
    R: ObjectRefCore + AnyCompatible + Placeholder + 'static,
{
    let type_index = R::ContainerType::type_index();
    let new = Function::from_typed(|| -> Result<R> { Ok(R::placeholder()) });
    register_type_attr(type_index, "__ffi_new__", Any::from(new))?;
    tvm_ffi::cached_global_func!("ffi._RegisterFFIInit").call_tuple((type_index,))?;
    if let Ok(expose) = Function::get_global("script.expose_script_type") {
        expose.call_tuple((String::from(support::type_key(type_index).as_str()),))?;
    }
    Ok(())
}

/// Register `value` as the type attribute `name` of the type of `type_index`.
fn register_type_attr(type_index: i32, name: &str, mut value: Any) -> Result<()> {
    // SAFETY: `name` and `value` outlive the call, which copies them.
    let name = unsafe { TVMFFIByteArray::from_str(name) };
    tvm_ffi::check_safe_call!(unsafe {
        TVMFFITypeRegisterAttr(type_index, &name, Any::as_data_ptr(&mut value))
    })
}

/// Define IR types as C++ does with `refl::ObjectDef` in a `TVM_FFI_STATIC_INIT_BLOCK`: each item
/// is an `rs_class` object struct and the `def_object_ref!` of its reference, written as outside a
/// block.
///
/// ```no_run
/// use tvm_tirx::ir::{Type, TypeObj};
/// use tvm_tirx::tvm_ffi::{self, rs_class};
///
/// tvm_tirx::register_types! {
///     /// The type of a pair of values, printed as `my_ext.Pair(first, second)`.
///     #[rs_class("my_ext.PairType", final)]
///     pub struct PairTypeObj {
///         base: TypeObj,
///         pub first: Type,
///         #[reflect(kw_only)]
///         pub second: Type,
///     }
///     def_object_ref!(pub PairType, Type, PairTypeObj);
/// }
///
/// ```
///
/// The block registers its types when its program or library loads, as a
/// [`register_ops!`](crate::register_ops) block registers its operators, and once per process.
/// Python then constructs a type from its fields, `my_ext.Pair(T.int32, second=T.float32)`, as it
/// constructs a C++ type, a JSON graph holds it, and, when Python has imported `tvm_tirx` before it
/// loads the library, the type gets the class its TVMScript constructor names. The type of each
/// field and the parent's object implement [`Placeholder`].
#[macro_export]
macro_rules! register_types {
    ($(
        $(#[$attr:meta])*
        $vis:vis struct $obj:ident {
            $base:ident: $base_ty:ty
            $(, $(#[$field_attr:meta])* $field_vis:vis $field:ident: $field_ty:ty)* $(,)?
        }
        def_object_ref!($ref_vis:vis $ref:ident, $parent:ty, $ref_obj:ident $(,)?);
    )*) => {
        $(
            $(#[$attr])*
            $vis struct $obj {
                $base: $base_ty,
                $($(#[$field_attr])* $field_vis $field: $field_ty,)*
            }

            $crate::tvm_ffi::def_object_ref!($ref_vis $ref, $parent, $ref_obj);

            impl $crate::ir::Placeholder for $obj {
                fn placeholder() -> Self {
                    $obj::new(
                        <$base_ty as $crate::ir::Placeholder>::placeholder(),
                        $(<$field_ty as $crate::ir::Placeholder>::placeholder(),)*
                    )
                }
            }

            impl $crate::ir::Placeholder for $ref {
                fn placeholder() -> Self {
                    <$obj as $crate::ir::Placeholder>::placeholder().into()
                }
            }
        )*

        $crate::__register_block!(
            "register_types!",
            stringify!($(
                $(#[$attr])*
                $vis struct $obj {
                    $base: $base_ty
                    $(, $(#[$field_attr])* $field_vis $field: $field_ty)*
                }
                def_object_ref!($ref_vis $ref, $parent, $ref_obj);
            )*),
            || {
                $($crate::ir::__register_type::<$ref>()?;)*
                Ok(())
            }
        );
    };
}
