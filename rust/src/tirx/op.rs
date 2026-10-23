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

//! Attributes of TIRx operator calls.
//!
//! The objects `python/tvm_tirx/tirx/op/__init__.py` defines. A [`CallFFIKernelAttr`] holds the
//! launch parameters, the launch fields, the number of kernel arguments and the kernel attributes
//! of a `call_ffi_kernel`. The functions build the TIRx operator calls that have a native
//! constructor.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::location::Location
// tvm-ffi-stubgen(import-object): crate::ir::r#type::Type
// tvm-ffi-stubgen(import-object): tvm_ffi::Result

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::attrs::AttrsObj;
use crate::ir::expr::Expr;
use crate::ir::location::Location;
use crate::ir::r#type::Type;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::Map;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.CallFFIKernelAttr
/// Complete: reflected fields fill [24, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.CallFFIKernelAttr"]
#[type_final]
pub struct CallFFIKernelAttrObj {
    base: AttrsObj,
    pub launch_params: Array<String>,
    pub launch_fields: Array<String>,
    pub num_kernel_args: i64,
    pub kernel_attrs: Map<String, i64>,
}

const _: () = {
    assert!(::core::mem::size_of::<CallFFIKernelAttrObj>() == 56);
    assert!(::core::mem::align_of::<CallFFIKernelAttrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct CallFFIKernelAttr {
    base: ObjectArc<CallFFIKernelAttrObj>,
}

impl Deref for CallFFIKernelAttr {
    type Target = CallFFIKernelAttrObj;
    fn deref(&self) -> &CallFFIKernelAttrObj {
        &self.base
    }
}

impl Deref for CallFFIKernelAttrObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl CallFFIKernelAttrObj {
    pub(crate) fn new(
        launch_params: Array<String>,
        launch_fields: Array<String>,
        num_kernel_args: i64,
        kernel_attrs: Map<String, i64>,
    ) -> Self {
        let base = AttrsObj::new();
        Self { base, launch_params, launch_fields, num_kernel_args, kernel_attrs }
    }
}

impl CallFFIKernelAttr {
    /// Lossless complete-field allocation.
    pub fn new(
        launch_params: impl Into<Array<String>>,
        launch_fields: impl Into<Array<String>>,
        num_kernel_args: i64,
        kernel_attrs: impl Into<Map<String, i64>>,
    ) -> Self {
        let obj = CallFFIKernelAttrObj::new(
            launch_params.into(),
            launch_fields.into(),
            num_kernel_args,
            kernel_attrs.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(CallFFIKernelAttr => Attrs);
// tvm-ffi-stubgen(end)

// The native `tirx.CallFFIKernelAttr` stores its fields as given, as the generated
// `CallFFIKernelAttr::new` does.

/// `log(exp(a) + exp(b))` of the floating-point `a` and `b` (`tirx._OpLogAddExp`).
pub fn logaddexp(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("tirx._OpLogAddExp")
        .call_tuple((&a.into(), &b.into(), &loc.into()))?
        .try_into()
}

/// `value` reinterpreted bit for bit as the primitive or pointer type `ty` (`tirx.reinterpret`).
pub fn reinterpret(
    ty: impl Into<Type>,
    value: impl Into<Expr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    tvm_ffi::cached_global_func!("tirx.reinterpret")
        .call_tuple((&ty.into(), &value.into(), &loc.into()))?
        .try_into()
}

/// A return from the current GPU thread, without a value (`tirx.gpu_thread_return`).
pub fn gpu_thread_return(loc: impl Into<Location>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("tirx.gpu_thread_return").call_tuple((&loc.into(),))?.try_into()
}
