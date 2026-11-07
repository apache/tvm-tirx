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

//! Tensors. A tensor is a [`Var`](crate::ir::Var) whose type is a [`TensorType`], which fixes its
//! shape, element type, memory scope, alignment and [`Layout`]; loads and stores subscript the
//! variable, and [`TensorVar`] matches one in an operator signature.

// tvm-ffi-stubgen(missing-docs): allow
// `custom-new` reserves `new` for a hand-written constructor (defaults); the generated
// `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): tirx.TensorType

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::sig
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// A tensor's shape is a list of `PrimExpr`s, as C++ `TensorTypeObj` holds it.
// tvm-ffi-stubgen(field): tirx.TensorType.shape -> Array<PrimExpr>
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::location::Location;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::Type;
use crate::ir::r#type::TypeObj;
use crate::ir::sig;
use crate::runtime::MemoryScope;
use crate::sym::analyzer::Analyzer;
use crate::tirx::layout::Layout;
use std::ops::Deref;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

/// The type of a tensor of `shape` and element type `dtype` in the memory `scope`, with its data
/// pointer aligned to `data_alignment` bytes and an optional `layout` from indices to storage.
///
/// The shape is the one loads and stores index. The data pointer is not part of the type; the
/// definition of the tensor variable binds it.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{Expr, PrimType, PrimVar, Subscript, Var, UNKNOWN_LOC};
/// use tvm_tirx::runtime::MemoryScope;
/// use tvm_tirx::script::script;
/// use tvm_tirx::tirx::TensorType;
/// use tvm_tirx::tvm_ffi::reflection::EnumDef;
///
/// let scratch = EnumDef::<MemoryScope>::new("my_ext.scratch")?.value();
/// let ty = TensorType::new([16, 16], PrimType::float(32), &scratch, 0, None, UNKNOWN_LOC)?;
/// assert_eq!(script(&ty)?, r#"T.Tensor((16, 16), T.float32, scope="my_ext.scratch")"#);
///
/// let a = Expr::from(Var::new("A", &ty, UNKNOWN_LOC));
/// let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let point = [Subscript::Index(i.into()), Subscript::Index(0.into())];
/// let element = a.subscript(point, UNKNOWN_LOC)?;
/// assert_eq!(script(&Expr::try_from(element)?)?, "A[i, 0]");
///
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/tirx.TensorType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorType {
    data_: ObjectArc<TensorTypeObj>,
}

/// The object behind [`TensorType`].
///
/// Complete: reflected fields fill [32, 72) exactly (alignment padding [60, 64)).
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TensorType"]
#[type_final]
pub struct TensorTypeObj {
    base_: TypeObj,
    pub dtype: PrimType,
    pub scope: MemoryScope,
    pub shape: Array<PrimExpr>,
    pub data_alignment: i32,
    pub layout: Option<Layout>,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorTypeObj>() == 72);
    assert!(::core::mem::align_of::<TensorTypeObj>() == 8);
};

impl Deref for TensorTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl TensorTypeObj {
    pub(crate) fn new(
        loc: Location,
        dtype: PrimType,
        scope: MemoryScope,
        shape: Array<PrimExpr>,
        data_alignment: i32,
        layout: Option<Layout>,
    ) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_, dtype, scope, shape, data_alignment, layout }
    }
}

#[allow(missing_docs)]
impl TensorType {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        dtype: impl Into<PrimType>,
        scope: impl Into<MemoryScope>,
        shape: impl Into<Array<PrimExpr>>,
        data_alignment: i32,
        layout: impl Into<Option<Layout>>,
    ) -> Self {
        let obj = TensorTypeObj::new(
            loc.into(),
            dtype.into(),
            scope.into(),
            shape.into(),
            data_alignment,
            layout.into(),
        );
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for TensorTypeObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(TensorType => Type);
// tvm-ffi-stubgen(end)

impl TensorType {
    /// Create the type of a tensor of `shape` and `dtype` in `scope`, aligned to `data_alignment`
    /// bytes (the runtime's allocation alignment when 0 or less), with `layout` if given.
    pub fn new(
        shape: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        dtype: impl Into<PrimType>,
        scope: impl Into<MemoryScope>,
        data_alignment: i32,
        layout: Option<&Layout>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let shape = shape
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        let data_alignment = match data_alignment <= 0 {
            true => alloc_alignment()?,
            false => data_alignment,
        };
        Ok(Self::from_fields(
            loc.into(),
            dtype.into(),
            scope.into(),
            shape,
            data_alignment,
            layout.cloned(),
        ))
    }

    /// Return whether the layout is absent or equals the default row-major layout of the shape
    /// (`tirx.TensorTypeHasDefaultLayout`).
    pub fn has_default_layout(&self, analyzer: Option<&Analyzer>) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.TensorTypeHasDefaultLayout")
            .call_tuple((self, analyzer.cloned()))?
            .try_into()
    }
}

/// A signature descriptor ([`sig::ArgType`]) that matches a [`Var`](crate::ir::Var) whose type is
/// a [`TensorType`]; it is not a type of its own.
pub struct TensorVar;

impl sig::ArgType for TensorVar {
    fn check(value: AnyView<'_>) -> bool {
        sig::var_of_type::<TensorTypeObj>(value)
    }

    fn type_str() -> std::string::String {
        crate::ir::expr::VarObj::TYPE_KEY.into()
    }
}

/// The runtime's allocation alignment, which a build may configure, read from the default the
/// reflection registers for `TensorType::data_alignment`.
fn alloc_alignment() -> Result<i32> {
    static ALLOC_ALIGNMENT: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    if let Some(alignment) = ALLOC_ALIGNMENT.get() {
        return Ok(*alignment);
    }
    let Some(default) = tvm_ffi::reflection::field_default::<TensorTypeObj>("data_alignment")
    else {
        let message = "tirx.TensorType registers no default data_alignment";
        return Err(tvm_ffi::Error::new(tvm_ffi::INTERNAL_ERROR, message, ""));
    };
    let alignment = i32::try_from(i64::try_from(default)?).map_err(|_| {
        tvm_ffi::Error::new(tvm_ffi::INTERNAL_ERROR, "the default data_alignment exceeds i32", "")
    })?;
    Ok(*ALLOC_ALIGNMENT.get_or_init(|| alignment))
}
