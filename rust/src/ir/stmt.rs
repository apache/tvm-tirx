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
// The generated constructors take one argument per reflected field.
#![allow(clippy::too_many_arguments)]

//! Statements: bindings, assertions, loops, conditionals, tensor stores and sequences.
//!
//! The objects `python/tvm_tirx/ir/stmt.py` defines. [`Stmt`] is the base of every statement; a
//! body given as one statement is wrapped in a [`SeqStmt`].
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::prim::op::PrimOps;
//! use tvm_tirx::ir::{
//!     AssertStmt, Bind, For, ForKind, PrimType, PrimVar, SeqStmt, Stmt, StringImm, Var,
//! };
//! use tvm_tirx::ir::UNKNOWN_LOC;
//! use tvm_tirx::script::script;
//!
//! let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
//! let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
//! let bind = Bind::new(&x, &i + 1, UNKNOWN_LOC)?;
//! let kind = StringImm::new("ValueError", UNKNOWN_LOC);
//! let check = AssertStmt::new(x.less(17), kind, [StringImm::new("x", UNKNOWN_LOC)], UNKNOWN_LOC)?;
//!
//! let body = SeqStmt::new([Stmt::from(bind), check.into()], UNKNOWN_LOC);
//! let loop_ = For::new(&i, 0, 16, 1, ForKind::Default, body, None, UNKNOWN_LOC)?;
//! let expected = "\
//! for i in range(16):
//!     x: T.let[T.int32] = i + 1
//!     assert x < 17, (\"ValueError\", [\"x\"])";
//! assert_eq!(script(&loop_)?, expected);
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// `custom-new` reserves `new` for a constructor with the C++ one's semantics (defaults, implicit
// conversions, validation); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): ir.Bind
// tvm-ffi-stubgen(custom-new): ir.AssertStmt
// tvm-ffi-stubgen(custom-new): ir.For
// tvm-ffi-stubgen(custom-new): ir.TensorStore
// tvm-ffi-stubgen(custom-new): ir.RegionStmt
// tvm-ffi-stubgen(custom-new): ir.SeqStmt
// tvm-ffi-stubgen(custom-new): ir.If
// tvm-ffi-stubgen(custom-new): ir.Evaluate
// tvm-ffi-stubgen(custom-new): ir.Return
// tvm-ffi-stubgen(custom-new): ir.While
// A loop's kind is C++ `ForKind`. The `enum` directive declares it as an open newtype of the
// native `int` field, so a value Rust does not name reads back intact.
// tvm-ffi-stubgen(enum): ir.For.kind -> ForKind(i32) { Default=0, Parallel=1, Vectorized=2, Unrolled=3 }
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::BigInt
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataType
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex
// tvm-ffi-stubgen(import-object): crate::ir::expr::Call
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImm
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImmObj
// tvm-ffi-stubgen(import-object): crate::ir::location::UNKNOWN_LOC
// tvm-ffi-stubgen(import-object): crate::ir::r#type::MissingType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::MissingTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::Type
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimVar
// Loop bounds, conditions and stored values are primitive, as C++ types them: `ForObj` holds a
// `PrimVar var` and `PrimExpr begin, end, step`.
// tvm-ffi-stubgen(field): ir.AssertStmt.condition -> PrimExpr
// tvm-ffi-stubgen(field): ir.For.var -> PrimVar
// tvm-ffi-stubgen(field): ir.For.begin -> PrimExpr
// tvm-ffi-stubgen(field): ir.For.end -> PrimExpr
// tvm-ffi-stubgen(field): ir.For.step -> PrimExpr
// tvm-ffi-stubgen(field): ir.While.condition -> PrimExpr
// tvm-ffi-stubgen(field): ir.TensorStore.indices -> Array<PrimExpr>
// tvm-ffi-stubgen(field): ir.TensorStore.value -> PrimExpr
// tvm-ffi-stubgen(field): ir.If.condition -> PrimExpr

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::Call;
use crate::ir::expr::Expr;
use crate::ir::expr::IntImm;
use crate::ir::expr::IntImmObj;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::PrimVar;
use crate::ir::expr::StringImm;
use crate::ir::expr::Var;
use crate::ir::location::Location;
use crate::ir::location::UNKNOWN_LOC;
use crate::ir::op::Op;
use crate::ir::r#type::MissingType;
use crate::ir::r#type::MissingTypeObj;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::Type;
use crate::ir::support;
use std::ops::Deref;
use std::ptr::addr_of_mut;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::BigInt;
use tvm_ffi::DLDataType;
use tvm_ffi::Error;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::ShallowCopy;
use tvm_ffi::String;
use tvm_ffi::UnsafeInit;
use tvm_ffi::VALUE_ERROR;
use tvm_ffi::reflection::drop_if_defined;
use tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Stmt
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Stmt"]
pub struct StmtObj {
    base: Object,
    pub loc: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<StmtObj>() == 32);
    assert!(::core::mem::align_of::<StmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Stmt {
    base: ObjectArc<StmtObj>,
}

impl Deref for Stmt {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl StmtObj {
    pub fn new(loc: Location) -> Self {
        let base = Object::new();
        Self { base, loc }
    }
}

impl Stmt {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = StmtObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for StmtObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base), loc: Clone::clone(&self.loc) }
    }
}

unsafe impl UnsafeInit for StmtObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base));
            drop_if_defined(addr_of_mut!((*this).loc));
        }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for StmtObj {}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Bind
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Bind"]
#[type_final]
pub struct BindObj {
    base: StmtObj,
    pub var: Var,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BindObj>() == 48);
    assert!(::core::mem::align_of::<BindObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Bind {
    base: ObjectArc<BindObj>,
}

impl Deref for Bind {
    type Target = BindObj;
    fn deref(&self) -> &BindObj {
        &self.base
    }
}

impl Deref for BindObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl BindObj {
    pub(crate) fn new(loc: Location, var: Var, value: Expr) -> Self {
        let base = StmtObj::new(loc);
        Self { base, var, value }
    }
}

impl Bind {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        var: impl Into<Var>,
        value: impl Into<Expr>,
    ) -> Self {
        let obj = BindObj::new(loc.into(), var.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BindObj {}

tvm_ffi::unsafe_impl_object_upcast!(Bind => Stmt);
// tvm-ffi-stubgen(end)

impl Bind {
    /// Create the statement binding `var` to `value`, of the same type, for the rest of the
    /// enclosing scope (C++ `Bind(var, value, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if the type of `value` is not structurally equal to the type of
    /// `var`.
    pub fn new(
        var: impl Into<Var>,
        value: impl IntoExpr,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (var, value) = (var.into(), value.into_owned_expr()?);
        if !support::structural_equal(&value.ty, &var.ty)? {
            let condition = "ffi::StructuralEqual()(value->ty, var->ty)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        Ok(Self::from_fields(loc.into(), var, value))
    }
}

// tvm-ffi-stubgen(begin): object/ir.AssertStmt
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.AssertStmt"]
#[type_final]
pub struct AssertStmtObj {
    base: StmtObj,
    pub condition: PrimExpr,
    pub error_kind: StringImm,
    pub message_parts: Array<StringImm>,
}

const _: () = {
    assert!(::core::mem::size_of::<AssertStmtObj>() == 56);
    assert!(::core::mem::align_of::<AssertStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct AssertStmt {
    base: ObjectArc<AssertStmtObj>,
}

impl Deref for AssertStmt {
    type Target = AssertStmtObj;
    fn deref(&self) -> &AssertStmtObj {
        &self.base
    }
}

impl Deref for AssertStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl AssertStmtObj {
    pub(crate) fn new(
        loc: Location,
        condition: PrimExpr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, condition, error_kind, message_parts }
    }
}

impl AssertStmt {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        condition: impl Into<PrimExpr>,
        error_kind: impl Into<StringImm>,
        message_parts: impl Into<Array<StringImm>>,
    ) -> Self {
        let obj = AssertStmtObj::new(
            loc.into(),
            condition.into(),
            error_kind.into(),
            message_parts.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for AssertStmtObj {}

tvm_ffi::unsafe_impl_object_upcast!(AssertStmt => Stmt);
// tvm-ffi-stubgen(end)

impl AssertStmt {
    /// Create the statement asserting the boolean `condition`, raising an `error_kind` error with
    /// the message `message_parts` when it fails (C++ `AssertStmt`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `condition` is not boolean.
    pub fn new(
        condition: impl IntoExpr<PrimExpr>,
        error_kind: impl Into<StringImm>,
        message_parts: impl IntoIterator<Item = impl Into<StringImm>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (condition, error_kind) = (condition.into_owned_expr()?, error_kind.into());
        let message_parts = message_parts.into_iter().map(Into::into).collect::<Array<StringImm>>();
        let condition_ty = condition.ty();
        if condition_ty.dtype.code != support::BOOL {
            let message = format!(
                "AssertStmt should have boolean condition, but received {} with dtype {}",
                support::repr(&condition),
                support::repr(&condition_ty)
            );
            let check = "condition_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, check, &message));
        }
        Ok(Self::from_fields(loc.into(), condition, error_kind, message_parts))
    }
}

// tvm-ffi-stubgen(begin): object/ir.For
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ForKind(i32);

#[allow(non_upper_case_globals)]
impl ForKind {
    pub const Default: Self = Self(0);
    pub const Parallel: Self = Self(1);
    pub const Vectorized: Self = Self(2);
    pub const Unrolled: Self = Self(3);
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl TryFrom<i64> for ForKind {
    type Error = Error;
    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(VALUE_ERROR, &format!("ForKind value {value} does not fit i32"), "")
        })
    }
}

/// Complete: reflected fields fill [32, 88) exactly (alignment padding [68, 72)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.For"]
#[type_final]
pub struct ForObj {
    base: StmtObj,
    pub var: PrimVar,
    pub begin: PrimExpr,
    pub end: PrimExpr,
    pub step: PrimExpr,
    pub kind: ForKind,
    pub body: SeqStmt,
    pub annotations: Map<String, Any>,
}

const _: () = {
    assert!(::core::mem::size_of::<ForObj>() == 88);
    assert!(::core::mem::align_of::<ForObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct For {
    base: ObjectArc<ForObj>,
}

impl Deref for For {
    type Target = ForObj;
    fn deref(&self) -> &ForObj {
        &self.base
    }
}

impl Deref for ForObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ForObj {
    pub(crate) fn new(
        loc: Location,
        var: PrimVar,
        begin: PrimExpr,
        end: PrimExpr,
        step: PrimExpr,
        kind: ForKind,
        body: SeqStmt,
        annotations: Map<String, Any>,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, var, begin, end, step, kind, body, annotations }
    }
}

impl For {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        var: impl Into<PrimVar>,
        begin: impl Into<PrimExpr>,
        end: impl Into<PrimExpr>,
        step: impl Into<PrimExpr>,
        kind: impl Into<ForKind>,
        body: impl Into<SeqStmt>,
        annotations: impl Into<Map<String, Any>>,
    ) -> Self {
        let obj = ForObj::new(
            loc.into(),
            var.into(),
            begin.into(),
            end.into(),
            step.into(),
            kind.into(),
            body.into(),
            annotations.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ForObj {}

tvm_ffi::unsafe_impl_object_upcast!(For => Stmt);
// tvm-ffi-stubgen(end)

impl Default for ForKind {
    /// The ordinary loop, of sequential iterations (C++ `ForKind::kDefault`).
    fn default() -> Self {
        Self::Default
    }
}

impl For {
    /// Create the loop of the scalar integer `var` from `begin` to `end` by `step`, running `body`
    /// with the `kind` semantics (C++ `For(var, begin, end, step, kind, body, annotations, loc)`).
    ///
    /// A constant bound or step of another integer type is converted to the type of `var`.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `var`, a bound or the step is not a scalar integer, or a bound
    /// or the step does not fit or match the type of `var`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        var: impl Into<PrimVar>,
        begin: impl IntoExpr<PrimExpr>,
        end: impl IntoExpr<PrimExpr>,
        step: impl IntoExpr<PrimExpr>,
        kind: ForKind,
        body: impl Into<Stmt>,
        annotations: Option<&Map<String, Any>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (var, begin, end, step) =
            (var.into(), begin.into_owned_expr()?, end.into_owned_expr()?, step.into_owned_expr()?);
        let var_expr: PrimExpr = var.clone().into();
        let var_ty = var.ty();
        let body = into_body(body.into())?;
        let annotations = annotations.cloned().unwrap_or_else(Map::new);
        for (expr, field) in [(&var_expr, "var"), (&begin, "begin"), (&end, "end"), (&step, "step")]
        {
            let ty = expr.ty();
            let dtype = ty.dtype;
            if !support::is_scalar(dtype) || !matches!(dtype.code, support::INT | support::UINT) {
                let message = format!(
                    "TIR For nodes require a scalar integer as the {field}, but received {} with \
                     dtype {}",
                    support::repr(expr),
                    support::repr(&ty)
                );
                let condition = "dtype.IsScalar() && (dtype.MatchesCode(DLDataTypeCode::kDLUInt, \
                                 DLDataTypeCode::kDLInt))";
                return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, &message));
            }
        }
        // A constant of another integer type takes the type of `var` if its value fits.
        let promote = |expr: PrimExpr| -> Result<PrimExpr> {
            let ty = expr.ty();
            if ty.dtype == var_ty.dtype {
                return Ok(expr);
            }
            if let Some(imm) = expr.downcast_ref::<IntImmObj>() {
                if !int_imm_value_fits(&imm.value, var_ty.dtype)? {
                    let message = format!(
                        "Literal value {} is not representable in the loop variable's dtype ({})",
                        imm.value,
                        support::repr(&var_ty)
                    );
                    let condition = "IntImmValueFits(a->value, var_ty)";
                    return Err(support::check_failed(
                        tvm_ffi::INTERNAL_ERROR,
                        condition,
                        &message,
                    ));
                }
                return Ok(IntImm::new(var_ty.clone(), imm.value.clone(), UNKNOWN_LOC)?.into());
            }
            if ty.dtype.bits > var_ty.dtype.bits {
                let message = format!(
                    " Loop variable's dtype ({}) is narrower than that of `begin` or `end` ({})",
                    support::repr(&var_ty),
                    support::repr(&ty)
                );
                let condition = "e_ty.bits() <= var_ty.bits()";
                return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, &message));
            }
            Ok(expr)
        };
        let same_type = |expr: &PrimExpr, condition: &str| -> Result<()> {
            let ty = expr.ty();
            if ty.dtype == var_ty.dtype {
                return Ok(());
            }
            let message = format!("{} vs {}", support::repr(&var_ty), support::repr(&ty));
            Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, &message))
        };
        let (begin, end) = (promote(begin)?, promote(end)?);
        same_type(&begin, "var.ty() == begin.ty()")?;
        same_type(&end, "var.ty() == end.ty()")?;
        let step = promote(step)?;
        same_type(&step, "var.ty() == step.ty()")?;
        Ok(Self::from_fields(loc.into(), var, begin, end, step, kind, body, annotations))
    }
}

/// Whether `value` is in the range of the integer type `dtype` (C++ `IntImmValueFits`).
fn int_imm_value_fits(value: &BigInt, dtype: DLDataType) -> Result<bool> {
    let bits = dtype.bits as u32;
    if dtype.code == support::UINT {
        let limit = BigInt::from(1i64).try_shl(bits as i64)?;
        return Ok(!value.is_negative() && *value < limit);
    }
    let limit = BigInt::from(1i64).try_shl(bits as i64 - 1)?;
    Ok(*value >= -limit.clone() && *value < limit)
}

// tvm-ffi-stubgen(begin): object/ir.While
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.While"]
#[type_final]
pub struct WhileObj {
    base: StmtObj,
    pub condition: PrimExpr,
    pub body: SeqStmt,
}

const _: () = {
    assert!(::core::mem::size_of::<WhileObj>() == 48);
    assert!(::core::mem::align_of::<WhileObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct While {
    base: ObjectArc<WhileObj>,
}

impl Deref for While {
    type Target = WhileObj;
    fn deref(&self) -> &WhileObj {
        &self.base
    }
}

impl Deref for WhileObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl WhileObj {
    pub(crate) fn new(loc: Location, condition: PrimExpr, body: SeqStmt) -> Self {
        let base = StmtObj::new(loc);
        Self { base, condition, body }
    }
}

impl While {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        condition: impl Into<PrimExpr>,
        body: impl Into<SeqStmt>,
    ) -> Self {
        let obj = WhileObj::new(loc.into(), condition.into(), body.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for WhileObj {}

tvm_ffi::unsafe_impl_object_upcast!(While => Stmt);
// tvm-ffi-stubgen(end)

impl While {
    /// Create the loop running `body` while the scalar `condition` holds, checked to be well formed
    /// (C++ `While(condition, body, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `condition` is a vector.
    pub fn new(
        condition: impl IntoExpr<PrimExpr>,
        body: impl Into<Stmt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let condition = condition.into_owned_expr()?;
        let condition_dtype = condition.dtype();
        let body = into_body(body.into())?;
        if !support::is_scalar(condition_dtype) {
            let check = "condition.ty().IsScalar()";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, check, ""));
        }
        Ok(Self::from_fields(loc, condition, body))
    }
}

// tvm-ffi-stubgen(begin): object/ir.TensorStore
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorStore"]
#[type_final]
pub struct TensorStoreObj {
    base: StmtObj,
    pub dest: Expr,
    pub indices: Array<PrimExpr>,
    pub value: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorStoreObj>() == 56);
    assert!(::core::mem::align_of::<TensorStoreObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorStore {
    base: ObjectArc<TensorStoreObj>,
}

impl Deref for TensorStore {
    type Target = TensorStoreObj;
    fn deref(&self) -> &TensorStoreObj {
        &self.base
    }
}

impl Deref for TensorStoreObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl TensorStoreObj {
    pub(crate) fn new(
        loc: Location,
        dest: Expr,
        indices: Array<PrimExpr>,
        value: PrimExpr,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, dest, indices, value }
    }
}

impl TensorStore {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        dest: impl Into<Expr>,
        indices: impl Into<Array<PrimExpr>>,
        value: impl Into<PrimExpr>,
    ) -> Self {
        let obj = TensorStoreObj::new(loc.into(), dest.into(), indices.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for TensorStoreObj {}

tvm_ffi::unsafe_impl_object_upcast!(TensorStore => Stmt);
// tvm-ffi-stubgen(end)

impl TensorStore {
    /// Create the store of `value` into the element of `dest` at `indices`, checked by the
    /// validation the type of `dest` registers (C++ `TensorStore(dest, indices, value, loc)`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the type of `dest` does not support `TensorStore`, or the error its
    /// validation raises.
    pub fn new(
        dest: impl IntoExpr,
        indices: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        value: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        static VALIDATE: support::TypeAttr =
            support::TypeAttr::new("__tvm_tensor_store_validate__");
        let (dest, value) = (dest.into_owned_expr()?, value.into_owned_expr()?);
        let indices = indices
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        let ty_index = AnyView::from(&dest.ty).type_index();
        let Some(validate) = VALIDATE.get(ty_index) else {
            let message =
                format!("Type {} does not support TensorStore", support::type_key(ty_index));
            return Err(support::check_failed(
                tvm_ffi::TYPE_ERROR,
                "validate.has_value()",
                &message,
            ));
        };
        let result = support::call_with_refs(validate, &dest, &indices, &value)?;
        if result.type_index() != TVMFFITypeIndex::kTVMFFINone as i32 {
            let message = "Packed hook result type mismatch";
            return Err(tvm_ffi::Error::new(tvm_ffi::TYPE_ERROR, message, ""));
        }
        Ok(Self::from_fields(loc.into(), dest, indices, value))
    }
}

// tvm-ffi-stubgen(begin): object/ir.RegionStmt
/// Complete: reflected fields fill [32, 80) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.RegionStmt"]
#[type_final]
pub struct RegionStmtObj {
    base: StmtObj,
    pub op: Op,
    pub args: Array<Expr>,
    pub body_params: Array<Var>,
    pub attrs: DictAttrs,
    pub body: SeqStmt,
    pub result_vars: Array<Var>,
}

const _: () = {
    assert!(::core::mem::size_of::<RegionStmtObj>() == 80);
    assert!(::core::mem::align_of::<RegionStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RegionStmt {
    base: ObjectArc<RegionStmtObj>,
}

impl Deref for RegionStmt {
    type Target = RegionStmtObj;
    fn deref(&self) -> &RegionStmtObj {
        &self.base
    }
}

impl Deref for RegionStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl RegionStmtObj {
    pub(crate) fn new(
        loc: Location,
        op: Op,
        args: Array<Expr>,
        body_params: Array<Var>,
        attrs: DictAttrs,
        body: SeqStmt,
        result_vars: Array<Var>,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, op, args, body_params, attrs, body, result_vars }
    }
}

impl RegionStmt {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        op: impl Into<Op>,
        args: impl Into<Array<Expr>>,
        body_params: impl Into<Array<Var>>,
        attrs: impl Into<DictAttrs>,
        body: impl Into<SeqStmt>,
        result_vars: impl Into<Array<Var>>,
    ) -> Self {
        let obj = RegionStmtObj::new(
            loc.into(),
            op.into(),
            args.into(),
            body_params.into(),
            attrs.into(),
            body.into(),
            result_vars.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for RegionStmtObj {}

tvm_ffi::unsafe_impl_object_upcast!(RegionStmt => Stmt);
// tvm-ffi-stubgen(end)

impl RegionStmt {
    /// Create the region of the region operator `op` on `args`, whose `body` binds `body_params`
    /// and defines `result_vars`, checked against the contract of the operator (C++ `RegionStmt`).
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if `body_params` do not match the parameters the operator gives or a
    /// variable is defined twice, or the error of the operator's validators.
    pub fn new(
        op: impl Into<Op>,
        args: impl IntoIterator<Item = impl IntoExpr>,
        body_params: impl IntoIterator<Item = impl Into<Var>>,
        attrs: impl Into<DictAttrs>,
        body: impl Into<Stmt>,
        result_vars: impl IntoIterator<Item = impl Into<Var>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (op, attrs, body) = (op.into(), attrs.into(), into_body(body.into())?);
        let args =
            args.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<Expr>>>()?;
        let body_params = body_params.into_iter().map(Into::into).collect::<Array<Var>>();
        let result_vars = result_vars.into_iter().map(Into::into).collect::<Array<Var>>();
        let name = op.name.as_str().to_string();
        let expected = region_body_params(&op, &args, &attrs)?;
        if body_params.len() != expected.len() {
            let condition = "body_params.size() == expected_params.size()";
            let message = format!("{name} expects {} body parameters", expected.len());
            let (given, wanted) = (body_params.len(), expected.len());
            return Err(support::compare_failed(
                tvm_ffi::VALUE_ERROR,
                condition,
                given,
                wanted,
                &message,
            ));
        }
        for (i, (param, expected)) in body_params.iter().zip(expected.iter()).enumerate() {
            if !support::structural_equal(&param.ty, &expected.ty)? {
                let condition =
                    "ffi::StructuralEqual()(body_params[i]->ty, expected_params[i]->ty)";
                let message =
                    format!("{name} body parameter {i} has a type inconsistent with its contract");
                return Err(support::check_failed(tvm_ffi::VALUE_ERROR, condition, &message));
            }
        }
        let mut definitions = std::collections::HashSet::new();
        for var in body_params.iter().chain(result_vars.iter()) {
            if !definitions.insert(object_address(var)) {
                let condition = "definitions.insert(var.get()).second";
                let message = "RegionStmt parameters and results must be distinct definitions";
                return Err(support::check_failed(tvm_ffi::VALUE_ERROR, condition, message));
            }
        }
        let region = Self::from_fields(loc.into(), op, args, body_params, attrs, body, result_vars);
        if let Some(validate) = support::NativeHook::of_op(&region.op, "FRegionValidate")? {
            validate.call_void(&region)?;
        }
        Ok(region)
    }
}

/// The parameters the body of a region of `op` on `args` binds, which the operator's
/// `FRegionGetBodyParams` gives once its validator accepts the arguments
/// (C++ `GetRegionBodyParams`).
fn region_body_params(op: &Op, args: &Array<Expr>, attrs: &DictAttrs) -> Result<Array<Var>> {
    let name = op.name.as_str();
    let Some(get_body_params) = support::NativeHook::of_op(op, "FRegionGetBodyParams")? else {
        let message = format!(
            "{name} does not support region construction: FRegionGetBodyParams is required"
        );
        return Err(support::check_failed(tvm_ffi::VALUE_ERROR, "IsRegionOp(op)", &message));
    };
    let signature = Call::from_fields(
        UNKNOWN_LOC,
        MissingType::new(),
        op.clone(),
        args.clone(),
        Some(attrs.clone().into()),
        Array::from_vec(vec![]),
    );
    signature.validate()?;
    let params: Array<Var> = get_body_params.call_as(&signature)?;
    let mut definitions = std::collections::HashSet::new();
    for param in params.iter() {
        if param.ty.downcast_ref::<MissingTypeObj>().is_some() {
            let condition = "!param->ty.as<MissingType>().has_value()";
            let message = format!("FRegionGetBodyParams for {name} must return typed variables");
            return Err(support::check_failed(tvm_ffi::VALUE_ERROR, condition, &message));
        }
        if !definitions.insert(object_address(param)) {
            let condition = "definitions.insert(param.get()).second";
            let message =
                format!("FRegionGetBodyParams for {name} must return distinct definitions");
            return Err(support::check_failed(tvm_ffi::VALUE_ERROR, condition, &message));
        }
    }
    Ok(params)
}

/// The address of the object `value` refers to, which identifies a definition.
fn object_address<T: ObjectRefCore>(value: &T) -> usize {
    // The pointer is only compared, never dereferenced.
    ObjectArc::as_ptr(ObjectRefCore::data(value)) as usize
}

// tvm-ffi-stubgen(begin): object/ir.SeqStmt
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base: StmtObj,
    pub seq: Array<Stmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<SeqStmtObj>() == 40);
    assert!(::core::mem::align_of::<SeqStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SeqStmt {
    base: ObjectArc<SeqStmtObj>,
}

impl Deref for SeqStmt {
    type Target = SeqStmtObj;
    fn deref(&self) -> &SeqStmtObj {
        &self.base
    }
}

impl Deref for SeqStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl SeqStmtObj {
    pub(crate) fn new(loc: Location, seq: Array<Stmt>) -> Self {
        let base = StmtObj::new(loc);
        Self { base, seq }
    }
}

impl SeqStmt {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, seq: impl Into<Array<Stmt>>) -> Self {
        let obj = SeqStmtObj::new(loc.into(), seq.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for SeqStmtObj {}

tvm_ffi::unsafe_impl_object_upcast!(SeqStmt => Stmt);
// tvm-ffi-stubgen(end)

impl SeqStmt {
    /// Create the sequence of `seq`, with nested sequences flattened and no-ops (`Evaluate(0)`)
    /// dropped (C++ `SeqStmt(seq, loc)`).
    pub fn new(seq: impl IntoIterator<Item = impl Into<Stmt>>, loc: impl Into<Location>) -> Self {
        let seq = seq.into_iter().map(Into::into).collect::<Array<Stmt>>();
        Self::from_fields(loc.into(), flatten(seq))
    }
}

/// `seq` with each nested sequence replaced by its statements and the no-ops removed.
fn flatten(seq: Array<Stmt>) -> Array<Stmt> {
    let nested_or_no_op =
        |stmt: &Stmt| stmt.downcast_ref::<SeqStmtObj>().is_some() || is_no_op(stmt);
    if !seq.iter().any(nested_or_no_op) {
        return seq;
    }
    fn append(stmt: Stmt, flattened: &mut Vec<Stmt>) {
        match stmt.downcast_ref::<SeqStmtObj>() {
            Some(nested) => nested.seq.iter().for_each(|child| append(child.clone(), flattened)),
            None if is_no_op(&stmt) => {}
            None => flattened.push(stmt),
        }
    }
    let mut flattened = Vec::new();
    seq.iter().for_each(|stmt| append(stmt.clone(), &mut flattened));
    Array::from_vec(flattened)
}

/// Whether `stmt` is `Evaluate(0)`, the no-op a sequence drops (C++ `IsSeqStmtNoOp`).
fn is_no_op(stmt: &Stmt) -> bool {
    stmt.downcast_ref::<EvaluateObj>()
        .and_then(|evaluate| evaluate.value.downcast_ref::<IntImmObj>())
        .is_some_and(|imm| imm.value == BigInt::from(0i64))
}

/// A statement as a body, as C++ converts a `Stmt` to a `SeqStmt`: a sequence as it is, any other
/// statement as the sequence of it at its location (C++ `SeqStmt(stmt)`).
fn into_body(stmt: Stmt) -> Result<SeqStmt> {
    if stmt.downcast_ref::<SeqStmtObj>().is_some() {
        return Ok(stmt.downcast().expect("a SeqStmt"));
    }
    let loc = stmt.loc.clone();
    Ok(SeqStmt::from_fields(loc, flatten(Array::from_vec(vec![stmt]))))
}

// tvm-ffi-stubgen(begin): object/ir.If
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.If"]
#[type_final]
pub struct IfObj {
    base: StmtObj,
    pub condition: PrimExpr,
    pub then_case: SeqStmt,
    pub else_case: Option<SeqStmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<IfObj>() == 56);
    assert!(::core::mem::align_of::<IfObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct If {
    base: ObjectArc<IfObj>,
}

impl Deref for If {
    type Target = IfObj;
    fn deref(&self) -> &IfObj {
        &self.base
    }
}

impl Deref for IfObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl IfObj {
    pub(crate) fn new(
        loc: Location,
        condition: PrimExpr,
        then_case: SeqStmt,
        else_case: Option<SeqStmt>,
    ) -> Self {
        let base = StmtObj::new(loc);
        Self { base, condition, then_case, else_case }
    }
}

impl If {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        condition: impl Into<PrimExpr>,
        then_case: impl Into<SeqStmt>,
        else_case: impl Into<Option<SeqStmt>>,
    ) -> Self {
        let obj = IfObj::new(loc.into(), condition.into(), then_case.into(), else_case.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for IfObj {}

tvm_ffi::unsafe_impl_object_upcast!(If => Stmt);
// tvm-ffi-stubgen(end)

impl If {
    /// Create the statement running `then_case` if `condition` holds, else `else_case` if given
    /// (C++ `If(condition, then_case, else_case, loc)`).
    pub fn new(
        condition: impl IntoExpr<PrimExpr>,
        then_case: impl Into<Stmt>,
        else_case: Option<&Stmt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let condition = condition.into_owned_expr()?;
        let then_case = into_body(then_case.into())?;
        let else_case = else_case.map(|stmt| into_body(stmt.clone())).transpose()?;
        Ok(Self::from_fields(loc.into(), condition, then_case, else_case))
    }
}

// tvm-ffi-stubgen(begin): object/ir.Evaluate
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base: StmtObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<EvaluateObj>() == 40);
    assert!(::core::mem::align_of::<EvaluateObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Evaluate {
    base: ObjectArc<EvaluateObj>,
}

impl Deref for Evaluate {
    type Target = EvaluateObj;
    fn deref(&self) -> &EvaluateObj {
        &self.base
    }
}

impl Deref for EvaluateObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl EvaluateObj {
    pub(crate) fn new(loc: Location, value: Expr) -> Self {
        let base = StmtObj::new(loc);
        Self { base, value }
    }
}

impl Evaluate {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, value: impl Into<Expr>) -> Self {
        let obj = EvaluateObj::new(loc.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for EvaluateObj {}

tvm_ffi::unsafe_impl_object_upcast!(Evaluate => Stmt);
// tvm-ffi-stubgen(end)

impl Evaluate {
    /// Create the statement evaluating `value` for its side effects (C++ `Evaluate(value, loc)`).
    pub fn new(value: impl Into<Expr>, loc: impl Into<Location>) -> Self {
        Self::from_fields(loc.into(), value.into())
    }
}

// tvm-ffi-stubgen(begin): object/ir.Break
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Break"]
#[type_final]
pub struct BreakObj {
    base: StmtObj,
}

const _: () = {
    assert!(::core::mem::size_of::<BreakObj>() == 32);
    assert!(::core::mem::align_of::<BreakObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Break {
    base: ObjectArc<BreakObj>,
}

impl Deref for Break {
    type Target = BreakObj;
    fn deref(&self) -> &BreakObj {
        &self.base
    }
}

impl Deref for BreakObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl BreakObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = StmtObj::new(loc);
        Self { base }
    }
}

impl Break {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = BreakObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BreakObj {}

tvm_ffi::unsafe_impl_object_upcast!(Break => Stmt);
// tvm-ffi-stubgen(end)

// `ir.Break` stores its location as given: the generated `Break::new` is the same.

// tvm-ffi-stubgen(begin): object/ir.Return
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Return"]
#[type_final]
pub struct ReturnObj {
    base: StmtObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<ReturnObj>() == 40);
    assert!(::core::mem::align_of::<ReturnObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Return {
    base: ObjectArc<ReturnObj>,
}

impl Deref for Return {
    type Target = ReturnObj;
    fn deref(&self) -> &ReturnObj {
        &self.base
    }
}

impl Deref for ReturnObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ReturnObj {
    pub(crate) fn new(loc: Location, value: Expr) -> Self {
        let base = StmtObj::new(loc);
        Self { base, value }
    }
}

impl Return {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, value: impl Into<Expr>) -> Self {
        let obj = ReturnObj::new(loc.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ReturnObj {}

tvm_ffi::unsafe_impl_object_upcast!(Return => Stmt);
// tvm-ffi-stubgen(end)

impl Return {
    /// Create the statement returning `value` from the enclosing function (C++ `Return`).
    pub fn new(value: impl Into<Expr>, loc: impl Into<Location>) -> Self {
        Self::from_fields(loc.into(), value.into())
    }
}

// tvm-ffi-stubgen(begin): object/ir.Continue
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Continue"]
#[type_final]
pub struct ContinueObj {
    base: StmtObj,
}

const _: () = {
    assert!(::core::mem::size_of::<ContinueObj>() == 32);
    assert!(::core::mem::align_of::<ContinueObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Continue {
    base: ObjectArc<ContinueObj>,
}

impl Deref for Continue {
    type Target = ContinueObj;
    fn deref(&self) -> &ContinueObj {
        &self.base
    }
}

impl Deref for ContinueObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ContinueObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = StmtObj::new(loc);
        Self { base }
    }
}

impl Continue {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = ContinueObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ContinueObj {}

tvm_ffi::unsafe_impl_object_upcast!(Continue => Stmt);
// tvm-ffi-stubgen(end)

// `ir.Continue` stores its location as given: the generated `Continue::new` is the same.
