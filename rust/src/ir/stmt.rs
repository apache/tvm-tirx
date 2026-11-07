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
// The generated constructors take one argument per reflected field.
#![allow(clippy::too_many_arguments)]

//! Statements. Every statement is a [`Stmt`]; the body of a loop, conditional or region is a
//! [`SeqStmt`], so a single statement given as a body is wrapped in one, and a [`Bind`] binds a
//! variable for the statements after it in the enclosing body.

// tvm-ffi-stubgen(missing-docs): allow
// `custom-new` reserves `new` for a hand-written constructor (defaults, implicit conversions,
// validation); the generated `from_fields` allocates the fields as given.
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
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataTypeCode
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
use crate::ir::support;
use std::ops::Deref;
use std::ptr::addr_of_mut;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::BigInt;
use tvm_ffi::DLDataTypeCode;
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

/// The base of every statement: its source location `loc`.
///
/// Each kind of statement is a subtype that adds its operands; it converts into a `Stmt` for
/// free, and a `Stmt` is checked back into one with `downcast_ref` or `downcast`.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::op::PrimOps;
/// use tvm_tirx::ir::{
///     AssertStmt, Bind, For, ForKind, PrimType, PrimVar, SeqStmt, Stmt, StringImm,
/// };
/// use tvm_tirx::ir::UNKNOWN_LOC;
/// use tvm_tirx::script::script;
///
/// let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// let bind = Bind::new(&x, &i + 1, UNKNOWN_LOC)?;
/// let kind = StringImm::new("ValueError", UNKNOWN_LOC);
/// let check = AssertStmt::new(x.less(17), kind, [StringImm::new("x", UNKNOWN_LOC)], UNKNOWN_LOC)?;
///
/// let body = SeqStmt::new([Stmt::from(bind), check.into()], UNKNOWN_LOC);
/// let loop_ = For::new(&i, 0, 16, 1, ForKind::Default, body, None, UNKNOWN_LOC)?;
/// let expected = "\
/// for i in range(16):
///     x: T.let[T.int32] = i + 1
///     assert x < 17, (\"ValueError\", [\"x\"])";
/// assert_eq!(script(&loop_)?, expected);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/ir.Stmt
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Stmt {
    data_: ObjectArc<StmtObj>,
}

/// The object behind [`Stmt`].
///
/// Complete: reflected fields fill [24, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Stmt"]
pub struct StmtObj {
    base_: Object,
    pub loc: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<StmtObj>() == 32);
    assert!(::core::mem::align_of::<StmtObj>() == 8);
};

#[allow(missing_docs)]
impl StmtObj {
    pub fn new(loc: Location) -> Self {
        let base_ = Object::new();
        Self { base_, loc }
    }
}

#[allow(missing_docs)]
impl Stmt {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = StmtObj::new(loc.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for StmtObj {
    fn shallow_copy(&self) -> Self {
        Self { base_: ShallowCopy::shallow_copy(&self.base_), loc: Clone::clone(&self.loc) }
    }
}

unsafe impl UnsafeInit for StmtObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base_));
            drop_if_defined(addr_of_mut!((*this).loc));
        }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for StmtObj {}
// tvm-ffi-stubgen(end)

/// The binding of the variable `var` to `value`, visible to the statements after it in the
/// enclosing body. It has no body of its own.
// tvm-ffi-stubgen(begin): object/ir.Bind
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Bind {
    data_: ObjectArc<BindObj>,
}

/// The object behind [`Bind`].
///
/// Complete: reflected fields fill [32, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Bind"]
#[type_final]
pub struct BindObj {
    base_: StmtObj,
    pub var: Var,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BindObj>() == 48);
    assert!(::core::mem::align_of::<BindObj>() == 8);
};

impl Deref for BindObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl BindObj {
    pub(crate) fn new(loc: Location, var: Var, value: Expr) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_, var, value }
    }
}

#[allow(missing_docs)]
impl Bind {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        var: impl Into<Var>,
        value: impl Into<Expr>,
    ) -> Self {
        let obj = BindObj::new(loc.into(), var.into(), value.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BindObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Bind => Stmt);
// tvm-ffi-stubgen(end)

impl Bind {
    /// Create the statement binding `var` to `value`, of the same type, for the statements
    /// after it in the enclosing body.
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
        if !tvm_ffi::structural_equal(&value.ty, &var.ty)? {
            let condition = "ffi::StructuralEqual()(value->ty, var->ty)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        Ok(Self::from_fields(loc.into(), var, value))
    }
}

/// The check that the boolean `condition` holds, raising an error of kind `error_kind`, such as
/// `"ValueError"`, whose message concatenates `message_parts` when it does not.
// tvm-ffi-stubgen(begin): object/ir.AssertStmt
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct AssertStmt {
    data_: ObjectArc<AssertStmtObj>,
}

/// The object behind [`AssertStmt`].
///
/// Complete: reflected fields fill [32, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.AssertStmt"]
#[type_final]
pub struct AssertStmtObj {
    base_: StmtObj,
    pub condition: PrimExpr,
    pub error_kind: StringImm,
    pub message_parts: Array<StringImm>,
}

const _: () = {
    assert!(::core::mem::size_of::<AssertStmtObj>() == 56);
    assert!(::core::mem::align_of::<AssertStmtObj>() == 8);
};

impl Deref for AssertStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl AssertStmtObj {
    pub(crate) fn new(
        loc: Location,
        condition: PrimExpr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_, condition, error_kind, message_parts }
    }
}

#[allow(missing_docs)]
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
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for AssertStmtObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(AssertStmt => Stmt);
// tvm-ffi-stubgen(end)

impl AssertStmt {
    /// Create the statement asserting the boolean `condition`, raising an `error_kind` error with
    /// the message `message_parts` when it fails.
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
        let condition_ty = condition.prim_ty();
        if !condition_ty.matches_code(&[DLDataTypeCode::kDLBool]) {
            let message = format!(
                "AssertStmt should have boolean condition, but received {} with dtype {}",
                support::repr(&condition),
                support::repr(condition_ty)
            );
            let check = "condition_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, check, &message));
        }
        Ok(Self::from_fields(loc.into(), condition, error_kind, message_parts))
    }
}

/// The loop of `var` from `begin` while below `end`, incremented by `step`, running `body`.
///
/// `kind` gives the execution semantics of the iterations, sequential, parallel, vectorized or
/// unrolled, which transformations preserve; `annotations` hold properties of the loop that the
/// consuming dialect interprets.
// tvm-ffi-stubgen(begin): object/ir.For
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct For {
    data_: ObjectArc<ForObj>,
}

#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ForKind(i32);

#[allow(missing_docs)]
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

/// The object behind [`For`].
///
/// Complete: reflected fields fill [32, 88) exactly (alignment padding [68, 72)).
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.For"]
#[type_final]
pub struct ForObj {
    base_: StmtObj,
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

impl Deref for ForObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
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
        let base_ = StmtObj::new(loc);
        Self { base_, var, begin, end, step, kind, body, annotations }
    }
}

#[allow(missing_docs)]
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
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ForObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(For => Stmt);
// tvm-ffi-stubgen(end)

impl Default for ForKind {
    /// The ordinary loop, of sequential iterations.
    fn default() -> Self {
        Self::Default
    }
}

impl For {
    /// Create the loop of the scalar integer `var` from `begin` to `end` by `step`, running `body`
    /// with the `kind` semantics and the `annotations` (none when `None`).
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
        let var_ty = var.prim_ty().clone();
        let body = into_body(body.into())?;
        let annotations = annotations.cloned().unwrap_or_else(Map::new);
        for (expr, field) in [(&var_expr, "var"), (&begin, "begin"), (&end, "end"), (&step, "step")]
        {
            let ty = expr.prim_ty();
            if !ty.is_scalar()
                || !ty.matches_code(&[DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLInt])
            {
                let message = format!(
                    "TIR For nodes require a scalar integer as the {field}, but received {} with \
                     dtype {}",
                    support::repr(expr),
                    support::repr(ty)
                );
                let condition = "dtype.IsScalar() && (dtype.MatchesCode(DLDataTypeCode::kDLUInt, \
                                 DLDataTypeCode::kDLInt))";
                return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, &message));
            }
        }
        // A constant of another integer type takes the type of `var` if its value fits.
        let promote = |expr: PrimExpr| -> Result<PrimExpr> {
            let ty = expr.prim_ty().clone();
            if ty.dtype == var_ty.dtype {
                return Ok(expr);
            }
            if let Some(imm) = expr.downcast_ref::<IntImmObj>() {
                if !int_imm_value_fits(&imm.value, &var_ty)? {
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
            if ty.bits() > var_ty.bits() {
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
            let ty = expr.prim_ty();
            if ty.dtype == var_ty.dtype {
                return Ok(());
            }
            let message = format!("{} vs {}", support::repr(&var_ty), support::repr(ty));
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

/// The loop running `body` while the scalar `condition` holds.
// tvm-ffi-stubgen(begin): object/ir.While
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct While {
    data_: ObjectArc<WhileObj>,
}

/// The object behind [`While`].
///
/// Complete: reflected fields fill [32, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.While"]
#[type_final]
pub struct WhileObj {
    base_: StmtObj,
    pub condition: PrimExpr,
    pub body: SeqStmt,
}

const _: () = {
    assert!(::core::mem::size_of::<WhileObj>() == 48);
    assert!(::core::mem::align_of::<WhileObj>() == 8);
};

impl Deref for WhileObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl WhileObj {
    pub(crate) fn new(loc: Location, condition: PrimExpr, body: SeqStmt) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_, condition, body }
    }
}

#[allow(missing_docs)]
impl While {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        condition: impl Into<PrimExpr>,
        body: impl Into<SeqStmt>,
    ) -> Self {
        let obj = WhileObj::new(loc.into(), condition.into(), body.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for WhileObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(While => Stmt);
// tvm-ffi-stubgen(end)

impl While {
    /// Create the loop running `body` while the scalar `condition` holds, checked to be well
    /// formed.
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
        let body = into_body(body.into())?;
        if !condition.prim_ty().is_scalar() {
            let check = "condition.ty().IsScalar()";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, check, ""));
        }
        Ok(Self::from_fields(loc, condition, body))
    }
}

/// The store of the primitive `value` at `indices` of `dest`: a store into a tensor, or into any
/// expression whose type supports it.
// tvm-ffi-stubgen(begin): object/ir.TensorStore
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorStore {
    data_: ObjectArc<TensorStoreObj>,
}

/// The object behind [`TensorStore`].
///
/// Complete: reflected fields fill [32, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorStore"]
#[type_final]
pub struct TensorStoreObj {
    base_: StmtObj,
    pub dest: Expr,
    pub indices: Array<PrimExpr>,
    pub value: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorStoreObj>() == 56);
    assert!(::core::mem::align_of::<TensorStoreObj>() == 8);
};

impl Deref for TensorStoreObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl TensorStoreObj {
    pub(crate) fn new(
        loc: Location,
        dest: Expr,
        indices: Array<PrimExpr>,
        value: PrimExpr,
    ) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_, dest, indices, value }
    }
}

#[allow(missing_docs)]
impl TensorStore {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        dest: impl Into<Expr>,
        indices: impl Into<Array<PrimExpr>>,
        value: impl Into<PrimExpr>,
    ) -> Self {
        let obj = TensorStoreObj::new(loc.into(), dest.into(), indices.into(), value.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for TensorStoreObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(TensorStore => Stmt);
// tvm-ffi-stubgen(end)

impl TensorStore {
    /// Create the store of `value` into the element of `dest` at `indices`, checked by the
    /// validation the type of `dest` registers.
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

/// A statement with one body, whose meaning the region operator `op` defines.
///
/// `args` and `attrs` are evaluated in the enclosing scope. `body_params` are the variables the
/// operator's contract defines for `body`, visible only there, and `result_vars` are defined for
/// the statements after the region.
// tvm-ffi-stubgen(begin): object/ir.RegionStmt
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RegionStmt {
    data_: ObjectArc<RegionStmtObj>,
}

/// The object behind [`RegionStmt`].
///
/// Complete: reflected fields fill [32, 80) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.RegionStmt"]
#[type_final]
pub struct RegionStmtObj {
    base_: StmtObj,
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

impl Deref for RegionStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
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
        let base_ = StmtObj::new(loc);
        Self { base_, op, args, body_params, attrs, body, result_vars }
    }
}

#[allow(missing_docs)]
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
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for RegionStmtObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(RegionStmt => Stmt);
// tvm-ffi-stubgen(end)

impl RegionStmt {
    /// Create the region of the region operator `op` on `args` with the attributes `attrs`,
    /// whose `body` binds `body_params` and which defines `result_vars`, checked against the
    /// contract of the operator.
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
            if !tvm_ffi::structural_equal(&param.ty, &expected.ty)? {
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

/// The statements `seq`, run in order: the body of a loop, conditional or region.
///
/// [`SeqStmt::new`] flattens nested sequences and drops no-ops.
// tvm-ffi-stubgen(begin): object/ir.SeqStmt
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SeqStmt {
    data_: ObjectArc<SeqStmtObj>,
}

/// The object behind [`SeqStmt`].
///
/// Complete: reflected fields fill [32, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base_: StmtObj,
    pub seq: Array<Stmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<SeqStmtObj>() == 40);
    assert!(::core::mem::align_of::<SeqStmtObj>() == 8);
};

impl Deref for SeqStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl SeqStmtObj {
    pub(crate) fn new(loc: Location, seq: Array<Stmt>) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_, seq }
    }
}

#[allow(missing_docs)]
impl SeqStmt {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, seq: impl Into<Array<Stmt>>) -> Self {
        let obj = SeqStmtObj::new(loc.into(), seq.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for SeqStmtObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(SeqStmt => Stmt);
// tvm-ffi-stubgen(end)

impl SeqStmt {
    /// Create the sequence of `seq`, with nested sequences flattened and no-ops (`Evaluate(0)`)
    /// dropped.
    pub fn new(seq: impl IntoIterator<Item = impl Into<Stmt>>, loc: impl Into<Location>) -> Self {
        let seq = seq.into_iter().map(Into::into).collect::<Array<Stmt>>();
        Self::from_fields(loc.into(), flatten(seq))
    }
}

/// The conditional running `then_case` when `condition` holds, and `else_case`, if any,
/// otherwise.
// tvm-ffi-stubgen(begin): object/ir.If
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct If {
    data_: ObjectArc<IfObj>,
}

/// The object behind [`If`].
///
/// Complete: reflected fields fill [32, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.If"]
#[type_final]
pub struct IfObj {
    base_: StmtObj,
    pub condition: PrimExpr,
    pub then_case: SeqStmt,
    pub else_case: Option<SeqStmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<IfObj>() == 56);
    assert!(::core::mem::align_of::<IfObj>() == 8);
};

impl Deref for IfObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl IfObj {
    pub(crate) fn new(
        loc: Location,
        condition: PrimExpr,
        then_case: SeqStmt,
        else_case: Option<SeqStmt>,
    ) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_, condition, then_case, else_case }
    }
}

#[allow(missing_docs)]
impl If {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        condition: impl Into<PrimExpr>,
        then_case: impl Into<SeqStmt>,
        else_case: impl Into<Option<SeqStmt>>,
    ) -> Self {
        let obj = IfObj::new(loc.into(), condition.into(), then_case.into(), else_case.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for IfObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(If => Stmt);
// tvm-ffi-stubgen(end)

impl If {
    /// Create the statement running `then_case` if `condition` holds, else `else_case` if given.
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

/// The evaluation of `value` for its side effects, its result discarded. `Evaluate` of the
/// constant `0` is a no-op.
// tvm-ffi-stubgen(begin): object/ir.Evaluate
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Evaluate {
    data_: ObjectArc<EvaluateObj>,
}

/// The object behind [`Evaluate`].
///
/// Complete: reflected fields fill [32, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base_: StmtObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<EvaluateObj>() == 40);
    assert!(::core::mem::align_of::<EvaluateObj>() == 8);
};

impl Deref for EvaluateObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl EvaluateObj {
    pub(crate) fn new(loc: Location, value: Expr) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_, value }
    }
}

#[allow(missing_docs)]
impl Evaluate {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, value: impl Into<Expr>) -> Self {
        let obj = EvaluateObj::new(loc.into(), value.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for EvaluateObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Evaluate => Stmt);
// tvm-ffi-stubgen(end)

impl Evaluate {
    /// Create the statement evaluating `value` for its side effects.
    pub fn new(value: impl Into<Expr>, loc: impl Into<Location>) -> Self {
        Self::from_fields(loc.into(), value.into())
    }
}

/// The exit from the innermost enclosing loop.
// tvm-ffi-stubgen(begin): object/ir.Break
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Break {
    data_: ObjectArc<BreakObj>,
}

/// The object behind [`Break`].
///
/// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Break"]
#[type_final]
pub struct BreakObj {
    base_: StmtObj,
}

const _: () = {
    assert!(::core::mem::size_of::<BreakObj>() == 32);
    assert!(::core::mem::align_of::<BreakObj>() == 8);
};

impl Deref for BreakObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl BreakObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl Break {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = BreakObj::new(loc.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BreakObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Break => Stmt);
// tvm-ffi-stubgen(end)

// `ir.Break` stores its location as given: the generated `Break::new` is the same.

/// The exit from the enclosing function, returning `value`.
// tvm-ffi-stubgen(begin): object/ir.Return
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Return {
    data_: ObjectArc<ReturnObj>,
}

/// The object behind [`Return`].
///
/// Complete: reflected fields fill [32, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Return"]
#[type_final]
pub struct ReturnObj {
    base_: StmtObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<ReturnObj>() == 40);
    assert!(::core::mem::align_of::<ReturnObj>() == 8);
};

impl Deref for ReturnObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl ReturnObj {
    pub(crate) fn new(loc: Location, value: Expr) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_, value }
    }
}

#[allow(missing_docs)]
impl Return {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, value: impl Into<Expr>) -> Self {
        let obj = ReturnObj::new(loc.into(), value.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ReturnObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Return => Stmt);
// tvm-ffi-stubgen(end)

impl Return {
    /// Create the statement returning `value` from the enclosing function.
    pub fn new(value: impl Into<Expr>, loc: impl Into<Location>) -> Self {
        Self::from_fields(loc.into(), value.into())
    }
}

/// The jump to the next iteration of the innermost enclosing loop.
// tvm-ffi-stubgen(begin): object/ir.Continue
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Continue {
    data_: ObjectArc<ContinueObj>,
}

/// The object behind [`Continue`].
///
/// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Continue"]
#[type_final]
pub struct ContinueObj {
    base_: StmtObj,
}

const _: () = {
    assert!(::core::mem::size_of::<ContinueObj>() == 32);
    assert!(::core::mem::align_of::<ContinueObj>() == 8);
};

impl Deref for ContinueObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl ContinueObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = StmtObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl Continue {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = ContinueObj::new(loc.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ContinueObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Continue => Stmt);
// tvm-ffi-stubgen(end)

// `ir.Continue` stores its location as given: the generated `Continue::new` is the same.

// ---------------------------------------------------------------------------------------------
// Helpers

/// Whether `value` is in the range of the integer type `ty`.
fn int_imm_value_fits(value: &BigInt, ty: &PrimType) -> Result<bool> {
    let bits = u32::from(ty.bits());
    if ty.matches_code(&[DLDataTypeCode::kDLUInt]) {
        let limit = BigInt::from(1i64).try_shl(bits as i64)?;
        return Ok(!value.is_negative() && *value < limit);
    }
    let limit = BigInt::from(1i64).try_shl(bits as i64 - 1)?;
    Ok(*value >= -limit.clone() && *value < limit)
}

/// The parameters the body of a region of `op` on `args` binds, which the operator's
/// `FRegionGetBodyParams` gives once its validator accepts the arguments.
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

/// Whether `stmt` is `Evaluate(0)`, the no-op a sequence drops.
fn is_no_op(stmt: &Stmt) -> bool {
    stmt.downcast_ref::<EvaluateObj>()
        .and_then(|evaluate| evaluate.value.downcast_ref::<IntImmObj>())
        .is_some_and(|imm| imm.value == BigInt::from(0i64))
}

/// A statement as a body: a sequence as it is, any other statement as the sequence of it at its
/// location.
fn into_body(stmt: Stmt) -> Result<SeqStmt> {
    if stmt.downcast_ref::<SeqStmtObj>().is_some() {
        return Ok(stmt.downcast().expect("a SeqStmt"));
    }
    let loc = stmt.loc.clone();
    Ok(SeqStmt::from_fields(loc, flatten(Array::from_vec(vec![stmt]))))
}
