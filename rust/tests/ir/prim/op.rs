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

//! Tests of `ir::prim::op`: the type-promoting operators and the math functions written in Rust,
//! against the native functions they follow (`prim._OpAdd` and the like). Every binary operator
//! runs on every pair of a grid of operands, which must give the same expression or an error of
//! the same kind with the same message. Literals are typed, and `-`, `!` and an `ExprResult` chain build,
//! as in C++.

use tvm_tirx::ir::prim::op::{self, ExprResult, PrimOps};
use tvm_tirx::ir::prim::{self, Add, Broadcast, BroadcastObj, Cast, CastObj, Not, Ramp, RampObj, Sub};
use tvm_tirx::ir::{
    Call, Expr, FloatImm, IntImm, Location, Op, PrimExpr, PrimType, Range, SourceLoc, SourceName,
    StringImm, Type, UnknownLocObj, UNKNOWN_LOC,
};
use tvm_tirx::sym;
use tvm_tirx::tvm_ffi::{
    Any, AnyCompatible, AnyView, BigInt, ObjectRefCast, ObjectRefCore, Result, String as FfiString,
    TYPE_ERROR,
};

use super::{dtype, int_value, var};
use crate::common::{agree, global, serial, structural_equal};
use crate::parse_dtype;

/// The primitive type `name`.
fn prim(name: &str) -> PrimType {
    PrimType::from_dtype(parse_dtype(name)).unwrap()
}

/// A variable of the primitive type `ty`, named after it.
fn prim_var(ty: &str) -> PrimExpr {
    var(&format!("v_{}", ty.replace('x', "_")), prim(ty)).into()
}

fn imm(ty: &str, value: i64) -> PrimExpr {
    IntImm::new(PrimType::from_dtype(parse_dtype(ty)).unwrap(), value, UNKNOWN_LOC).unwrap().into()
}

fn big_imm(ty: &str, value: BigInt) -> PrimExpr {
    IntImm::new(PrimType::from_dtype(parse_dtype(ty)).unwrap(), value, UNKNOWN_LOC).unwrap().into()
}

fn fimm(ty: &str, value: f64) -> PrimExpr {
    FloatImm::new(PrimType::from_dtype(parse_dtype(ty)).unwrap(), value, UNKNOWN_LOC)
        .unwrap()
        .into()
}

fn repr<T>(value: &T) -> String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let text: FfiString =
        global("ffi.ReprPrint").call_packed(&[AnyView::from(value)]).unwrap().try_into().unwrap();
    text.as_str().to_string()
}

/// The native operator `name` applied to two expressions.
fn native2(name: &str, a: impl Into<Expr>, b: impl Into<Expr>) -> Expr {
    global(name).call_tuple((a.into(), b.into(), Location::unknown())).unwrap().try_into().unwrap()
}

/// The native unary function `name`.
fn native1(name: &str, x: impl Into<Expr>) -> Result<Any> {
    global(name).call_tuple((x.into(), Location::unknown()))
}

fn same(built: ExprResult, expected: &impl AnyCompatible) -> bool {
    structural_equal(&PrimExpr::try_from(built).unwrap(), expected)
}

/// The cases where the Rust operator and the native one disagree, and how many were compared.
#[derive(Default)]
struct Outcome {
    compared: usize,
    built: usize,
    errors: usize,
    mismatches: Vec<String>,
}

impl Outcome {
    /// The Rust result and the native one agree: the same expression, or an error of the same
    /// kind with the same message.
    fn check(
        &mut self,
        what: impl FnOnce() -> String,
        rust: Result<PrimExpr>,
        native: Result<Any>,
    ) {
        self.compared += 1;
        let mismatch = match (rust, native) {
            (Ok(rust), Ok(native)) => {
                self.built += 1;
                let native: Expr = native.try_into().unwrap();
                match structural_equal(&rust, &native) {
                    true => None,
                    false => Some(format!("{} vs {}", repr(&rust), repr(&native))),
                }
            }
            (Err(rust), Err(native)) => {
                self.errors += 1;
                match rust.kind() == native.kind() && rust.message() == native.message() {
                    true => None,
                    false => Some(format!(
                        "{}: {:?} vs {}: {:?}",
                        rust.kind(),
                        rust.message(),
                        native.kind(),
                        native.message()
                    )),
                }
            }
            (Ok(rust), Err(native)) => {
                Some(format!("Rust gave {}, C++ failed with {}", repr(&rust), native.message()))
            }
            (Err(rust), Ok(native)) => {
                let native: Expr = native.try_into().unwrap();
                Some(format!("Rust failed with {}, C++ gave {}", rust.message(), repr(&native)))
            }
        };
        if let Some(mismatch) = mismatch {
            self.mismatches.push(format!("{}: {mismatch}", what()));
        }
    }

    fn assert_agrees(&self, what: &str) {
        let shown: Vec<&str> = self.mismatches.iter().take(20).map(String::as_str).collect();
        assert!(
            self.mismatches.is_empty(),
            "{what}: {} of {} cases disagree:\n{}",
            self.mismatches.len(),
            self.compared,
            shown.join("\n")
        );
        println!(
            "{what}: {} cases ({} expressions, {} errors) agree",
            self.compared, self.built, self.errors
        );
    }
}

/// Operands of every kind the operators treat differently: variables, constants including 0, 1,
/// -1, the limits of each type and values that overflow, vectors, broadcasts, a ramp, the
/// symbolic infinities and `vscale()` calls.
fn operands() -> Vec<PrimExpr> {
    let mut operands: Vec<PrimExpr> = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint32",
        "uint64",
        "bool",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int32x4",
        "int64x4",
        "float32x4",
        "boolx4",
        "int32x8",
    ]
    .into_iter()
    .map(prim_var)
    .collect();
    operands.push(var("w", PrimType::int(32)).into());
    for (ty, values) in [
        ("int8", vec![0i64, 1, -1, 127, -128, 5]),
        ("int16", vec![0, 1, 32767, -3]),
        ("int32", vec![0, 1, -1, 2, 3, 7, -5, 31, 32, i64::from(i32::MAX), i64::from(i32::MIN)]),
        ("int64", vec![0, 1, -1, 3, 63, 64, 1 << 40, i64::MAX, i64::MIN]),
        ("uint8", vec![0, 1, 7, 255]),
        ("uint32", vec![0, 1, 7, i64::from(u32::MAX)]),
        ("bool", vec![0, 1]),
    ] {
        for value in values {
            operands.push(imm(ty, value));
        }
    }
    operands.push(imm("uint64", 0));
    operands.push(imm("uint64", 5));
    operands.push(big_imm("uint64", BigInt::from(u64::MAX)));
    for (ty, values) in [
        ("float16", vec![0.0, -0.0, 1.0, 2.5, 65504.0]),
        (
            "float32",
            vec![0.0, -0.0, 1.0, 2.5, -2.5, 3.0e38, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
        ),
        ("float64", vec![0.0, 1.0, -4.5, 1e300, f64::INFINITY, f64::NAN]),
        ("bfloat16", vec![0.0, 1.5]),
    ] {
        for value in values {
            operands.push(fimm(ty, value));
        }
    }
    operands.push(Broadcast::new(imm("int32", 7), 4, UNKNOWN_LOC).unwrap().into());
    operands.push(Broadcast::new(imm("int32", 0), 4, UNKNOWN_LOC).unwrap().into());
    operands.push(Broadcast::new(fimm("float32", 2.5), 4, UNKNOWN_LOC).unwrap().into());
    operands.push(Ramp::new(imm("int32", 0), imm("int32", 1), 4, UNKNOWN_LOC).unwrap().into());
    operands.push(sym::pos_inf().unwrap());
    operands.push(sym::neg_inf().unwrap());
    operands.push(op::vscale().unwrap());
    operands.push(op::vscale().unwrap());
    operands
}

type Binary = fn(&PrimExpr, &PrimExpr) -> ExprResult;

#[test]
fn binary_operators_agree_with_cpp() {
    let _serial = serial();
    let operators: Vec<(&str, Binary)> = vec![
        ("prim._OpAdd", |a, b| a + b),
        ("prim._OpSub", |a, b| a - b),
        ("prim._OpMul", |a, b| a * b),
        ("prim._OpDiv", |a, b| a.div(b)),
        ("prim._OpTruncDiv", |a, b| a.truncdiv(b)),
        ("prim._OpTruncMod", |a, b| a.truncmod(b)),
        ("prim._OpFloorDiv", |a, b| a.floordiv(b)),
        ("prim._OpFloorMod", |a, b| a.floormod(b)),
        ("prim._OpIndexDiv", |a, b| a.indexdiv(b)),
        ("prim._OpIndexMod", |a, b| a.indexmod(b)),
        ("prim._OpCeilDiv", |a, b| a.ceildiv(b)),
        ("prim._OpMin", |a, b| a.min(b)),
        ("prim._OpMax", |a, b| a.max(b)),
        ("prim._OpEQ", |a, b| a.equal(b)),
        ("prim._OpNE", |a, b| a.not_equal(b)),
        ("prim._OpLT", |a, b| a.less(b)),
        ("prim._OpLE", |a, b| a.less_equal(b)),
        ("prim._OpGT", |a, b| a.greater(b)),
        ("prim._OpGE", |a, b| a.greater_equal(b)),
        ("prim._OpAnd", |a, b| a.logical_and(b)),
        ("prim._OpOr", |a, b| a.logical_or(b)),
        ("prim.bitwise_and", |a, b| a & b),
        ("prim.bitwise_or", |a, b| a | b),
        ("prim.bitwise_xor", |a, b| a ^ b),
        ("prim.left_shift", |a, b| a << b),
        ("prim.right_shift", |a, b| a >> b),
    ];
    let operands = operands();
    let mut total = Outcome::default();
    for (name, operator) in operators {
        let native_op = global(name);
        let mut outcome = Outcome::default();
        for a in &operands {
            for b in &operands {
                let native = native_op.call_tuple((a.clone(), b.clone(), Location::unknown()));
                let what = || format!("{name}({}, {})", repr(a), repr(b));
                outcome.check(what, operator(a, b).try_into(), native);
            }
        }
        outcome.assert_agrees(name);
        total.compared += outcome.compared;
        total.built += outcome.built;
        total.errors += outcome.errors;
    }
    total.assert_agrees("all binary operators");
}

#[test]
fn unary_and_ternary_operators_agree_with_cpp() {
    let _serial = serial();
    let operands = operands();
    let mut outcome = Outcome::default();
    // C++ `bitwise_neg` has no global; it checks the operand type and builds `BitwiseNot`, whose
    // own check rejects the same operands with another message.
    let bitwise_not = global("prim.BitwiseNot");
    for a in &operands {
        let native = bitwise_not.call_tuple((a.clone(), Location::unknown()));
        match a.bitwise_not().try_into() {
            Ok(rust) => outcome.check(|| format!("~{}", repr(a)), Ok(rust), native),
            Err(_) => assert!(native.is_err(), "~{}: only Rust rejects it", repr(a)),
        }
    }
    outcome.assert_agrees("bitwise_neg");

    let mut outcome = Outcome::default();
    let conditions: Vec<PrimExpr> = vec![
        prim_var("bool"),
        imm("bool", 0),
        imm("bool", 1),
        prim_var("boolx4"),
        prim_var("int32"),
        imm("int32", 1),
    ];
    let values: Vec<PrimExpr> = operands.iter().step_by(3).cloned().collect();
    let if_then_else = global("prim._OpIfThenElse");
    for c in &conditions {
        for t in &values {
            for f in &values {
                let native =
                    if_then_else.call_tuple((c.clone(), t.clone(), f.clone(), Location::unknown()));
                let what = || format!("if_then_else({}, {}, {})", repr(c), repr(t), repr(f));
                outcome.check(what, c.if_then_else(t, f).try_into(), native);
            }
        }
    }
    outcome.assert_agrees("prim._OpIfThenElse");
}

#[test]
fn literals_and_compound_expressions_agree_with_cpp() {
    let _serial = serial();
    // A literal is typed after the other side, then the operator folds as for two constants.
    let x = prim_var("int64");
    let add = global("prim._OpAdd");
    let mut outcome = Outcome::default();
    outcome.check(
        || "x + 3".into(),
        (&x + 3).try_into(),
        add.call_tuple((x.clone(), imm("int64", 3), Location::unknown())),
    );
    outcome.check(
        || "3 + x".into(),
        (3i32 + &x).try_into(),
        add.call_tuple((imm("int64", 3), x.clone(), Location::unknown())),
    );
    let floordiv = global("prim._OpFloorDiv");
    let rust = ((&x + 0) * 1).floordiv(1);
    outcome.check(
        || "((x + 0) * 1) // 1".into(),
        rust.try_into(),
        floordiv.call_tuple((x.clone(), imm("int64", 1), Location::unknown())),
    );
    // Unary minus folds a constant whose negation fits its type and is `0 - x` otherwise.
    let sub = global("prim._OpSub");
    for value in [imm("int64", i64::MIN), imm("int32", 5), fimm("float32", 2.5), prim_var("int32")]
    {
        let zero = op::make_const(value.ty(), 0).unwrap();
        let native = sub.call_tuple((zero, value.clone(), Location::unknown()));
        outcome.check(|| format!("-{}", repr(&value)), (-&value).try_into(), native);
    }
    outcome.assert_agrees("literals");
}

#[test]
fn literals_are_typed_as_in_cpp() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let w = var("w", PrimType::int(64));
    let v = var("v", PrimType::int(32).with_lanes(4).unwrap());
    let f = var("f", PrimType::float(32));

    // An integer is `MakeConst(other.ty(), v)`, on either side.
    assert!(same(&x + 1, &native2("prim._OpAdd", &x, imm("int32", 1))));
    assert!(same(1 + &x, &native2("prim._OpAdd", imm("int32", 1), &x)));
    assert!(same(&w - 1, &native2("prim._OpSub", &w, imm("int64", 1))));
    let two_x4 = op::make_const(PrimType::int(32).with_lanes(4).unwrap(), 2).unwrap();
    assert!(same(&v * 2, &native2("prim._OpMul", &v, two_x4)));
    assert!(same(&f + 1, &native2("prim._OpAdd", &f, fimm("float32", 1.0))));
    // A `double` is a float64 `FloatImm`, a `float` a float32 one, whatever the other side.
    assert!(same(&f + 1.5, &native2("prim._OpAdd", &f, fimm("float64", 1.5))));
    assert!(same(&f + 1.5_f32, &native2("prim._OpAdd", &f, fimm("float32", 1.5))));
    // The named operations and the free functions type a literal the same way.
    assert!(same(x.greater_equal(0), &native2("prim._OpGE", &x, imm("int32", 0))));
    assert!(same(w.floordiv(8), &native2("prim._OpFloorDiv", &w, imm("int64", 8))));
    assert!(same(&x >> 3, &native2("prim.right_shift", &x, imm("int32", 3))));
    let free = op::floordiv(&w, 8).unwrap();
    assert!(structural_equal(&free, &native2("prim._OpFloorDiv", &w, imm("int64", 8))));
    let free = op::sub(1, &w).unwrap();
    assert!(structural_equal(&free, &native2("prim._OpSub", imm("int64", 1), &w)));
    let free = op::add(&f, 1.5).unwrap();
    assert!(structural_equal(&free, &native2("prim._OpAdd", &f, fimm("float64", 1.5))));
    // Two literals are the constants `From` makes of them.
    let free = op::add(1, 2.5_f32).unwrap();
    assert!(structural_equal(
        &free,
        &native2("prim._OpAdd", imm("int32", 1), fimm("float32", 2.5))
    ));
}

/// An `ExprResult` passes into the free functions and the constructors, which report its error.
#[test]
fn expression_results_pass_into_functions_and_constructors() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    // C++ `floordiv(x + 1, 4)`.
    let index = op::floordiv(&x + 1, 4).unwrap();
    let sum = native2("prim._OpAdd", &x, imm("int32", 1));
    assert!(structural_equal(&index, &native2("prim._OpFloorDiv", sum.clone(), imm("int32", 4))));
    assert!(structural_equal(&op::floor(&x + 1).unwrap(), &sum));
    let raw: Expr = Add::new(&x + 1, &x * 2, UNKNOWN_LOC).unwrap().into();
    let product = PrimExpr::try_from(native2("prim._OpMul", &x, imm("int32", 2))).unwrap();
    let expected = Add::new(PrimExpr::try_from(&sum).unwrap(), product, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&raw, &expected));
    let range = Range::new(0, &x + 1, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&range.extent, &sum));
    assert!(structural_equal(
        &op::all([(&x + 1).greater(0), x.less(4)]).unwrap().ty,
        &PrimType::bool()
    ));

    // The first error of the chain is what the function or constructor reports.
    let f = var("f", PrimType::float(32));
    let failed = f.truncdiv(&x) + 1;
    let expected = PrimExpr::try_from(failed.clone()).err().unwrap();
    for error in [
        op::floordiv(failed.clone(), 4).err().unwrap(),
        op::floordiv(4, &failed).err().unwrap(),
        op::floor(&failed).err().unwrap(),
        Add::new(&failed, &x, UNKNOWN_LOC).err().unwrap(),
        Range::new(0, failed, UNKNOWN_LOC).err().unwrap(),
    ] {
        assert_eq!(error.kind(), expected.kind());
        assert_eq!(error.message(), expected.message());
    }
}

/// The `_at` variants build at the given location; the others at the unknown one.
#[test]
fn located_variants() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let here: Location = SourceLoc::new(SourceName::get("ops.rs").unwrap(), 1, 2, 3, 4).into();
    let located: prim::FloorDiv = op::floordiv_at(&x, 4, here.clone()).unwrap().try_cast().unwrap();
    assert!(located.loc.same_as(&here));
    let unlocated: prim::FloorDiv = op::floordiv(&x, 4).unwrap().try_cast().unwrap();
    assert!(unlocated.loc.downcast_ref::<UnknownLocObj>().is_some());
    // A folded constant is at the unknown location, as C++ `TryConstFold` builds it.
    let folded: IntImm = op::add_at(1, 2, here.clone()).unwrap().try_cast().unwrap();
    assert!(folded.loc.downcast_ref::<UnknownLocObj>().is_some());
    let cast: Cast = op::cast_at(&x, PrimType::int(64), here.clone()).unwrap().try_cast().unwrap();
    assert!(cast.loc.same_as(&here));
    let call: Call =
        op::exp_at(var("f", PrimType::float(32)), here.clone()).unwrap().try_cast().unwrap();
    assert!(call.loc.same_as(&here));
    let limit: IntImm =
        op::max_value_at(PrimType::int(8), here.clone()).unwrap().try_cast().unwrap();
    assert!(limit.loc.same_as(&here));
}

#[test]
fn unary_operators_follow_neg_and_logical_not() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let p = var("p", PrimType::bool());
    // `-x` is `MakeConst(x.ty(), 0) - x`; a constant is negated in place when the result fits.
    assert!(same(-&x, &native2("prim._OpSub", imm("int32", 0), &x)));
    assert!(same(-imm("int32", 5), &imm("int32", -5)));
    let min = imm("int64", i64::MIN);
    assert!(same(-&min, &native2("prim._OpSub", imm("int64", 0), &min)));
    assert!(same(-fimm("float32", 0.0), &fimm("float32", -0.0)));
    // `!p` is the raw `Not`; a constant folds.
    let not_p = Not::new(&p, UNKNOWN_LOC).unwrap();
    assert!(same(!&p, &not_p));
    assert!(same(!imm("bool", 1), &imm("bool", 0)));
}

#[test]
fn select_folding_and_the_first_error() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let r = var("r", PrimType::int(32));
    // `select` is the raw constructor, built from operands under construction.
    let built = PrimExpr::try_from(r.greater_equal(0).select(&r, &r - 1)).unwrap();
    let cond = native2("prim._OpGE", &r, imm("int32", 0));
    let minus = native2("prim._OpSub", &r, imm("int32", 1));
    let expected: Expr = global("prim.Select")
        .call_tuple((cond, Expr::from(&r), minus, Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(structural_equal(&built, &expected));
    // The raw `Sub` does not fold; the operator does.
    let raw = Sub::new(3, 1, UNKNOWN_LOC).unwrap();
    assert!(!structural_equal(&Expr::from(raw), &imm("int32", 2)));
    assert!(same(imm("int32", 3) - 1, &imm("int32", 2)));

    // Integer division of a float raises natively; later steps pass the error through.
    let f = var("f", PrimType::float(32));
    let Err(error) = PrimExpr::try_from((f.truncdiv(&x) + 1).less(&x)) else {
        panic!("expected an error")
    };
    let direct = global("prim._OpTruncDiv").call_tuple((
        Expr::from(&f),
        Expr::from(&x),
        Location::unknown(),
    ));
    let direct = direct.err().expect("the native operation raises");
    assert_eq!(error.kind(), direct.kind());
    assert_eq!(error.message(), direct.message());
    // The type system rejects `string + 1`; the check a non-PrimType operand needs fails.
    let string: Expr = StringImm::new("s", UNKNOWN_LOC).into();
    assert!(PrimExpr::try_from(string).is_err());
}

/// Variables and constants of the types the math functions treat differently.
fn values() -> Vec<PrimExpr> {
    let mut values: Vec<PrimExpr> = [
        "int32",
        "int64",
        "uint8",
        "uint32",
        "bool",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int32x4",
        "float32x4",
        "bfloat16x4",
    ]
    .into_iter()
    .map(prim_var)
    .collect();
    values.extend([
        imm("int32", -5),
        imm("int32", 7),
        imm("int64", i64::MIN),
        imm("uint32", 7),
        imm("bool", 1),
        fimm("float32", 2.5),
        fimm("float32", -2.5),
        fimm("float32", 3.5),
        fimm("float32", -0.5),
        fimm("float64", 1e300),
        fimm("float16", -1.25),
        fimm("bfloat16", 1.5),
    ]);
    values
}

#[test]
fn unary_functions_agree_with_cpp() {
    let _serial = serial();
    type Unary = fn(&PrimExpr) -> ExprResult;
    let functions: [(&str, Unary); 11] = [
        ("prim.floor", |x| x.floor()),
        ("prim.ceil", |x| x.ceil()),
        ("prim.round", |x| x.round()),
        ("prim.nearbyint", |x| x.nearbyint()),
        ("prim.trunc", |x| x.trunc()),
        ("prim.abs", |x| x.abs()),
        ("prim.isnan", |x| x.isnan()),
        ("prim.isinf", |x| x.isinf()),
        ("prim.isfinite", |x| x.isfinite()),
        ("prim.likely", |x| x.likely()),
        ("prim.assume", |x| x.assume()),
    ];
    for (name, function) in functions {
        for x in values() {
            agree(
                &format!("{name}({})", repr(&x)),
                PrimExpr::try_from(function(&x)),
                native1(name, &x),
            );
        }
    }
    // `round` folds a tie to even, as `std::nearbyint` does.
    assert!(same(fimm("float32", 2.5).round(), &fimm("float32", 2.0)));
}

#[test]
fn limits_agree_with_cpp() {
    let _serial = serial();
    let types = [
        "int8",
        "int32",
        "int64",
        "uint8",
        "uint32",
        "uint64",
        "bool",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
        "float6_e2m3fn",
        "float4_e2m1fn",
        "int32x4",
        "float32x4",
    ];
    for name in types {
        let ty = prim(name);
        let native =
            |function: &str| global(function).call_tuple((ty.clone(), Location::unknown()));
        agree(&format!("infinity({name})"), op::infinity(&ty), native("prim.infinity"));
        agree(&format!("max_value({name})"), op::max_value(&ty), native("prim.max_value"));
        agree(&format!("min_value({name})"), op::min_value(&ty), native("prim.min_value"));
    }
}

#[test]
fn cast_agrees_with_cpp() {
    let _serial = serial();
    let mut values = values();
    values.push(Broadcast::new(prim_var("float32"), 4, UNKNOWN_LOC).unwrap().into());
    values.push(Ramp::new(prim_var("int32"), imm("int32", 1), 4, UNKNOWN_LOC).unwrap().into());
    values.push(var("void", PrimType::void()).into());
    let targets = [
        "int32",
        "int64",
        "uint8",
        "uint32",
        "bool",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int32x4",
        "float32x4",
        "int64x4",
        "float16x4",
        "int32x8",
    ];
    for x in &values {
        for name in targets {
            let ty = prim(name);
            let native =
                global("prim._cast").call_tuple((x.clone(), ty.clone(), Location::unknown()));
            agree(&format!("cast({}, {name})", repr(x)), PrimExpr::try_from(x.cast(ty)), native);
        }
    }
}

#[test]
fn type_matching_agrees_with_the_native_operators() {
    let _serial = serial();
    let pairs = [
        ("int32", "int64"),
        ("int64", "int32"),
        ("int32", "uint32"),
        ("uint32", "int32"),
        ("uint8", "int32"),
        ("uint32", "int64"),
        ("bool", "int32"),
        ("int32", "bool"),
        ("float16", "float32"),
        ("int32", "float32"),
        ("float32", "int64"),
        ("bfloat16", "float32"),
        ("bfloat16", "int32"),
        ("bool", "float16"),
        ("int32x4", "int32"),
        ("float32", "float32x4"),
        ("int32", "float32x4"),
        ("float32x4", "int64"),
        ("int32x4", "int32x8"),
    ];
    let raw_add = |x: PrimExpr, y: PrimExpr| -> Result<PrimExpr> {
        op::match_types(x, y).and_then(|(x, y)| Ok(Add::new(x, y, UNKNOWN_LOC)?.into()))
    };
    for (a, b) in pairs {
        let (x, y) = (var("x", prim(a)), var("y", prim(b)));
        // The raw `Add` of the matched sides is what C++ `x + y` builds when nothing folds.
        let native =
            global("prim._OpAdd").call_tuple((Expr::from(&x), Expr::from(&y), Location::unknown()));
        agree(&format!("{a} + {b}"), raw_add(x.into(), y.into()), native);
    }
    // A constant side is cast by folding.
    let (x, three) = (prim_var("float32"), imm("int32", 3));
    let native = global("prim._OpAdd").call_tuple((x.clone(), three.clone(), Location::unknown()));
    agree("float32 + 3", raw_add(x, three), native);
}

#[test]
fn pow_and_fmod_agree_with_cpp() {
    let _serial = serial();
    let pairs: Vec<(PrimExpr, PrimExpr)> = vec![
        (prim_var("float32"), var("w", PrimType::float(32)).into()),
        (prim_var("float16"), prim_var("float32")),
        (prim_var("float32"), imm("int32", 2)),
        (prim_var("float32x4"), prim_var("float32")),
        (prim_var("float32"), fimm("float16", 0.5)),
        (prim_var("int32"), var("j", PrimType::int(32)).into()),
    ];
    for (x, y) in &pairs {
        let native = global("prim._OpPow").call_tuple((x.clone(), y.clone(), Location::unknown()));
        agree(&format!("pow({}, {})", repr(x), repr(y)), PrimExpr::try_from(x.pow(y)), native);
    }
    // `fmod` has no global: the sides are matched and the call is typed as them.
    let (h, f) = (prim_var("float16"), prim_var("float32"));
    let widened: PrimExpr = Cast::new(PrimType::float(32), &h, UNKNOWN_LOC).unwrap().into();
    let fmod = Op::get("prim.fmod").unwrap();
    let ty: Type = PrimType::float(32).into();
    let expected =
        Call::new(Some(&ty), fmod, [widened, f.clone()], None, Vec::<Type>::new(), UNKNOWN_LOC);
    assert!(structural_equal(&PrimExpr::try_from(h.fmod(&f)).unwrap(), &expected.unwrap()));
    let error = PrimExpr::try_from(prim_var("int32").fmod(imm("int32", 2))).err().unwrap();
    assert!(error.message().contains("fmod only applies to float"), "{}", error.message());
}

#[test]
fn intrinsics_are_typed_calls() {
    let _serial = serial();
    // A call typed by a `PrimType`, checked to be a `PrimExpr`.
    let call = |name: &str, op: &str, args: Vec<PrimExpr>| -> PrimExpr {
        let ty: Type = prim(name).into();
        let call =
            Call::new(Some(&ty), Op::get(op).unwrap(), args, None, Vec::<Type>::new(), UNKNOWN_LOC);
        PrimExpr::try_from(Expr::from(call.unwrap())).unwrap()
    };
    let cast = |value: PrimExpr, name: &str| -> PrimExpr {
        Cast::new(prim(name), value, UNKNOWN_LOC).unwrap().into()
    };

    let x = prim_var("float32");
    assert!(same(x.exp(), &call("float32", "prim.exp", vec![x.clone()])));
    assert!(same(x.tan(), &call("float32", "prim.tan", vec![x.clone()])));
    assert!(structural_equal(&op::exp(&x).unwrap(), &call("float32", "prim.exp", vec![x.clone()])));
    // Only the trigonometric and hyperbolic ones reject an integer.
    let i = prim_var("int32");
    assert!(same(i.exp(), &call("int32", "prim.exp", vec![i.clone()])));
    assert!(same(i.clz(), &call("int32", "prim.clz", vec![i.clone()])));
    let error = PrimExpr::try_from(i.tan()).err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error
        .message()
        .ends_with("prim.tan only supports floating-point inputs, but got T.int32"));
    // A bfloat16 argument is computed in float32 and cast back, keeping its lanes.
    for (ty, wide) in [("bfloat16", "float32"), ("bfloat16x4", "float32x4")] {
        let b = prim_var(ty);
        for (built, op) in
            [(b.sqrt(), "prim.sqrt"), (b.log2(), "prim.log2"), (b.tanh(), "prim.tanh")]
        {
            let expected = cast(call(wide, op, vec![cast(b.clone(), wide)]), ty);
            assert!(same(built, &expected), "{op}({ty})");
        }
    }
    // A binary intrinsic is typed as its first argument, its arguments unmatched; `fma` too.
    let h = prim_var("float16");
    assert!(same(x.atan2(&h), &call("float32", "prim.atan2", vec![x.clone(), h.clone()])));
    assert!(same(x.copysign(&h), &call("float32", "prim.copysign", vec![x.clone(), h.clone()])));
    let fma = op::fma(&x, &h, &x).unwrap();
    assert!(structural_equal(&fma, &call("float32", "prim.fma", vec![x.clone(), h, x.clone()])));
    assert!(same(ExprResult::from(op::vscale()), &call("int32", "prim.vscale", vec![])));
    // Two `vscale()` calls are equal: a bool, as every comparison folds to.
    assert!(same(
        ExprResult::from(op::vscale()).equal(ExprResult::from(op::vscale())),
        &imm("bool", 1)
    ));
}

/// C++ `cast(value, t)` through its global.
fn native_cast(value: impl Into<Expr>, name: &str) -> Expr {
    global("prim._cast")
        .call_tuple((value.into(), prim(name), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap()
}

#[test]
fn cast_folds_and_broadcasts_as_cpp() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let f = var("f", PrimType::float(32));
    let ramp = |base: PrimExpr| -> PrimExpr {
        Ramp::new(base, imm("int32", 1), 4, UNKNOWN_LOC).unwrap().into()
    };
    // The last lane of `overflowing` is beyond the int32 range.
    let (constant, overflowing) = (ramp(imm("int32", 0)), ramp(imm("int32", i32::MAX.into())));

    // A value already of the type is itself.
    assert!(PrimExpr::try_from(x.cast(PrimType::int(32))).unwrap().same_as(&x));

    let cases: Vec<(PrimExpr, &str)> = vec![
        // A constant folds to the new type when it is the same value there; one the type cannot
        // hold, and anything else, becomes a `Cast` node.
        (imm("int32", 5), "int64"),
        (fimm("float32", 2.5), "int32"),
        (imm("int32", 1), "float16"),
        (imm("int32", -5), "uint8"),
        (x.clone().into(), "int64"),
        // A scalar cast to a vector type is broadcast, its element cast first if needed.
        (x.clone().into(), "int32x4"),
        (f.clone().into(), "int32x4"),
        (imm("int32", 3), "int64x4"),
        // A broadcast is cast element by element, and so is a ramp of constants cast to a 32-bit
        // or 64-bit integer type when every lane fits both types; any other ramp becomes a `Cast`
        // node.
        (Broadcast::new(&f, 4, UNKNOWN_LOC).unwrap().into(), "int32x4"),
        (constant.clone(), "int64x4"),
        (overflowing.clone(), "int64x4"),
        (ramp(x.clone().into()), "int64x4"),
        (constant.clone(), "float32x4"),
        (constant.clone(), "int16x4"),
    ];
    for (value, name) in &cases {
        assert!(same(value.cast(prim(name)), &native_cast(value, name)), "{name}");
    }
    assert!(same(imm("int32", 5).cast(prim("int64")), &imm("int64", 5)));
    let node = |value: &PrimExpr, name: &str| PrimExpr::try_from(value.cast(prim(name))).unwrap();
    let broadcast = node(&f.clone().into(), "int32x4");
    let element = &broadcast.downcast_ref::<BroadcastObj>().unwrap().value;
    assert!(element.downcast_ref::<CastObj>().is_some());
    assert!(node(&imm("int32", -5), "uint8").downcast_ref::<CastObj>().is_some());
    assert!(node(&constant, "int64x4").downcast_ref::<RampObj>().is_some());
    assert!(node(&overflowing, "int64x4").downcast_ref::<CastObj>().is_some());
    assert!(node(&ramp(x.clone().into()), "int64x4").downcast_ref::<CastObj>().is_some());
    assert!(node(&constant, "float32x4").downcast_ref::<CastObj>().is_some());
    assert!(node(&constant, "int16x4").downcast_ref::<CastObj>().is_some());
    let raw: Expr = Cast::new(PrimType::int(64), &x, UNKNOWN_LOC).unwrap().into();
    assert!(same(x.cast(PrimType::int(64)), &raw));

    // The cast chains with the operators.
    let sum = native2("prim._OpAdd", native_cast(&x, "int64"), imm("int64", 1));
    assert!(same(x.cast(PrimType::int(64)) + 1, &sum));

    // A void value cannot be cast, with the C++ error.
    let void = var("v", PrimType::void());
    let Err(error) = PrimExpr::try_from(void.cast(PrimType::int(32))) else {
        panic!("a void value was cast")
    };
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error.message().contains("void sentinel type"), "{}", error.message());
}

#[test]
fn arithmetic_operators() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));

    // Promotion: int32 + float32 is float32, and constants fold.
    assert_eq!(dtype(&op::add(&x, 1.0f32).unwrap()), "float32");
    assert_eq!(dtype(&op::add(&x, IntImm::int64(1)).unwrap()), "int64");
    macro_rules! folded {
        ($op:path, $a:expr, $b:expr) => {
            int_value($op($a, $b).unwrap())
        };
    }
    assert_eq!(folded!(op::add, 7, 2), 9);
    assert_eq!(folded!(op::sub, 7, 2), 5);
    assert_eq!(folded!(op::mul, 7, 2), 14);
    assert_eq!(folded!(op::div, -7, 2), -3);
    assert_eq!(folded!(op::truncdiv, -7, 2), -3);
    assert_eq!(folded!(op::truncmod, -7, 2), -1);
    assert_eq!(folded!(op::floordiv, -7, 2), -4);
    assert_eq!(folded!(op::floormod, -7, 2), 1);
    assert_eq!(folded!(op::indexdiv, 7, 2), 3);
    assert_eq!(folded!(op::indexmod, 7, 2), 1);
    assert_eq!(folded!(op::ceildiv, 7, 2), 4);
    assert_eq!(folded!(op::min, 7, 2), 2);
    assert_eq!(folded!(op::max, 7, 2), 7);
    assert_eq!(folded!(op::bitwise_and, 6, 3), 2);
    assert_eq!(folded!(op::bitwise_or, 6, 3), 7);
    assert_eq!(folded!(op::bitwise_xor, 6, 3), 5);
    assert_eq!(folded!(op::shift_left, 1, 3), 8);
    assert_eq!(folded!(op::shift_right, 8, 3), 1);
    let not = op::bitwise_not(0).unwrap();
    assert!(not.try_cast::<prim::BitwiseNot>().is_ok(), "the node, not folded");

    // On a variable they build the nodes.
    let floordiv = op::floordiv(&x, 2).unwrap();
    assert!(floordiv.try_cast::<prim::FloorDiv>().is_ok());
    assert_eq!(dtype(&op::pow(2.0f32, 3.0f32).unwrap()), "float32");
}

#[test]
fn comparison_and_logic_operators() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let one = PrimExpr::from(1);
    for cmp in [op::eq, op::ne, op::lt, op::le, op::gt, op::ge] {
        assert_eq!(dtype(&cmp(PrimExpr::from(&x), one.clone()).unwrap()), "bool");
    }
    assert!(op::lt(&x, &one).unwrap().try_cast::<prim::LT>().is_ok());

    let conds = [op::lt(&x, &one).unwrap(), op::gt(&x, -5).unwrap()];
    assert!(op::all(conds.clone()).unwrap().try_cast::<prim::And>().is_ok());
    assert!(op::any(conds.clone()).unwrap().try_cast::<prim::Or>().is_ok());
    assert!(structural_equal(&op::all([&conds[0]]).unwrap(), &conds[0]));
    assert!(op::any(Vec::<PrimExpr>::new()).is_err());
    assert!(structural_equal(
        &op::logical_and(&conds[0], &conds[1]).unwrap(),
        &op::all(conds.clone()).unwrap()
    ));

    let choice = op::if_then_else(&conds[0], &x, &one).unwrap();
    assert_eq!(dtype(&choice), "int32");
    assert_eq!(dtype(&op::likely(&conds[0]).unwrap()), "bool");
    assert_eq!(dtype(&op::assume(&conds[0]).unwrap()), "bool");
}

#[test]
fn casts_and_conversions() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    assert_eq!(dtype(&op::cast(&x, PrimType::float(16)).unwrap()), "float16");
    assert_eq!(dtype(&op::convert(5i64).unwrap()), "int32");
    assert_eq!(dtype(&op::convert(2.5f64).unwrap()), "float32");
    assert!(structural_equal(&op::convert(Expr::from(&x)).unwrap(), &x));
}

#[test]
fn math_operators() {
    let _serial = serial();
    let f = var("f", PrimType::float(32));
    for unary in [op::ceil, op::floor, op::round, op::nearbyint, op::trunc] {
        assert_eq!(dtype(&unary(PrimExpr::from(&f)).unwrap()), "float32");
    }
    for predicate in [op::isnan, op::isfinite, op::isinf] {
        assert_eq!(dtype(&predicate(PrimExpr::from(&f)).unwrap()), "bool");
    }
    assert_eq!(int_value(op::abs(-3).unwrap()), 3);
    assert_eq!(int_value(op::max_value(PrimType::int(8)).unwrap()), 127);
    assert_eq!(int_value(op::min_value(PrimType::int(8)).unwrap()), -128);
    let inf: FloatImm = op::infinity(PrimType::float(32)).unwrap().try_cast().unwrap();
    assert!(inf.value.is_infinite());
}

/// `make_const` builds the constant of the kind of its type, broadcast over the lanes of a vector.
#[test]
fn make_const() {
    let _serial = serial();
    let seven = op::make_const(PrimType::int(32), 7).unwrap();
    assert_eq!(dtype(&seven), "int32");
    assert_eq!(int_value(seven), 7);
    assert_eq!(int_value(op::make_const(PrimType::bool(), 1).unwrap()), 1);
    assert_eq!(int_value(op::make_const(PrimType::uint(8), 255).unwrap()), 255);
    assert!(op::make_const(PrimType::uint(8), -1).is_err());
    assert!(op::make_const(PrimType::int(8), 128).is_err());
    for ty in [PrimType::float(16), PrimType::bfloat(16), PrimType::float(64)] {
        let two: FloatImm = op::make_const(ty.clone(), 2).unwrap().try_cast().unwrap();
        assert_eq!(two.value, 2.0);
        assert!(structural_equal(&two.ty, &ty));
    }
    // A float value too, as C++ `MakeConst(ty, double)`.
    let half: FloatImm = op::make_const(PrimType::float(32), 0.5).unwrap().try_cast().unwrap();
    assert_eq!(half.value, 0.5);
    assert_eq!(int_value(op::make_const(PrimType::int(32), 2.0).unwrap()), 2);
    assert!(op::make_const(PrimType::void(), 0).is_err());

    let vector = op::make_const(PrimType::int(32).with_lanes(4).unwrap(), 1).unwrap();
    assert_eq!(dtype(&vector), "int32x4");
    let broadcast: prim::Broadcast = vector.try_cast().unwrap();
    assert_eq!(int_value(broadcast.value.clone()), 1);
    assert_eq!(int_value(broadcast.lanes.clone()), 4);

    let scalable = PrimType::int(32).with_scalable_lanes(4).unwrap();
    let vector = op::make_const(scalable.clone(), 1).unwrap();
    assert!(structural_equal(&vector.ty, &scalable));
    let broadcast: prim::Broadcast = vector.try_cast().unwrap();
    let lanes = broadcast.lanes.downcast_ref::<prim::MulObj>().expect("vscale * 4");
    assert_eq!(int_value(lanes.b.clone()), 4);
}
