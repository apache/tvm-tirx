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

//! Tests of `ir::prim::expr`.

use tvm_tirx::ir::prim;
use tvm_tirx::ir::{
    UNKNOWN_LOC, UnknownLoc, MissingType, Expr, PrimExpr, PrimType, PrimVar, StringImm, Type, Var,
};
use tvm_tirx::tvm_ffi::{Any, AnyCompatible, AnyView, Result};

use super::{dtype, var};
use crate::common::{agree, agree_kind, native, serial, structural_equal};

#[test]
fn node_constructors() {
    let _serial = serial();
    let (x, y) = (var("x", PrimType::int(32)), var("y", PrimType::int(32)));
    let (p, q) = (var("p", PrimType::bool()), var("q", PrimType::bool()));

    let add = prim::Add::new(&x, &y, UNKNOWN_LOC).unwrap();
    let a: &PrimExpr = &add.a;
    assert!(structural_equal(a, &x));
    assert_eq!(dtype(&add), "int32");
    assert!(prim::Add::new(&x, 1.0f32, UNKNOWN_LOC).is_err(), "no type promotion");

    let binary: Vec<Expr> = vec![
        prim::Sub::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Mul::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Div::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Mod::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::FloorDiv::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::FloorMod::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Min::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::Max::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::LShift::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::RShift::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::BitwiseAnd::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::BitwiseOr::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::BitwiseXor::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::BitwiseNot::new(&x, UNKNOWN_LOC).unwrap().into(),
    ];
    for expr in &binary {
        assert_eq!(dtype(expr), "int32");
    }
    let compare: Vec<Expr> = vec![
        prim::EQ::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::NE::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::LT::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::LE::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::GT::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::GE::new(&x, &y, UNKNOWN_LOC).unwrap().into(),
        prim::And::new(&p, &q, UNKNOWN_LOC).unwrap().into(),
        prim::Or::new(&p, &q, UNKNOWN_LOC).unwrap().into(),
        prim::Not::new(&p, UNKNOWN_LOC).unwrap().into(),
    ];
    for expr in &compare {
        assert_eq!(dtype(expr), "bool");
    }

    let cast = prim::Cast::new(PrimType::float(32), &x, UNKNOWN_LOC).unwrap();
    assert_eq!(dtype(&cast), "float32");
    let select = prim::Select::new(&p, &x, &y, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&select.true_value, &x));
    let let_ = prim::Let::new(&x, 1, &add, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&let_.var, &x));
}

/// The native constructor `name` on `args` and the location.
fn native_with_loc<T: AnyCompatible>(name: &str, args: &[&T]) -> Result<Any> {
    let loc = UnknownLoc::new();
    let mut views: Vec<AnyView> = args.iter().map(|arg| AnyView::from(*arg)).collect();
    views.push(AnyView::from(&loc));
    native(name, &views)
}

#[test]
fn binary_nodes_agree_with_cpp() {
    let _serial = serial();
    let (x, y) = (var("x", PrimType::int(32)), var("y", PrimType::int(32)));
    let (u, v4) = (var("u", PrimType::uint(8)), var("v", PrimType::int(32).with_lanes(4).unwrap()));
    let (p, q) = (var("p", PrimType::bool()), var("q", PrimType::bool()));
    let p4 = var("p4", PrimType::bool().with_lanes(4).unwrap());
    let f = var("f", PrimType::float(32));
    let text: Expr = StringImm::new("s", UNKNOWN_LOC).into();
    let operands: Vec<(&str, &PrimVar, &PrimVar)> = vec![
        ("int32", &x, &y),
        ("uint8", &u, &u),
        ("int32x4", &v4, &v4),
        ("bool", &p, &q),
        ("float32", &f, &f),
        ("int32 and float32", &x, &f),
        ("int32 and int32x4", &x, &v4),
    ];
    macro_rules! check {
        ($($node:ident),+) => {$(
            for (what, a, b) in &operands {
                let what = format!("{}({what})", stringify!($node));
                let native = native_with_loc(concat!("prim.", stringify!($node)), &[*a, *b]);
                agree(&what, prim::$node::new(*a, *b, UNKNOWN_LOC), native);
            }
            // The type system rejects a string operand; its check fails as the native one does.
            let what = format!("{}(string)", stringify!($node));
            let name = concat!("prim.", stringify!($node));
            let native = native_with_loc(name, &[&text, x.as_expr()]);
            let rust = PrimExpr::try_from(&text).and_then(|a| prim::$node::new(a, &x, UNKNOWN_LOC));
            agree_kind(&what, rust, native);
        )+};
    }
    check!(Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max);
    check!(LShift, RShift, BitwiseAnd, BitwiseOr, BitwiseXor);
    check!(EQ, NE, LT, LE, GT, GE);
    let logical: Vec<(&str, &PrimVar, &PrimVar)> = vec![
        ("bool", &p, &q),
        ("boolx4", &p4, &p4),
        ("bool and boolx4", &p, &p4),
        ("int32 and bool", &x, &p),
        ("bool and int32", &p, &x),
    ];
    for (what, a, b) in &logical {
        agree(
            &format!("And({what})"),
            prim::And::new(*a, *b, UNKNOWN_LOC),
            native_with_loc("prim.And", &[*a, *b]),
        );
        agree(
            &format!("Or({what})"),
            prim::Or::new(*a, *b, UNKNOWN_LOC),
            native_with_loc("prim.Or", &[*a, *b]),
        );
    }
    for (what, a) in [("int32", &x), ("bool", &p), ("boolx4", &p4), ("float32", &f)] {
        let native_not = native_with_loc("prim.Not", &[a]);
        agree(&format!("Not({what})"), prim::Not::new(a, UNKNOWN_LOC), native_not);
        let native_bitwise_not = native_with_loc("prim.BitwiseNot", &[a]);
        agree(
            &format!("BitwiseNot({what})"),
            prim::BitwiseNot::new(a, UNKNOWN_LOC),
            native_bitwise_not,
        );
    }
}

#[test]
fn other_nodes_agree_with_cpp() {
    let _serial = serial();
    let (x, y) = (var("x", PrimType::int(32)), var("y", PrimType::int(32)));
    let (p, p4) = (var("p", PrimType::bool()), var("p4", PrimType::bool().with_lanes(4).unwrap()));
    let (v4, w4) = (
        var("v", PrimType::int(32).with_lanes(4).unwrap()),
        var("w", PrimType::int(32).with_lanes(4).unwrap()),
    );
    let f = var("f", PrimType::float(32));

    for (what, target, value) in [
        ("Cast(int32 to float32)", PrimType::float(32), &x),
        ("Cast(int32x4 to float32x4)", PrimType::float(32).with_lanes(4).unwrap(), &v4),
        ("Cast(int32 to float32x4)", PrimType::float(32).with_lanes(4).unwrap(), &x),
    ] {
        let target_ty: Type = target.clone().into();
        let native_cast =
            native("prim.Cast", &[(&target_ty).into(), value.into(), (&UnknownLoc::new()).into()]);
        agree(what, prim::Cast::new(target, value, UNKNOWN_LOC), native_cast);
    }

    for (what, condition, t, f_) in [
        ("Select", &p, &x, &y),
        ("Select(vector)", &p4, &v4, &w4),
        ("Select(scalar condition)", &p, &v4, &w4),
        ("Select(lanes)", &p4, &x, &y),
        ("Select(condition type)", &x, &x, &y),
        ("Select(value types)", &p, &x, &f),
    ] {
        let native_select = native_with_loc("prim.Select", &[condition, t, f_]);
        agree(what, prim::Select::new(condition, t, f_, UNKNOWN_LOC), native_select);
    }

    let z = var("z", PrimType::int(32));
    let z_var = Var::from(&z);
    let tensor_ty: Type = PrimType::float(32).into();
    let tensor = Var::new("t", tensor_ty, UNKNOWN_LOC);
    let missing = Var::new("m", MissingType::new(), UNKNOWN_LOC);
    for (what, var, value) in [
        ("Let", &z_var, &y),
        ("Let(value type)", &z_var, &f),
        ("Let(var type)", &tensor, &y),
        ("Let(untyped var)", &missing, &y),
    ] {
        let body = PrimExpr::from(if what == "Let" { &z } else { &x });
        let native_let = native(
            "prim.Let",
            &[var.into(), value.into(), (&body).into(), (&UnknownLoc::new()).into()],
        );
        agree(what, prim::Let::new(var, value, &body, UNKNOWN_LOC), native_let);
    }
}
