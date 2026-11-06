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

//! Tests of `ir::stmt`.

use tvm_tirx::ir::*;
use tvm_tirx::tvm_ffi::{Any, Array, Map, String as FfiString};

use super::{int32, int32_var, op, seq, tensor_var};
use crate::common::{agree, native, serial, structural_equal};

#[test]
fn statements() {
    let _serial = serial();
    let x = int32_var("x");
    let y = int32_var("y");
    let bind = Bind::new(&y, op("prim._OpAdd", &x, int32(1)), UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&bind.var, &y));
    let evaluate = Evaluate::new(&x, UNKNOWN_LOC);
    let ret = Return::new(&x, UNKNOWN_LOC);
    assert!(structural_equal(&ret.value, &x));

    let body = seq([Stmt::from(&bind), Stmt::from(&evaluate)]);
    assert_eq!(body.seq.len(), 2);
    let nested = SeqStmt::new([Stmt::from(&body), Stmt::from(&ret)], UNKNOWN_LOC);
    assert_eq!(nested.seq.len(), 3);

    let kind = ForKind::Unrolled;
    let annotations: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let for_ = For::new(
        &x,
        int32(0),
        int32(16),
        int32(1),
        kind,
        &evaluate,
        Some(&annotations),
        UNKNOWN_LOC,
    )
    .unwrap();
    assert_eq!(for_.kind, kind as i32);
    assert_eq!(for_.body.seq.len(), 1);
    let less = op("prim._OpLT", &x, int32(4));
    let while_ = While::new(&less, &body, UNKNOWN_LOC).unwrap();
    assert_eq!(while_.body.seq.len(), 2);
    let if_ = If::new(&less, &evaluate, Some(&Stmt::from(&ret)), UNKNOWN_LOC).unwrap();
    assert!(if_.else_case.is_some());
    assert!(If::new(&less, &evaluate, None, UNKNOWN_LOC).unwrap().else_case.is_none());

    let message = [StringImm::new("boom", UNKNOWN_LOC)];
    let kind_name = StringImm::new("RuntimeError", UNKNOWN_LOC);
    let assert_ = AssertStmt::new(&less, &kind_name, &message, UNKNOWN_LOC).unwrap();
    assert_eq!(assert_.message_parts.len(), 1);

    let a = tensor_var("A");
    let one = FloatImm::float32(1.0);
    let store = TensorStore::new(&a, [&x], &one, UNKNOWN_LOC).unwrap();
    assert_eq!(store.indices.len(), 1);

    let attrs = DictAttrs::new(Map::new());
    let region = RegionStmt::new(
        Op::get("tirx.device_scope").unwrap(),
        Vec::<Expr>::new(),
        Vec::<Var>::new(),
        &attrs,
        &evaluate,
        Vec::<Var>::new(),
        UNKNOWN_LOC,
    )
    .unwrap();
    assert_eq!(region.body.seq.len(), 1);
}

/// A variable of the primitive type `ty`.
fn prim_var(name: &str, ty: PrimType) -> PrimVar {
    PrimVar::new(name, ty, UNKNOWN_LOC)
}

/// A `For` case: its name, variable, begin, end, step, body and annotations.
type ForCase<'a> =
    (&'a str, &'a PrimVar, PrimExpr, PrimExpr, PrimExpr, Stmt, Option<Map<FfiString, Any>>);

/// A `RegionStmt` case: its name, operator, arguments, body parameters, attributes and results.
type RegionCase<'a> = (&'a str, &'a Op, Vec<Expr>, Vec<Var>, &'a DictAttrs, Vec<Var>);

#[test]
fn statements_agree_with_cpp() {
    let _serial = serial();
    let (x, y) = (prim_var("x", PrimType::int(32)), prim_var("y", PrimType::int(32)));
    let (f, p) = (prim_var("f", PrimType::float(32)), prim_var("p", PrimType::bool()));
    let evaluate: Stmt = Evaluate::new(&x, UNKNOWN_LOC).into();
    let no_op: Stmt = Evaluate::new(int32(0), UNKNOWN_LOC).into();

    for (what, var, value) in [("Bind", &y, Expr::from(&x)), ("Bind(type)", &y, Expr::from(&f))] {
        let native_bind =
            native("ir.Bind", &[var.into(), (&value).into(), (&Location::unknown()).into()]);
        agree(what, Bind::new(var, &value, UNKNOWN_LOC), native_bind);
    }

    let kind = StringImm::new("RuntimeError", UNKNOWN_LOC);
    let parts: Array<StringImm> = [StringImm::new("failed", UNKNOWN_LOC)].into_iter().collect();
    for (what, condition) in
        [("AssertStmt", PrimExpr::from(&p)), ("AssertStmt(condition)", PrimExpr::from(&x))]
    {
        let native_assert = native(
            "ir.AssertStmt",
            &[(&condition).into(), (&kind).into(), (&parts).into(), (&Location::unknown()).into()],
        );
        agree(what, AssertStmt::new(&condition, &kind, parts.iter(), UNKNOWN_LOC), native_assert);
    }

    let (i, i64_var) = (prim_var("i", PrimType::int(32)), prim_var("j", PrimType::int(64)));
    let note: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let body_seq: Stmt = SeqStmt::new([evaluate.clone(), evaluate.clone()], UNKNOWN_LOC).into();
    let cases: Vec<ForCase> = vec![
        ("For", &i, int32(0), int32(8), int32(1), evaluate.clone(), None),
        ("For(annotations)", &i, int32(0), int32(8), int32(1), body_seq.clone(), Some(note)),
        ("For(no-op body)", &i, int32(0), int32(8), int32(1), no_op.clone(), None),
        (
            "For(int64 var, int32 bounds)",
            &i64_var,
            int32(0),
            int32(8),
            int32(2),
            evaluate.clone(),
            None,
        ),
        (
            "For(int64 bound literal)",
            &i,
            IntImm::int64(0).into(),
            IntImm::int64(8).into(),
            int32(1),
            evaluate.clone(),
            None,
        ),
        (
            "For(literal out of range)",
            &i,
            int32(0),
            IntImm::int64(1 << 40).into(),
            int32(1),
            evaluate.clone(),
            None,
        ),
        (
            "For(uint8 literal)",
            &i,
            IntImm::new(PrimType::uint(8), 3, UNKNOWN_LOC).unwrap().into(),
            int32(8),
            int32(1),
            evaluate.clone(),
            None,
        ),
        (
            "For(narrower var)",
            &i,
            int32(0),
            PrimExpr::from(&i64_var),
            int32(1),
            evaluate.clone(),
            None,
        ),
        ("For(float var)", &f, int32(0), int32(8), int32(1), evaluate.clone(), None),
        ("For(float bound)", &i, int32(0), PrimExpr::from(&f), int32(1), evaluate.clone(), None),
        (
            "For(mismatched end)",
            &i64_var,
            IntImm::int64(0).into(),
            PrimExpr::from(&x),
            IntImm::int64(1).into(),
            evaluate.clone(),
            None,
        ),
    ];
    for (what, var, begin, end, step, body, annotations) in cases {
        let kind = ForKind::Parallel as i32;
        let native_for = native(
            "ir.For",
            &[
                var.into(),
                (&begin).into(),
                (&end).into(),
                (&step).into(),
                (&kind).into(),
                (&body).into(),
                (&annotations).into(),
                (&Location::unknown()).into(),
            ],
        );
        let rust = For::new(
            var,
            &begin,
            &end,
            &step,
            ForKind::Parallel,
            &body,
            annotations.as_ref(),
            UNKNOWN_LOC,
        );
        agree(what, rust, native_for);
    }

    let v4 = prim_var("v", PrimType::int(32).with_lanes(4).unwrap());
    for (what, condition) in
        [("While", PrimExpr::from(&p)), ("While(vector condition)", PrimExpr::from(&v4))]
    {
        let native_while = native(
            "ir.While",
            &[(&condition).into(), (&evaluate).into(), (&Location::unknown()).into()],
        );
        agree(what, While::new(&condition, &evaluate, UNKNOWN_LOC), native_while);
    }

    let a = tensor_var("A");
    let (one, index): (PrimExpr, PrimExpr) = (FloatImm::float32(1.0).into(), x.clone().into());
    let stores: Vec<(&str, Expr, Vec<PrimExpr>, PrimExpr)> = vec![
        ("TensorStore", Expr::from(&a), vec![index.clone()], one.clone()),
        ("TensorStore(rank)", Expr::from(&a), vec![index.clone(), index.clone()], one.clone()),
        ("TensorStore(value type)", Expr::from(&a), vec![index.clone()], index.clone()),
        ("TensorStore(no tensor)", Expr::from(&x), vec![index.clone()], one.clone()),
    ];
    for (what, dest, indices, value) in stores {
        let array: Array<PrimExpr> = indices.iter().cloned().collect();
        let native_store = native(
            "ir.TensorStore",
            &[(&dest).into(), (&array).into(), (&value).into(), (&Location::unknown()).into()],
        );
        agree(what, TensorStore::new(&dest, indices, &value, UNKNOWN_LOC), native_store);
    }

    let nested: Stmt =
        SeqStmt::new([evaluate.clone(), no_op.clone(), evaluate.clone()], UNKNOWN_LOC).into();
    let sequences: Vec<(&str, Vec<Stmt>)> = vec![
        ("SeqStmt", vec![evaluate.clone(), evaluate.clone()]),
        ("SeqStmt(nested and no-ops)", vec![nested.clone(), no_op.clone(), evaluate.clone()]),
        ("SeqStmt(empty)", vec![]),
    ];
    for (what, stmts) in sequences {
        let array: Array<Stmt> = stmts.iter().cloned().collect();
        let native_seq = native("ir.SeqStmt", &[(&array).into(), (&Location::unknown()).into()]);
        agree(what, Ok(SeqStmt::new(stmts, UNKNOWN_LOC)), native_seq);
    }

    let ifs: Vec<(&str, PrimExpr, Stmt, Option<Stmt>)> = vec![
        ("If", PrimExpr::from(&p), evaluate.clone(), None),
        ("If(else)", PrimExpr::from(&p), nested.clone(), Some(no_op.clone())),
        ("If(int condition)", PrimExpr::from(&x), evaluate.clone(), Some(evaluate.clone())),
    ];
    for (what, condition, then_case, else_case) in ifs {
        let native_if = native(
            "ir.If",
            &[
                (&condition).into(),
                (&then_case).into(),
                (&else_case).into(),
                (&Location::unknown()).into(),
            ],
        );
        agree(what, If::new(&condition, &then_case, else_case.as_ref(), UNKNOWN_LOC), native_if);
    }

    let native_evaluate =
        native("ir.Evaluate", &[(&Expr::from(&x)).into(), (&Location::unknown()).into()]);
    agree("Evaluate", Ok(Evaluate::new(&x, UNKNOWN_LOC)), native_evaluate);
    let native_return =
        native("ir.Return", &[(&Expr::from(&x)).into(), (&Location::unknown()).into()]);
    agree("Return", Ok(Return::new(&x, UNKNOWN_LOC)), native_return);
}

#[test]
fn regions_agree_with_cpp() {
    let _serial = serial();
    let launch = Op::get("tirx.launch_thread").unwrap();
    let compute = Op::get("tirx.compute_scope").unwrap();
    let exp = Op::get("prim.exp").unwrap();
    let tag = |text: &str| -> Expr { StringImm::new(text, UNKNOWN_LOC).into() };
    let int = |value: i64| -> Expr { int32(value).into() };
    let (t, t64, r): (Var, Var, Var) = (
        prim_var("t", PrimType::int(32)).into(),
        prim_var("t64", PrimType::int(64)).into(),
        prim_var("r", PrimType::int(32)).into(),
    );
    let body: Stmt = Evaluate::new(&t, UNKNOWN_LOC).into();
    let no_attrs = DictAttrs::new(Map::new());
    let note: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let attrs = DictAttrs::new(note);
    let cases: Vec<RegionCase> = vec![
        (
            "RegionStmt",
            &launch,
            vec![tag("threadIdx.x"), int(8)],
            vec![t.clone()],
            &no_attrs,
            vec![],
        ),
        (
            "RegionStmt(parameter count)",
            &launch,
            vec![tag("threadIdx.x"), int(8)],
            vec![],
            &no_attrs,
            vec![],
        ),
        (
            "RegionStmt(parameter type)",
            &launch,
            vec![tag("threadIdx.x"), int(8)],
            vec![t64.clone()],
            &no_attrs,
            vec![],
        ),
        (
            "RegionStmt(results)",
            &launch,
            vec![tag("threadIdx.x"), int(8)],
            vec![t.clone()],
            &no_attrs,
            vec![r.clone()],
        ),
        (
            "RegionStmt(repeated definition)",
            &launch,
            vec![tag("threadIdx.x"), int(8)],
            vec![t.clone()],
            &no_attrs,
            vec![t.clone()],
        ),
        (
            "RegionStmt(empty tag)",
            &launch,
            vec![tag(""), int(8)],
            vec![t.clone()],
            &no_attrs,
            vec![],
        ),
        ("RegionStmt(signature)", &launch, vec![int(8)], vec![t.clone()], &no_attrs, vec![]),
        ("RegionStmt(no body parameters)", &compute, vec![tag("work")], vec![], &no_attrs, vec![]),
        ("RegionStmt(attrs)", &compute, vec![tag("work")], vec![], &attrs, vec![]),
        ("RegionStmt(not a region)", &exp, vec![int(1)], vec![], &no_attrs, vec![]),
    ];
    for (what, op, args, params, attrs, results) in cases {
        let (args_array, params_array, results_array): (Array<Expr>, Array<Var>, Array<Var>) = (
            args.iter().cloned().collect(),
            params.iter().cloned().collect(),
            results.iter().cloned().collect(),
        );
        let native_region = native(
            "ir.RegionStmt",
            &[
                op.into(),
                (&args_array).into(),
                (&params_array).into(),
                attrs.into(),
                (&body).into(),
                (&results_array).into(),
                (&Location::unknown()).into(),
            ],
        );
        let rust = RegionStmt::new(op, args, params, attrs, &body, results, UNKNOWN_LOC);
        agree(what, rust, native_region);
    }
}
