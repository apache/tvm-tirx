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

//! Tests of `ir::expr`.

use tvm_tirx::ir::prim::Broadcast;
use tvm_tirx::ir::*;
use tvm_tirx::tvm_ffi::{
    Any, AnyView, Array, BigInt, DLDataType, Function, ObjectRefCore, Result, String as FfiString,
    TYPE_ERROR,
};

use super::{int32, int32_var, op, parse_dtype, tensor_var};
use crate::common::{agree, agree_fresh, agree_kind, native, serial, structural_equal};

#[test]
fn expressions() {
    let _serial = serial();
    let x = int32_var("x");
    assert_eq!(x.name.as_str(), "x");
    assert_eq!(GlobalVar::new("main", UNKNOWN_LOC).name_hint.as_str(), "main");
    assert_eq!(StringImm::new("hi", UNKNOWN_LOC).value.as_str(), "hi");
    assert_eq!(FloatImm::float32(1.5).value, 1.5);
    assert!(IntImm::new(PrimType::uint(8), 256, UNKNOWN_LOC).is_err());
    assert_eq!(DataTypeImm::new(dtype::int(8), UNKNOWN_LOC).value, dtype::int(8));
    let i32_ty = PrimType::int(32);
    let constant = GenericConst::new(&i32_ty, Any::from(i32_ty.clone()), UNKNOWN_LOC).unwrap();
    assert!(GenericConst::new(&i32_ty, Any::from(1i64), UNKNOWN_LOC).is_err());
    assert!(structural_equal(&constant.ty, &i32_ty));

    let tuple = Tuple::new([PrimExpr::from(&x), int32(1)], UNKNOWN_LOC);
    let item = TupleGetItem::new(&tuple, 1, UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&item.ty, &i32_ty));
    assert!(TupleGetItem::new(&tuple, 2, UNKNOWN_LOC).is_err());
    let subscripted =
        Expr::from(&tuple).subscript([Subscript::Index(int32(0).into())], UNKNOWN_LOC).unwrap();
    assert!(Expr::try_from(subscripted).is_ok());

    let a = tensor_var("A");
    let load = TensorLoad::new(&a, [&x], UNKNOWN_LOC).unwrap();
    assert_eq!(load.indices.len(), 1);
    let slice = Expr::from(&a)
        .subscript(
            [Subscript::Slice(Some(int32(0).into()), Some(int32(4).into()), None)],
            UNKNOWN_LOC,
        )
        .unwrap();
    assert!(Expr::try_from(slice).is_ok());

    let exp = Op::get("prim.exp").unwrap();
    let one = FloatImm::float32(1.0);
    let call = Call::new(None, &exp, [&one], None, Vec::<Type>::new(), UNKNOWN_LOC).unwrap();
    call.validate().unwrap();
    assert!(structural_equal(&call.reinfer_type().unwrap(), &call.ty));

    let lambda = LambdaExpr::new([&x], op("prim._OpAdd", &x, int32(1)), UNKNOWN_LOC);
    let applied = PrimExpr::try_from(lambda.apply([int32(2)]).unwrap()).unwrap();
    let analyzer = tvm_tirx::sym::Analyzer::new();
    assert!(analyzer.can_prove_equal(applied, int32(3)).unwrap());
}

#[test]
fn ranges() {
    let _serial = serial();
    let range = Range::new(int32(0), int32(8), UNKNOWN_LOC).unwrap();
    let extent: &PrimExpr = &range.extent;
    assert!(structural_equal(extent, &int32(8)));
    let range = Range::new(int32(2), int32(8), UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&range.extent, &int32(6)));
    let range = Range::from_min_extent(int32(2), int32(8), UNKNOWN_LOC).unwrap();
    assert!(structural_equal(&range.extent, &int32(8)));
    let extent: PrimExpr = Range::get_extent_from_begin_end(int32(2), int32(8)).unwrap();
    assert!(structural_equal(&extent, &int32(6)));
}

#[test]
fn constants_agree_with_cpp() {
    let _serial = serial();
    let float_imm = |name: &str, value: f64| {
        let native = native(
            "ir.FloatImm",
            &[(&parse_dtype(name)).into(), (&value).into(), (&Location::unknown()).into()],
        );
        let what = format!("FloatImm({name}, {value})");
        agree(
            &what,
            PrimType::from_dtype(parse_dtype(name))
                .and_then(|ty| FloatImm::new(ty, value, UNKNOWN_LOC)),
            native,
        );
    };
    for (name, value) in [
        ("float32", 1.5),
        ("float32", f64::INFINITY),
        ("float32", f64::NAN),
        ("float32", 1e39),
        ("float32", -1e39),
        ("float16", 1e9),
        ("float16", -65504.0),
        ("bfloat16", 1e39),
        ("float64", 1e300),
        ("float8_e4m3fn", 500.0),
        ("float8_e4m3fnuz", -1.0),
        ("float8_e5m2", -60000.0),
        ("float8_e8m0fnu", 1e39),
        ("float6_e2m3fn", 8.0),
        ("float4_e2m1fn", -7.0),
        ("int32", 1.0),
        ("float32x4", 1.0),
    ] {
        float_imm(name, value);
    }

    let int_imm = |name: &str, value: BigInt| {
        let native = native(
            "ir.IntImm",
            &[(&parse_dtype(name)).into(), (&value).into(), (&Location::unknown()).into()],
        );
        let what = format!("IntImm({name}, {value})");
        agree(
            &what,
            PrimType::from_dtype(parse_dtype(name))
                .and_then(|ty| IntImm::new(ty, value.clone(), UNKNOWN_LOC)),
            native,
        );
    };
    let big = |text: &str| -> BigInt { text.parse().unwrap() };
    for (name, value) in [
        ("int32", big("7")),
        ("int8", big("300")),
        ("int8", big("-129")),
        ("int64", big("-9223372036854775808")),
        ("int128", big("170141183460469231731687303715884105727")),
        ("int128", big("170141183460469231731687303715884105728")),
        ("uint8", big("256")),
        ("uint8", big("-1")),
        ("uint64", big("18446744073709551615")),
        ("uint128", big("340282366920938463463374607431768211456")),
        ("bool", big("1")),
        ("bool", big("2")),
        ("int1", big("-1")),
        ("int32x4", big("1")),
        ("float32", big("1")),
    ] {
        int_imm(name, value);
    }

    let i32_ty = PrimType::int(32);
    let missing: Type = MissingType::new().into();
    let generic = |what: &str, ty: &Type, value: Any| {
        let native =
            native("ir.GenericConst", &[ty.into(), (&value).into(), (&Location::unknown()).into()]);
        agree(what, GenericConst::new(ty, value.clone(), UNKNOWN_LOC), native);
    };
    let i32_ty_as_type: Type = i32_ty.clone().into();
    generic("GenericConst(object)", &i32_ty_as_type, Any::from(i32_ty.clone()));
    generic("GenericConst(int)", &i32_ty_as_type, Any::from(1i64));
    generic("GenericConst(bool)", &i32_ty_as_type, Any::from(true));
    generic("GenericConst(float)", &i32_ty_as_type, Any::from(1.0f64));
    generic("GenericConst(string)", &i32_ty_as_type, Any::from(FfiString::from("s")));
    generic("GenericConst(missing type)", &missing, Any::from(i32_ty.clone()));

    let int8 = dtype::int(8);
    let native_dtype = native("ir.DataTypeImm", &[(&int8).into(), (&Location::unknown()).into()]);
    agree("DataTypeImm", Ok(DataTypeImm::new(int8, UNKNOWN_LOC)), native_dtype);
    let text = FfiString::from("hi");
    let native_string = native("ir.StringImm", &[(&text).into(), (&Location::unknown()).into()]);
    agree("StringImm", Ok(StringImm::new("hi", UNKNOWN_LOC)), native_string);
}

/// The constants of the builtin types, and the Rust values that convert to them, are the native
/// constants of those types.
#[test]
fn builtin_constants() {
    let _serial = serial();
    let int = |dtype: DLDataType, value: i64| {
        native(
            "ir.IntImm",
            &[(&dtype).into(), (&BigInt::from(value)).into(), (&Location::unknown()).into()],
        )
    };
    let float = |dtype: DLDataType, value: f64| {
        native("ir.FloatImm", &[(&dtype).into(), (&value).into(), (&Location::unknown()).into()])
    };
    agree("IntImm::int32", Ok(IntImm::int32(7)), int(dtype::int(32), 7));
    agree("IntImm::int32(min)", Ok(IntImm::int32(i32::MIN)), int(dtype::int(32), i32::MIN as i64));
    agree("IntImm::int64", Ok(IntImm::int64(1 << 40)), int(dtype::int(64), 1 << 40));
    agree("IntImm::bool", Ok(IntImm::bool(true)), int(dtype::bool(), 1));
    agree("FloatImm::float32", Ok(FloatImm::float32(1.5)), float(dtype::float(32), 1.5));
    agree(
        "FloatImm::float32(max)",
        Ok(FloatImm::float32(f32::MAX)),
        float(dtype::float(32), f32::MAX as f64),
    );
    agree("FloatImm::float64", Ok(FloatImm::float64(1e300)), float(dtype::float(64), 1e300));

    agree("Expr::from(i32)", Ok(Expr::from(3i32)), int(dtype::int(32), 3));
    agree("Expr::from(i64)", Ok(Expr::from(3i64)), int(dtype::int(64), 3));
    agree("Expr::from(bool)", Ok(Expr::from(false)), int(dtype::bool(), 0));
    agree("Expr::from(f32)", Ok(Expr::from(0.5f32)), float(dtype::float(32), 0.5));
    agree("Expr::from(f64)", Ok(Expr::from(0.5f64)), float(dtype::float(64), 0.5));
    let folded: IntImm = op("prim._OpAdd", 1i32, 2i32).downcast().unwrap();
    assert_eq!(folded.value, BigInt::from(3));
}

#[test]
fn shared_nodes_are_the_native_ones() {
    let _serial = serial();
    let unknown: Location = native("ir.UnknownLoc", &[]).unwrap().try_into().unwrap();
    assert!(UnknownLoc::new().same_as(&unknown));
    let missing: Type = native("ir.MissingType", &[]).unwrap().try_into().unwrap();
    assert!(MissingType::new().same_as(&missing));
    let int32_dtype = dtype::int(32);
    let int32_ty: Type =
        native("ir.PrimType", &[(&int32_dtype).into()]).unwrap().try_into().unwrap();
    assert!(PrimType::from_dtype(int32_dtype).unwrap().same_as(&int32_ty));
}

#[test]
fn expressions_agree_with_cpp() {
    let _serial = serial();
    let (x, n) = (int32_var("x"), int32_var("n"));
    let untyped = Var::new("u", MissingType::new(), UNKNOWN_LOC);
    let i32_ty: Type = PrimType::int(32).into();

    let name = FfiString::from("main");
    let native_global = native("ir.GlobalVar", &[(&name).into(), (&Location::unknown()).into()]);
    agree_fresh("GlobalVar", Ok(GlobalVar::new("main", UNKNOWN_LOC)), native_global);
    let native_var = native(
        "ir.Var",
        &[(&FfiString::from("x")).into(), (&i32_ty).into(), (&Location::unknown()).into()],
    );
    agree_fresh("Var", Ok(Var::new("x", i32_ty, UNKNOWN_LOC)), native_var);
    let missing: Type = MissingType::new().into();
    let native_untyped = native(
        "ir.Var",
        &[(&FfiString::from("u")).into(), (&missing).into(), (&Location::unknown()).into()],
    );
    agree_fresh("Var(untyped)", Ok(Var::new("u", MissingType::new(), UNKNOWN_LOC)), native_untyped);

    for (what, fields) in [
        ("Tuple(typed)", vec![Expr::from(&x), int32(1).into()]),
        ("Tuple(untyped)", vec![Expr::from(&x), Expr::from(&untyped)]),
        ("Tuple(empty)", vec![]),
    ] {
        let array: Array<Expr> = fields.iter().cloned().collect();
        let native_tuple = native("ir.Tuple", &[(&array).into(), (&Location::unknown()).into()]);
        agree(what, Ok(Tuple::new(fields, UNKNOWN_LOC)), native_tuple);
    }

    let tuple: Expr = Tuple::new([PrimExpr::from(&x), int32(1)], UNKNOWN_LOC).into();
    for (what, value, index) in [
        ("TupleGetItem", tuple.clone(), 1),
        ("TupleGetItem(out of bounds)", tuple.clone(), 2),
        ("TupleGetItem(negative)", tuple.clone(), -1),
        ("TupleGetItem(untyped)", Expr::from(&untyped), 0),
    ] {
        let native_item = native(
            "ir.TupleGetItem",
            &[(&value).into(), (&index).into(), (&Location::unknown()).into()],
        );
        agree(what, TupleGetItem::new(&value, index, UNKNOWN_LOC), native_item);
    }

    let a = tensor_var("A");
    let load = |what: &str, source: Expr, indices: Vec<PrimExpr>| {
        let array: Array<PrimExpr> = indices.iter().cloned().collect();
        let native_load = native(
            "ir.TensorLoad",
            &[(&source).into(), (&array).into(), (&Location::unknown()).into()],
        );
        agree(what, TensorLoad::new(&source, indices, UNKNOWN_LOC), native_load);
    };
    load("TensorLoad", Expr::from(&a), vec![PrimExpr::from(&x)]);
    load("TensorLoad(rank)", Expr::from(&a), vec![PrimExpr::from(&x), PrimExpr::from(&x)]);
    load("TensorLoad(no tensor)", Expr::from(&x), vec![PrimExpr::from(&x)]);
    let text: Expr = StringImm::new("s", UNKNOWN_LOC).into();
    let array: Array<Expr> = [text.clone()].into_iter().collect();
    let native_load = native(
        "ir.TensorLoad",
        &[(&Expr::from(&a)).into(), (&array).into(), (&Location::unknown()).into()],
    );
    // The type system rejects a string index; its check fails as the native conversion does.
    let rust_load =
        PrimExpr::try_from(text).and_then(|index| TensorLoad::new(&a, [index], UNKNOWN_LOC));
    agree_kind("TensorLoad(index)", rust_load, native_load);

    let vars: Array<Var> = [Var::from(&x)].into_iter().collect();
    let body = op("prim._OpAdd", &x, int32(1));
    let native_lambda =
        native("ir.LambdaExpr", &[(&vars).into(), (&body).into(), (&Location::unknown()).into()]);
    agree("LambdaExpr", Ok(LambdaExpr::new([&x], &body, UNKNOWN_LOC)), native_lambda);

    let x_plus_4 = op("prim._OpAdd", &x, int32(4));
    let four_plus_x = op("prim._OpAdd", int32(4), &x);
    for (what, begin, end) in [
        ("Range(x, x + 4)", PrimExpr::from(&x), x_plus_4.clone()),
        ("Range(x, 4 + x)", PrimExpr::from(&x), four_plus_x),
        ("Range(0, n)", int32(0), PrimExpr::from(&n)),
        ("Range(x, n)", PrimExpr::from(&x), PrimExpr::from(&n)),
    ] {
        let native_range =
            native("ir.Range", &[(&begin).into(), (&end).into(), (&Location::unknown()).into()]);
        agree(what, Range::new(&begin, &end, UNKNOWN_LOC), native_range);
    }
    // The native constructor spells `[0, n)` as `Range(n)`; Rust gives the begin.
    let native_range =
        native("ir.Range", &[(&n).into(), (&None::<Expr>).into(), (&Location::unknown()).into()]);
    agree("Range(n)", Range::new(int32(0), &n, UNKNOWN_LOC), native_range);
    let native_min_extent = native(
        "ir.RangeFromMinExtent",
        &[(&Expr::from(&x)).into(), (&int32(4)).into(), (&Location::unknown()).into()],
    );
    agree(
        "Range::from_min_extent",
        Range::from_min_extent(&x, int32(4), UNKNOWN_LOC),
        native_min_extent,
    );
    let native_extent =
        native("ir.RangeGetExtentFromBeginEnd", &[(&Expr::from(&x)).into(), (&x_plus_4).into()]);
    agree(
        "Range::get_extent_from_begin_end",
        Range::get_extent_from_begin_end(&x, &x_plus_4),
        native_extent,
    );
}

/// A test operator `name` with the attributes `attrs`, registered once per test binary.
fn test_op(name: &str, attrs: &[(&str, Any)]) -> Op {
    if let Ok(op) = Op::get(name) {
        return op;
    }
    Op::register(name, "").unwrap();
    let op = Op::get(name).unwrap();
    let set_attr = tvm_ffi::reflection::get_method("ir.Op", "_set_attr").unwrap();
    for (key, value) in attrs {
        let key = FfiString::from(*key);
        set_attr
            .call_packed(&[(&op).into(), (&key).into(), value.into(), (&false).into()])
            .unwrap();
    }
    op
}

#[test]
fn calls_agree_with_cpp() {
    let _serial = serial();
    let x = int32_var("x");
    let f32_ty: Type = PrimType::float(32).into();
    let one: Expr = FloatImm::float32(1.0).into();
    let i16_ty: Type = PrimType::int(16).into();
    // A callback captures thread-safe state: a typed value, converted on each call.
    fn returns<T: Clone + Into<Any> + Send + Sync + 'static>(value: T) -> Function {
        Function::from_packed(move |_args: &[AnyView]| -> Result<Any> { Ok(value.clone().into()) })
    }
    let packed_infer =
        test_op("rust_test.packed_infer", &[("FInferType", returns(i16_ty.clone()).into())]);
    let packed_mistyped =
        test_op("rust_test.packed_mistyped", &[("FInferType", returns(5i64).into())]);
    let fixed =
        test_op("rust_test.fixed_return", &[("TFixedReturnType", Any::from(f32_ty.clone()))]);
    let plain = test_op("rust_test.no_inference", &[]);

    let callee_ty: Type = FuncType::new([f32_ty.clone()], i16_ty.clone(), UNKNOWN_LOC).into();
    let callee = Var::new("f", callee_ty, UNKNOWN_LOC);
    let cases: Vec<(&str, Option<Type>, Expr, Vec<Expr>)> = vec![
        (
            "Call(typed)",
            Some(f32_ty.clone()),
            Op::get("prim.exp").unwrap().into(),
            vec![one.clone()],
        ),
        ("Call(FInferType)", None, Op::get("prim.exp").unwrap().into(), vec![one.clone()]),
        ("Call(packed FInferType)", None, packed_infer.into(), vec![Expr::from(&x)]),
        ("Call(mistyped packed FInferType)", None, packed_mistyped.into(), vec![Expr::from(&x)]),
        ("Call(TFixedReturnType)", None, fixed.into(), vec![]),
        ("Call(no inference)", None, plain.into(), vec![]),
        ("Call(function callee)", None, Expr::from(&callee), vec![one.clone()]),
        ("Call(untyped callee)", None, Expr::from(&x), vec![]),
    ];
    for (what, ty, op, args) in cases {
        let array: Array<Expr> = args.iter().cloned().collect();
        let (attrs, ty_args) = (Option::<Attrs>::None, Array::<Type>::from_vec(vec![]));
        let native_call = native(
            "ir.Call",
            &[
                (&ty).into(),
                (&op).into(),
                (&array).into(),
                (&attrs).into(),
                (&ty_args).into(),
                (&Location::unknown()).into(),
            ],
        );
        let rust = Call::new(ty.as_ref(), &op, args, None, Vec::<Type>::new(), UNKNOWN_LOC);
        agree(what, rust.clone(), native_call);
        if let Ok(call) = rust {
            let native_type = native("ir.CallReinferType", &[(&call).into()]);
            agree(&format!("{what}.reinfer_type"), call.reinfer_type(), native_type);
        }
    }

    // `prim.exp` validates its argument count and type.
    let exp: Expr = Op::get("prim.exp").unwrap().into();
    for (what, args) in [
        ("Call::validate", vec![one.clone()]),
        ("Call::validate(arity)", vec![one.clone(), one.clone()]),
    ] {
        let call =
            Call::new(Some(&f32_ty), &exp, args, None, Vec::<Type>::new(), UNKNOWN_LOC).unwrap();
        let native_validate = native("ir.CallValidate", &[(&call).into()]).map(|_| ());
        match (call.validate(), native_validate) {
            (Ok(()), Ok(())) => {}
            (Err(rust), Err(native)) => {
                assert_eq!(
                    (rust.kind(), rust.message()),
                    (native.kind(), native.message()),
                    "{what}"
                )
            }
            (rust, native) => panic!("{what}: {rust:?} vs. {native:?}"),
        }
    }
}

// Typed views: `PrimExpr`, `IntExpr` and `PrimVar`.

#[test]
fn views_check_the_type() {
    let _serial = serial();
    let i = Expr::from(int32_var("i"));
    let x = Expr::from(Var::new("x", PrimType::float(32), UNKNOWN_LOC));
    let a = Expr::from(tensor_var("A"));
    let text = Expr::from(StringImm::new("s", UNKNOWN_LOC));

    // An expression is a view only when its type is one the view accepts.
    let prim = PrimExpr::try_from(&i).unwrap();
    assert_eq!(prim.dtype(), dtype::int(32));
    assert_eq!(prim.ty().dtype, dtype::int(32));
    assert!(prim.same_as(&i));
    assert_eq!(PrimExpr::try_from(&x).unwrap().dtype(), dtype::float(32));
    let error = PrimExpr::try_from(&a).err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error.message().contains("`ir.Var[ty=tirx.TensorType]` to `ir.PrimExpr`"), "{error}");
    let error = PrimExpr::try_from(&text).err().unwrap();
    assert!(error.message().contains("ir.StringImm[ty=ir.StringType]"), "{error}");

    let index = IntExpr::try_from(&i).unwrap();
    assert!(IntExpr::try_from(&x).is_err());
    assert!(IntExpr::try_from(PrimExpr::from(true)).is_err());
    // A prim node is a `PrimExpr` without a check, but a vector one is no `IntExpr`.
    let vector: PrimExpr = Broadcast::new(&prim, 4, UNKNOWN_LOC).unwrap().into();
    assert!(IntExpr::try_from(vector).is_err());

    // A view widens for free and reads as the views it refines.
    let widened: PrimExpr = index.clone().into();
    let expr: Expr = index.clone().into();
    assert!(widened.same_as(&expr));
    assert_eq!(index.dtype(), dtype::int(32));
    assert!(index.as_expr().same_as(&i));
    let borrowed: &PrimExpr = index.as_ref();
    assert!(borrowed.same_as(&i));
    // A view reads the fields of its object.
    let ty: &Type = &index.ty;
    assert_eq!(ty.downcast_ref::<PrimTypeObj>().unwrap().dtype, dtype::int(32));
}

#[test]
fn constants_and_loads_are_views() {
    let _serial = serial();
    // Constants and loads are typed by a `PrimType`, so they convert without a check.
    assert_eq!(PrimExpr::from(IntImm::int64(1)).dtype(), dtype::int(64));
    assert_eq!(PrimExpr::from(FloatImm::float32(0.5)).dtype(), dtype::float(32));
    let load = TensorLoad::new(tensor_var("A"), [int32_var("i")], UNKNOWN_LOC).unwrap();
    let indices: &Array<PrimExpr> = &load.indices;
    assert_eq!(indices.get(0).unwrap().dtype(), dtype::int(32));
    assert_eq!(PrimExpr::from(&load).dtype(), dtype::float(32));

    // A load takes `PrimExpr` indices: the type system rejects a string, whose check fails.
    let text = Expr::from(StringImm::new("s", UNKNOWN_LOC));
    let error = PrimExpr::try_from(text).err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);

    // Rust literals.
    assert_eq!(PrimExpr::from(1).dtype(), dtype::int(32));
    assert_eq!(PrimExpr::from(1i64).dtype(), dtype::int(64));
    assert_eq!(PrimExpr::from(true).dtype(), dtype::bool());
    assert_eq!(PrimExpr::from(1.5f32).dtype(), dtype::float(32));
    assert_eq!(PrimExpr::from(1.5f64).dtype(), dtype::float(64));
    assert_eq!(IntExpr::from(7).dtype(), dtype::int(32));
    assert_eq!(IntExpr::from(7i64).dtype(), dtype::int(64));
}

#[test]
fn any_converts_as_in_cpp() {
    let _serial = serial();
    // An expression is checked; a bool, an int or a float becomes a constant, as C++
    // `TypeTraits<PrimExpr>` converts it: an int that fits `int32` is one, a float is `float32`.
    let i = int32_var("i");
    assert!(PrimExpr::try_from(Any::from(i.clone())).unwrap().same_as(&i));
    assert!(PrimExpr::try_from(Any::from(tensor_var("A"))).is_err());
    assert_eq!(PrimExpr::try_from(Any::from(true)).unwrap().dtype(), dtype::bool());
    assert_eq!(PrimExpr::try_from(Any::from(3i64)).unwrap().dtype(), dtype::int(32));
    assert_eq!(PrimExpr::try_from(Any::from(1i64 << 40)).unwrap().dtype(), dtype::int(64));
    assert_eq!(PrimExpr::try_from(Any::from(0.5f64)).unwrap().dtype(), dtype::float(32));
    assert!(PrimExpr::try_from(Any::from(tvm_tirx::tvm_ffi::String::from("s"))).is_err());
    assert_eq!(IntExpr::try_from(Any::from(3i64)).unwrap().dtype(), dtype::int(32));
    assert!(IntExpr::try_from(Any::from(true)).is_err());
    assert!(IntExpr::try_from(Any::from(0.5f64)).is_err());

    // An array of views checks each element, and converts one of literals element by element.
    let mixed: Array<Any> = [Any::from(i.clone()), Any::from(2i64)].into_iter().collect();
    let views: Array<PrimExpr> = Any::from(mixed).try_into().unwrap();
    assert!(views.get(0).unwrap().same_as(&i));
    assert_eq!(views.get(1).unwrap().dtype(), dtype::int(32));
    let exprs: Array<Expr> =
        [Expr::from(i), StringImm::new("s", UNKNOWN_LOC).into()].into_iter().collect();
    assert!(Array::<PrimExpr>::try_from(Any::from(exprs)).is_err());
}

#[test]
fn operators_take_views() {
    let _serial = serial();
    let i = IntExpr::try_from(Expr::from(int32_var("i"))).unwrap();
    let j = int32_var_expr("j");
    let sum = PrimExpr::try_from(&i + &j + 1).unwrap();
    assert_eq!(sum.dtype(), dtype::int(32));
    assert_eq!(IntExpr::try_from(i.clone() * 2).unwrap().dtype(), dtype::int(32));
    // An `Expr` has no operators: the type system rejects a tensor operand, whose check fails.
    let error = PrimExpr::try_from(Expr::from(tensor_var("A"))).err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error.message().contains("ir.Var[ty=tirx.TensorType]"), "{error}");
    // `ObjectRefCore` downcasts between views as it does between node types.
    let back: IntExpr = sum.downcast().unwrap();
    assert_eq!(back.dtype(), dtype::int(32));
}

#[test]
fn prim_vars_are_views() {
    let _serial = serial();
    // `PrimVar::new` types the variable by a `PrimType`, and reads it as a `Var`.
    let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
    assert_eq!(i.name.as_str(), "i");
    assert_eq!(i.dtype(), dtype::int(32));
    assert_eq!(i.ty().dtype, dtype::int(32));

    // It widens for free into a `Var`, a `PrimExpr` and an `Expr`: the same node.
    let var: Var = i.clone().into();
    let prim: PrimExpr = (&i).into();
    let expr: Expr = i.clone().into();
    assert!(var.same_as(&i) && prim.same_as(&i) && expr.same_as(&i));
    let borrowed: &Var = i.as_ref();
    assert!(borrowed.same_as(&i));
    assert!(i.as_expr().same_as(&i));

    // A `Var` narrows only through the check of C++ `PrimVar`: `downcast_ref` borrows it in place
    // and `downcast` takes it.
    assert!(var.downcast_ref::<PrimVar>().unwrap().same_as(&i));
    let vector = Var::new("v", PrimType::float(32).with_lanes(4).unwrap(), UNKNOWN_LOC);
    assert_eq!(vector.downcast::<PrimVar>().unwrap().dtype().lanes, 4);
    let untyped = Var::new("u", MissingType::new(), UNKNOWN_LOC);
    assert!(untyped.downcast_ref::<PrimVar>().is_none());
    let error = PrimVar::try_from(&untyped).unwrap_err();
    assert_eq!(error.kind(), TYPE_ERROR);
    // C++ `TypeTraits<PrimVar>` names the variable class on both sides.
    assert_eq!(error.message(), "Cannot convert from type `ir.Var` to `ir.Var`");
    assert!(tensor_var("A").downcast::<PrimVar>().is_err());

    // It takes part in arithmetic as a `PrimExpr`.
    let sum = PrimExpr::try_from(&i * 4 + 1).unwrap();
    assert_eq!(sum.dtype(), dtype::int(32));
}

/// An `int32` variable as a `PrimExpr`.
fn int32_var_expr(name: &str) -> PrimExpr {
    PrimExpr::try_from(Expr::from(int32_var(name))).unwrap()
}
