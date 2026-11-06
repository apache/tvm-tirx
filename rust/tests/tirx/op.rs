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

//! Tests of `tirx::op`: `logaddexp`, `reinterpret` and `fast_erf_float_expr` written in Rust,
//! against the native functions they follow.

use tvm_tirx::ir::{
    Call, Expr, FloatImm, FuncType, IntImm, UnknownLoc, Op, PrimExpr, PrimType, PrimVar, PtrType,
    StringImm, Type, UNKNOWN_LOC,
};
use tvm_tirx::tirx::{fast_erf_float_expr, gpu_thread_return, logaddexp, reinterpret};
use tvm_tirx::tvm_ffi::{AnyView, DLDataTypeExt, Function, ObjectRefCore, String as FfiString};

use super::{memory_scope, var};
use crate::common::{agree, global, serial, structural_equal};

fn repr<T>(value: &T) -> String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let text: FfiString =
        global("ffi.ReprPrint").call_packed(&[AnyView::from(value)]).unwrap().try_into().unwrap();
    text.as_str().to_string()
}

fn prim(name: &str) -> PrimType {
    PrimType::from_dtype(tvm_tirx::tvm_ffi::DLDataType::try_from_str(name).unwrap()).unwrap()
}

fn prim_var(name: &str) -> PrimVar {
    PrimVar::new(&format!("v_{}", name.replace('x', "_")), prim(name), UNKNOWN_LOC)
}

fn ptr(element: &str) -> Type {
    PtrType::new(prim(element), &memory_scope()).unwrap().into()
}

#[test]
fn ops() {
    let _serial = serial();
    let a = PrimVar::new("a", PrimType::float(32), UNKNOWN_LOC);
    let b = PrimVar::new("b", PrimType::float(32), UNKNOWN_LOC);
    let sum = logaddexp(&a, &b).unwrap();
    assert!(structural_equal(&sum.ty, &PrimType::float(32)));
    let bits = reinterpret(PrimType::int(32), &a).unwrap();
    assert!(structural_equal(&bits.ty, &PrimType::int(32)));
    let ret = gpu_thread_return().unwrap();
    assert!(structural_equal(&ret.ty, &PrimType::void()));
}

#[test]
fn logaddexp_agrees_with_cpp() {
    let _serial = serial();
    let pairs: Vec<(PrimExpr, PrimExpr)> = vec![
        (prim_var("float32").into(), PrimVar::new("w", PrimType::float(32), UNKNOWN_LOC).into()),
        (prim_var("float16").into(), prim_var("float32").into()),
        (prim_var("float32").into(), IntImm::int32(2).into()),
        (prim_var("float32x4").into(), prim_var("float32").into()),
        (
            prim_var("float32").into(),
            FloatImm::new(prim("float16"), 0.5, UNKNOWN_LOC).unwrap().into(),
        ),
        (prim_var("int32").into(), PrimVar::new("j", PrimType::int(32), UNKNOWN_LOC).into()),
    ];
    for (x, y) in &pairs {
        let native =
            global("tirx._OpLogAddExp").call_tuple((x.clone(), y.clone(), UnknownLoc::new()));
        let what = format!("logaddexp({}, {})", repr(x), repr(y));
        agree(&what, logaddexp(x, y), native);
    }
}

#[test]
fn reinterpret_agrees_with_cpp() {
    let _serial = serial();
    let text: Expr = StringImm::new("data", UNKNOWN_LOC).into();
    let pointer = var("p", ptr("float32"));
    let function: Type = FuncType::new(Vec::<Type>::new(), PrimType::int(32), UNKNOWN_LOC).into();
    let cases: Vec<(Type, Expr)> = vec![
        (PrimType::int(32).into(), prim_var("float32").into()),
        (PrimType::float(32).into(), prim_var("float32").into()),
        (PrimType::int(64).into(), prim_var("float32").into()),
        (prim("float32x4").into(), prim_var("int32x4").into()),
        (prim("int16x2").into(), prim_var("float32").into()),
        (ptr("int8"), prim_var("int64").into()),
        (ptr("int8"), prim_var("int32").into()),
        (PrimType::int(64).into(), pointer.clone().into()),
        (PrimType::int(32).into(), pointer.into()),
        (ptr("int8"), text.clone()),
        (PrimType::int(32).into(), text),
        (function, prim_var("int32").into()),
    ];
    for (ty, x) in cases {
        let native =
            global("tirx.reinterpret").call_tuple((ty.clone(), x.clone(), UnknownLoc::new()));
        let what = format!("reinterpret({}, {})", repr(&ty), repr(&x));
        agree(&what, reinterpret(ty, x), native);
    }
}

#[test]
fn fast_erf_follows_the_native_lowering() {
    let _serial = serial();
    let erf = Op::get("prim.erf").unwrap();
    // Metal lowers `erf` to C++ `fast_erf_float_expr`; without the Metal backend, the shape of the
    // approximation is checked instead.
    let rules = Op::attr_map::<Function>("metal.FLowerIntrinsic").unwrap();
    let rule = rules.and_then(|rules| rules.get(&erf).unwrap());
    for (ty, bits) in [(PrimType::float(32), 32), (PrimType::float(16), 16)] {
        let x = PrimVar::new("x", ty.clone(), UNKNOWN_LOC);
        let rust = fast_erf_float_expr(&x, bits).unwrap();
        assert!(structural_equal(&rust.ty, &ty));
        assert!(rust.downcast_ref::<tvm_tirx::ir::prim::DivObj>().is_some(), "p / q");
        if let Some(rule) = &rule {
            let ty: Type = ty.into();
            let call: Expr = Call::new(
                Some(&ty),
                erf.clone(),
                [Expr::from(&x)],
                None,
                Vec::<Type>::new(),
                UNKNOWN_LOC,
            )
            .unwrap()
            .into();
            let native: Expr = rule.call_tuple((call,)).unwrap().try_into().unwrap();
            assert!(
                structural_equal(&rust, &native),
                "{}: {} vs {}",
                repr(&ty),
                repr(&rust),
                repr(&native)
            );
        }
    }
    // The operator chain checks its operands as C++ does: an integer argument is matched to the
    // float constants; a non-primitive one now fails the `PrimVar` check before the call.
    assert!(fast_erf_float_expr(prim_var("int32"), 32).is_ok());
    assert!(var("s", ptr("int8")).downcast::<PrimVar>().is_err());
}
