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

//! Tests of `tvm_tirx::sym`, laid out like `src/sym/`: the tests of `src/sym/<path>.rs` are in
//! `tests/sym/<path>.rs`.

#[path = "../common/mod.rs"]
mod common;

mod analyzer;
mod int_set;
mod iter_affine_map;

use tvm_tirx::ir::{UNKNOWN_LOC, dtype, Expr, IntImm, Location, PrimType, Range, Var};
use tvm_tirx::sym::Analyzer;
use tvm_tirx::tvm_ffi::String as FfiString;
use tvm_tirx::tvm_ffi::object::ObjectRef;

use common::global;

fn int32(value: i64) -> Expr {
    IntImm::from_dtype(dtype::int(32), value, UNKNOWN_LOC).unwrap().into()
}

fn op(name: &str, a: impl Into<Expr>, b: impl Into<Expr>) -> Expr {
    global(name).call_tuple((a.into(), b.into(), Location::unknown())).unwrap().try_into().unwrap()
}

fn var(name: &str) -> Var {
    global("ir.Var")
        .call_tuple((FfiString::from(name), PrimType::int(32), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap()
}

fn range(begin: i64, extent: i64) -> Range {
    Range::from_fields(int32(begin), int32(extent), Location::unknown())
}

// Constraint data is passed opaquely until the Rust bindings expose typed contexts.
fn context(constraints: impl IntoIterator<Item = ObjectRef>) -> ObjectRef {
    let context: ObjectRef = global("sym.ConstraintContext")
        .call_tuple((None::<ObjectRef>, 0_i64))
        .unwrap()
        .try_into()
        .unwrap();
    for constraint in constraints {
        global("sym.ConstraintContextAppend")
            .call_packed(&[
                tvm_tirx::tvm_ffi::AnyView::from(&context),
                tvm_tirx::tvm_ffi::AnyView::from(&constraint),
            ])
            .unwrap();
    }
    context
}

fn range_constraint(var: &Var, domain: Range) -> ObjectRef {
    global("sym.RangeConstraint").call_tuple((var, domain)).unwrap().try_into().unwrap()
}

fn predicate_constraint(predicate: Expr) -> ObjectRef {
    global("sym.PredicateConstraint").call_tuple((predicate, false)).unwrap().try_into().unwrap()
}

struct Fixture {
    analyzer: Analyzer,
    x: Var,
}

/// An analyzer with `x` bound to `[0, 16)`, through the native functions only.
fn fixture() -> Fixture {
    let x = var("x");
    // A typed `Analyzer` handle goes through `call_tuple` like any other object.
    let analyzer: Analyzer = global("sym.Analyzer").call_tuple(()).unwrap().try_into().unwrap();
    let constraints = context([range_constraint(&x, range(0, 16))]);
    global("sym.AnalyzerSyncConstraintContext")
        .call_packed(&[
            tvm_tirx::tvm_ffi::AnyView::from(&analyzer),
            tvm_tirx::tvm_ffi::AnyView::from(&constraints),
        ])
        .unwrap();
    Fixture { analyzer, x }
}
