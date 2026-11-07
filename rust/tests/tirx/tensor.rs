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

//! Tests of `tirx::tensor`: tensor types.

use tvm_tirx::ir::{UNKNOWN_LOC, UnknownLoc, Expr, PrimExpr, PrimType, StringImm};
use tvm_tirx::tirx::{Layout, TensorType};
use tvm_tirx::tvm_ffi::Array;

use super::{int32, memory_scope, tile_8x8};
use crate::common::{agree, agree_kind, native, serial};

#[test]
fn tensor_types() {
    let _serial = serial();
    let scope = memory_scope();
    let ty =
        TensorType::new([int32(1)], PrimType::float(32), &scope, 0, None, UNKNOWN_LOC).unwrap();
    assert_eq!(ty.data_alignment, 64);
    assert!(ty.has_default_layout(None).unwrap());
    let tile = tile_8x8();
    let laid_out = TensorType::new(
        [int32(8), int32(8)],
        PrimType::float(32),
        &scope,
        16,
        Some(&Layout::from(&tile)),
        UNKNOWN_LOC,
    )
    .unwrap();
    assert!(laid_out.has_default_layout(None).unwrap());
}

#[test]
fn tensor_types_agree_with_cpp() {
    let _serial = serial();
    let scope = memory_scope();
    let float32 = PrimType::float(32);
    let layout: Layout = tile_8x8().into();
    let text: Expr = StringImm::new("s", UNKNOWN_LOC).into();
    let cases: Vec<(&str, Vec<PrimExpr>, i32, Option<Layout>)> = vec![
        ("TensorType(default alignment)", vec![int32(1)], 0, None),
        ("TensorType(negative alignment)", vec![int32(4), int32(4)], -8, None),
        ("TensorType(alignment)", vec![int32(8), int32(8)], 128, Some(layout)),
    ];
    for (what, shape, alignment, layout) in cases {
        let array: Array<PrimExpr> = shape.iter().cloned().collect();
        let native_type = native(
            "tirx.TensorType",
            &[
                (&array).into(),
                (&float32).into(),
                (&scope).into(),
                (&alignment).into(),
                (&layout).into(),
                (&UnknownLoc::new()).into(),
            ],
        );
        let rust =
            TensorType::new(shape, &float32, &scope, alignment, layout.as_ref(), UNKNOWN_LOC);
        agree(what, rust, native_type);
    }
    let array: Array<Expr> = [text.clone()].into_iter().collect();
    let (alignment, layout) = (0i32, Option::<Layout>::None);
    let native_type = native(
        "tirx.TensorType",
        &[
            (&array).into(),
            (&float32).into(),
            (&scope).into(),
            (&alignment).into(),
            (&layout).into(),
            (&UnknownLoc::new()).into(),
        ],
    );
    // A non-primitive extent is not a `PrimExpr`: the check comes before `TensorType::new`.
    agree_kind("TensorType(shape)", PrimExpr::try_from(text), native_type);
}
