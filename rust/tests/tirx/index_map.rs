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

//! Tests of `tirx::index_map`.

use tvm_tirx::ir::{dtype, PrimExpr, PrimType, Var};
use tvm_tirx::tirx::IndexMap;
use tvm_tirx::tvm_ffi::{Array, DLDevice, DLDeviceType};

use super::{as_i64, int32, op, prim_var, range};
use crate::common::{serial, structural_equal};

#[test]
fn index_maps() {
    let _serial = serial();
    let (i, j) = (prim_var("i", PrimType::int(32)), prim_var("j", PrimType::int(32)));
    let transpose = IndexMap::new(
        Array::from_vec(vec![Var::from(&i), Var::from(&j)]),
        Array::from_vec(vec![PrimExpr::from(&j), PrimExpr::from(&i)]),
        None,
    );
    let mapped = transpose.map_indices([int32(1), int32(2)], None).unwrap();
    assert_eq!((as_i64(&mapped.get(0).unwrap()), as_i64(&mapped.get(1).unwrap())), (2, 1));
    let shape = transpose.map_shape([int32(4), int32(8)], None).unwrap();
    assert_eq!((as_i64(&shape.get(0).unwrap()), as_i64(&shape.get(1).unwrap())), (8, 4));
    let inverse = transpose.inverse([range(4), range(8)], None).unwrap();
    assert_eq!(inverse.final_indices.len(), 2);

    let k = prim_var("k", PrimType::int(32));
    let split = IndexMap::new(
        Array::from_vec(vec![Var::from(&k)]),
        Array::from_vec(vec![
            op("prim._OpFloorDiv", &k, int32(4)),
            op("prim._OpFloorMod", &k, int32(4)),
        ]),
        None,
    );
    let (unsplit, predicate) = split.non_surjective_inverse([range(14)], None).unwrap();
    assert_eq!(unsplit.final_indices.len(), 1);
    assert!(!structural_equal(&predicate, &int32(0)));

    let cpu = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = tvm_tirx::runtime::empty(&[4, 8], dtype::float(32), cpu, None).unwrap();
    let transposed = transpose.map_tensor(&tensor).unwrap();
    assert_eq!(transposed.shape(), &[8, 4]);
}
