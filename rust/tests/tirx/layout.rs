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

//! Tests of `tirx::layout`: layout axes, tile layouts and swizzled layouts.

use tvm_tirx::ir::Expr;
use tvm_tirx::sym::Analyzer;
use tvm_tirx::tirx::{Axis, ComposeLayout, Iter, Layout, TileLayout};
use tvm_tirx::tvm_ffi::{Map, String as FfiString};

use super::{as_i64, int32, tile_8x8};
use crate::common::{global, serial};

#[test]
fn memory_axes() {
    let _serial = serial();
    let m = Axis::get("m").unwrap();
    assert!(m.is_memory().unwrap() && !m.is_thread().unwrap());
    assert!(m.backend().unwrap().is_none());
    assert!(m.get_scope().unwrap().is_none());
}

#[test]
fn layouts() {
    let _serial = serial();
    let analyzer = Analyzer::new();
    let tile = tile_8x8();
    let layout = Layout::from(&tile);
    layout.check_well_formed().unwrap();
    assert_eq!(as_i64(&layout.size(None).unwrap()), 64);
    assert_eq!(as_i64(&layout.span(Some("m")).unwrap()), 64);
    let key = FfiString::from("m");
    let at = |coord: Map<FfiString, Expr>| as_i64(&coord.get(&key).unwrap().unwrap());
    assert_eq!(at(layout.apply([int32(1), int32(2)], &analyzer).unwrap()), 10);
    assert_eq!(at(layout.apply_linear(int32(10), &analyzer).unwrap()), 10);
    assert_eq!(at(layout.apply_with_shape([int32(10)], [int32(64)], &analyzer).unwrap()), 10);
    let canonical = layout.canonicalize().unwrap();
    assert_eq!(as_i64(&canonical.size(None).unwrap()), 64);

    assert!(tile.is_trivial().unwrap());
    assert!(tile.is_contiguous(None).unwrap());
    let (grouped, separators) = tile.group([int32(8), int32(8)]).unwrap();
    assert_eq!(separators.len(), 3);
    assert!(grouped.is_trivial().unwrap());
    assert!(tile.get_scope().unwrap().is_none());

    let tiled = layout.tile(&tile, [int32(8), int32(8)], [int32(8), int32(8)]).unwrap();
    assert_eq!(as_i64(&tiled.size(None).unwrap()), 64 * 64);

    // Memory offset 10 becomes 10 ^ ((10 >> 3) & 3) = 11.
    let swizzled = ComposeLayout::new(0, 2, 3, &tile, true).unwrap();
    assert_eq!(swizzled.atom_len().unwrap(), 3);
    assert_eq!(at(Layout::from(&swizzled).apply([int32(1), int32(2)], &analyzer).unwrap()), 11);
}

#[test]
fn thread_layout_scope() {
    let _serial = serial();
    global("tirx.cuda.RegisterLayoutAxes").call_tuple(()).unwrap();
    let laneid = Axis::get("laneid").unwrap();
    let lanes =
        TileLayout::new([Iter::new(int32(32), int32(1), laneid)], Vec::<Iter>::new(), &Map::new());
    let (inner, outer) = lanes.get_scope().unwrap().unwrap();
    assert_eq!(inner.local_name().unwrap().as_str(), "thread");
    assert_eq!(outer.local_name().unwrap().as_str(), "warp");
}
