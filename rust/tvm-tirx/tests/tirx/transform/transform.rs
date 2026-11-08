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

//! Tests of `tirx::transform::transform`: the TIRx passes.

use tvm_tirx::ir::transform::Pass;
use tvm_tirx::target::Target;
use tvm_tirx::tirx::transform as tx;

use crate::common::serial;
use crate::{module_of, void_function};

#[test]
fn passes() {
    let _serial = serial();
    let target = Target::new("llvm").unwrap();
    let passes: Vec<Pass> = vec![
        tx::annotate_entry_func().unwrap(),
        tx::bf16_compute_legalize().unwrap(),
        tx::bf16_storage_legalize().unwrap(),
        tx::bind_target(&target).unwrap(),
        tx::common_subexpr_elim().unwrap(),
        tx::convert_ssa().unwrap(),
        tx::decorate_device_scope().unwrap(),
        tx::fp8_compute_legalize("float16").unwrap(),
        tx::fp8_storage_legalize().unwrap(),
        tx::flatten_tensor().unwrap(),
        tx::force_narrow_index_to_int32().unwrap(),
        tx::hoist_if().unwrap(),
        tx::lower_intrin().unwrap(),
        tx::lower_tvm_builtin().unwrap(),
        tx::lower_tensor_layouts().unwrap(),
        tx::lower_thread_allreduce().unwrap(),
        tx::make_packed_api().unwrap(),
        tx::narrow_data_type(32).unwrap(),
        tx::pointer_value_type_rewrite().unwrap(),
        tx::remove_no_op().unwrap(),
        tx::renormalize_split_pattern().unwrap(),
        tx::skip_assert().unwrap(),
        tx::split_host_device().unwrap(),
        tx::stmt_simplify().unwrap(),
        tx::storage_rewrite().unwrap(),
        tx::tile_dispatch().unwrap(),
        tx::unroll_loop().unwrap(),
        tx::vectorize_loop(true).unwrap(),
        tx::verify_memory().unwrap(),
    ];
    assert_eq!(passes.len(), 29);
}

#[test]
fn filter() {
    let _serial = serial();
    let func = void_function(vec![]);
    let kept = tx::filter(|_func| Ok(true)).unwrap().run(module_of("main", &func)).unwrap();
    assert!(kept.contains("main").unwrap());
    let removed = tx::filter(|_func| Ok(false)).unwrap().run(module_of("main", &func)).unwrap();
    assert!(!removed.contains("main").unwrap());
}
