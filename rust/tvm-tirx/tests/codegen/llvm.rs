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

//! Tests of `codegen::llvm`: the queries of the LLVM backend.

use tvm_tirx::codegen::{self, llvm};
use tvm_tirx::target::Target;

use crate::common::load_llvm_backend;

#[test]
fn llvm_queries() {
    load_llvm_backend();
    assert!(llvm::version_major().unwrap() >= 10);
    let sqrt = llvm::lookup_intrinsic_id("llvm.sqrt").unwrap();
    assert!(sqrt > 0);
    assert_eq!(llvm::get_intrinsic_name(sqrt).unwrap().as_str(), "llvm.sqrt");
    assert_eq!(llvm::lookup_intrinsic_id("no.such.intrinsic").unwrap(), 0);

    let triple = llvm::get_system_triple().unwrap();
    assert_eq!(llvm::get_default_target_triple().unwrap().as_str(), triple.as_str());
    assert!(!llvm::get_process_triple().unwrap().is_empty());
    assert_eq!(
        llvm::get_host_cpu_name().unwrap().as_str(),
        llvm::get_system_cpu().unwrap().as_str()
    );
    let _ = llvm::get_host_cpu_features().unwrap();
    assert!(!llvm::get_system_x86_vendor().unwrap().is_empty());
    assert!(!llvm::get_targets().unwrap().is_empty());
    assert!(llvm::target_enabled("llvm").unwrap());

    let target = Target::new("llvm").unwrap();
    assert!(!llvm::get_cpu_archlist(Some(&target)).unwrap().is_empty());
    let features = llvm::get_cpu_features(Some(&target)).unwrap();
    let feature = features.keys().next().expect("the host CPU has features");
    assert!(llvm::cpu_has_feature(feature.as_str(), Some(&target)).unwrap());
    assert!(!llvm::cpu_has_feature("no_such_feature", Some(&target)).unwrap());
    let _ = codegen::target_has_feature(feature.as_str(), Some(&target)).unwrap();
    assert!(llvm::get_vector_width(Some(&target)).unwrap() > 0);
    assert_eq!(llvm::get_vector_width(Some(&Target::new("c").unwrap())).unwrap(), -1);

    // Without a target, the queries use the current one.
    let _scope = target.scope().unwrap();
    assert_eq!(llvm::get_cpu_features(None).unwrap().len(), features.len());
    let cpu = llvm::get_system_cpu().unwrap();
    assert!(
        llvm::is_valid_cpu(cpu.as_str(), triple.as_str()).unwrap() || cpu.as_str() == "generic"
    );
}
