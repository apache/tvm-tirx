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

//! Tests of `tvm_tirx::target`, which is the single file `src/target.rs`: targets, target kinds,
//! tags and the code generators of the target kinds.

#[path = "../common/mod.rs"]
mod common;

use tvm_tirx::target::{self, Target, TargetKind, TargetKindObj, TargetObj};
use tvm_tirx::tvm_ffi::{Any, Map, Object, ObjectRefCore, String as FfiString};

use common::{assert_type_contracts, empty_module, global, load_llvm_backend};

fn config(kind: &str) -> Map<FfiString, Any> {
    Map::from_iter([(FfiString::from("kind"), Any::from(FfiString::from(kind)))])
}

/// A target built by the native function, without the Rust wrapper.
fn native_target(name: &str) -> Target {
    global("target.Target").call_tuple((FfiString::from(name),)).unwrap().try_into().unwrap()
}

#[test]
fn generated_types_match_the_registry() {
    // Both hold members reflection does not describe, so both bindings are opaque.
    assert_type_contracts!(false; TargetObj: Object, TargetKindObj: Object);
}

#[test]
fn native_targets_are_read_through_accessors() {
    let llvm = native_target("llvm");
    let kind: TargetKind = llvm.kind.clone();
    assert_eq!(kind.name().unwrap().as_str(), "llvm");
    assert!(kind.default_keys().unwrap().iter().any(|key| key.as_str() == "cpu"));
    assert!(llvm.keys.iter().any(|key| key.as_str() == "cpu"));
    assert!(llvm.host.is_none());

    // The kind's reflected device type is what the native query reports.
    let cuda = native_target("cuda");
    assert_eq!(cuda.kind.name().unwrap().as_str(), "cuda");
    let device_type: i64 = global("target.TargetGetDeviceType")
        .call_tuple((cuda.clone(),))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(cuda.kind.default_device_type().unwrap(), device_type);
    assert_ne!(device_type, kind.default_device_type().unwrap());

    // `host` is declared as an `ObjectRef`; it holds a `Target`.
    let with_host: Target = global("target.WithHost")
        .call_tuple((cuda.clone(), llvm.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    let host: Target = with_host.host.clone().unwrap().downcast().unwrap();
    assert_eq!(host.kind.name().unwrap().as_str(), "llvm");
    assert!(with_host.host.clone().unwrap().downcast::<TargetKind>().is_err());
}

#[test]
fn construction_and_queries() {
    let llvm = Target::new("llvm").unwrap();
    assert_eq!(llvm.kind.name().unwrap().as_str(), "llvm");
    assert_eq!(llvm.get_target_device_type().unwrap(), 1);
    assert!(llvm.export().unwrap().contains_key(&FfiString::from("kind")).unwrap());
    assert!(llvm.as_json().unwrap().as_str().contains("llvm"));
    assert!(llvm.feature("no_such_feature").unwrap().is_none());

    let c = Target::from_config(config("c")).unwrap();
    assert_eq!(c.kind.name().unwrap().as_str(), "c");
    assert!(llvm.with_host(Some(&c)).unwrap().host.is_some());
    assert!(llvm.with_host(None).unwrap().host.is_none());

    let kinds = Target::list_kinds().unwrap();
    assert!(kinds.iter().any(|kind| kind.as_str() == "llvm"));
    let options = llvm.kind.options().unwrap();
    assert!(options.contains_key(&FfiString::from("mtriple")).unwrap());
    assert_eq!(TargetKind::options_from_name("llvm").unwrap().len(), options.len());
}

#[test]
fn scope_ends_on_drop() {
    assert!(Target::current(true).unwrap().is_none());
    assert!(Target::current(false).is_err());
    {
        let _scope = Target::new("c").unwrap().scope().unwrap();
        let current = Target::current(false).unwrap().unwrap();
        assert_eq!(current.kind.name().unwrap().as_str(), "c");
    }
    assert!(Target::current(true).unwrap().is_none());
}

#[test]
fn tags() {
    let tagged = target::register_tag("rust-test/llvm", config("llvm"), true).unwrap();
    assert_eq!(tagged.kind.name().unwrap().as_str(), "llvm");
    let tags = target::list_tags().unwrap();
    assert!(tags.contains_key(&FfiString::from("rust-test/llvm")).unwrap());
    assert!(target::register_tag("rust-test/llvm", config("llvm"), false).is_err());
}

#[test]
fn build_by_kind() {
    load_llvm_backend();
    let c = Target::new("c").unwrap();
    target::build("c", empty_module(), &c).unwrap();
    target::build("llvm", empty_module(), Target::new("llvm").unwrap()).unwrap();
    assert!(target::build("no_such_kind", empty_module(), &c).is_err());
}
