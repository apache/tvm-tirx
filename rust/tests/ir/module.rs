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

//! Tests of `ir::module`.

use tvm_tirx::ir::{UNKNOWN_LOC, BaseFunc, DictAttrs, Expr, GlobalInfo, GlobalVar, IRModule};
use tvm_tirx::tvm_ffi::{Any, Array, Map, String as FfiString};

use super::{function, int32};
use crate::common::{agree, native, serial, structural_equal};

#[test]
fn modules() {
    let _serial = serial();
    let func = function(0);
    let main = GlobalVar::new("main", UNKNOWN_LOC);
    let functions: Map<GlobalVar, BaseFunc> = [(main.clone(), func.clone())].into_iter().collect();
    let module = IRModule::new(&functions, DictAttrs::new(Map::new()), &Map::new()).unwrap();
    assert!(module.contains("main").unwrap());
    assert!(module.contains_global_var(&main).unwrap());
    assert!(module.has_global_var("main").unwrap());
    assert!(structural_equal(&module.get_global_var("main").unwrap(), &main));
    assert_eq!(module.get_global_vars().unwrap().len(), 1);
    assert!(structural_equal(&module.lookup(&main).unwrap(), &func));
    assert!(structural_equal(&module.lookup_name("main").unwrap(), &func));

    let other = GlobalVar::new("other", UNKNOWN_LOC);
    module.add(&other, function(1), false).unwrap();
    module.update_func(&other, function(3)).unwrap();
    assert!(structural_equal(&module.lookup(&other).unwrap(), &function(3)));
    module.remove("other").unwrap();
    assert!(!module.contains("other").unwrap());
    module.add(&other, function(1), false).unwrap();
    module.remove_global_var(&other).unwrap();
    assert!(!module.contains_global_var(&other).unwrap());

    let extra_var = GlobalVar::new("extra", UNKNOWN_LOC);
    let extra_functions: Map<GlobalVar, BaseFunc> =
        [(extra_var.clone(), function(4))].into_iter().collect();
    let extra = IRModule::new(&extra_functions, DictAttrs::new(Map::new()), &Map::new()).unwrap();
    module.update(&extra).unwrap();
    assert!(module.contains("extra").unwrap());
    module.update_global_info("info", Vec::<GlobalInfo>::new()).unwrap();
    assert!(module.global_infos.get(&FfiString::from("info")).is_ok());

    let copy = module.shallow_copy().unwrap();
    copy.remove("extra").unwrap();
    assert!(module.contains("extra").unwrap());

    assert!(module.get_attr("note").unwrap().is_none());
    let module = module.with_attr("note", int32(7)).unwrap();
    let note = module.get_attr("note").unwrap().unwrap();
    assert!(structural_equal(&Expr::try_from(note).unwrap(), &int32(7)));
    let attrs: Map<FfiString, Any> =
        [(FfiString::from("other"), Any::from(int32(1)))].into_iter().collect();
    let module = module.with_attrs(&attrs).unwrap();
    assert!(module.get_attr("other").unwrap().is_some());
    let module = module.without_attr("note").unwrap();
    assert!(module.get_attr("note").unwrap().is_none());

    let from_expr = IRModule::from_expr(function(5), &Map::new()).unwrap();
    assert!(from_expr.contains("main").unwrap());
}

/// An `IRModule` case: its name, functions and attributes.
type ModuleCase<'a> = (&'a str, Vec<(GlobalVar, BaseFunc)>, Option<&'a DictAttrs>);

#[test]
fn modules_agree_with_cpp() {
    let _serial = serial();
    let (main, other) = (GlobalVar::new("main", UNKNOWN_LOC), GlobalVar::new("other", UNKNOWN_LOC));
    let twin = GlobalVar::new("main", UNKNOWN_LOC);
    let note: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let attrs = DictAttrs::new(note);
    let infos: Map<FfiString, Array<GlobalInfo>> =
        [(FfiString::from("info"), Array::from_vec(vec![]))].into_iter().collect();
    let cases: Vec<ModuleCase> = vec![
        ("IRModule", vec![(main.clone(), function(0)), (other.clone(), function(1))], None),
        ("IRModule(attrs)", vec![(main.clone(), function(0))], Some(&attrs)),
        ("IRModule(empty)", vec![], None),
        ("IRModule(duplicate names)", vec![(main.clone(), function(0)), (twin, function(1))], None),
    ];
    for (what, functions, attrs) in cases {
        let functions: Map<GlobalVar, BaseFunc> = functions.into_iter().collect();
        let attrs_arg = attrs.cloned();
        let native_module =
            native("ir.IRModule", &[(&functions).into(), (&attrs_arg).into(), (&infos).into()]);
        let attrs = attrs.cloned().unwrap_or_else(|| DictAttrs::new(Map::new()));
        let rust = IRModule::new(&functions, attrs, &infos);
        if let (Ok(rust), Ok(native_module)) = (&rust, &native_module) {
            let native_module: IRModule = native_module.clone().try_into().unwrap();
            assert!(
                structural_equal(
                    &rust.get_global_vars().unwrap(),
                    &native_module.get_global_vars().unwrap()
                ),
                "{what}"
            );
        }
        agree(what, rust, native_module);
    }
}
