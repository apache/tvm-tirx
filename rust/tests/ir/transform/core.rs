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

//! Tests of `ir::transform::core`: pass contexts, module passes and sequential passes.

use tvm_tirx::ir::transform::*;
use tvm_tirx::tvm_ffi::{Any, Map, String as FfiString};

use super::{logging_pass, no_instruments, Log};
use crate::common::{empty_module, serial};

#[test]
fn pass_context_scope() {
    let _serial = serial();
    let default_level = PassContext::current().unwrap().opt_level;
    let ctx = PassContext::create(3, ["a"], ["b"], no_instruments(), None).unwrap();
    assert_eq!(ctx.required_pass.len(), 1);
    {
        let _outer = ctx.enter().unwrap();
        assert_eq!(PassContext::current().unwrap().opt_level, 3);
        let inner = PassContext::create(1, [""; 0], [""; 0], no_instruments(), None).unwrap();
        {
            let _inner = inner.enter().unwrap();
            assert_eq!(PassContext::current().unwrap().opt_level, 1);
        }
        assert_eq!(PassContext::current().unwrap().opt_level, 3);
    }
    assert_eq!(PassContext::current().unwrap().opt_level, default_level);

    let configs = PassContext::list_configs().unwrap();
    assert!(!configs.is_empty());
    let unknown: Map<FfiString, Any> =
        [(FfiString::from("no.such.option"), Any::from(1i64))].into_iter().collect();
    assert!(PassContext::create(2, [""; 0], [""; 0], no_instruments(), Some(&unknown)).is_err());
}

#[test]
fn module_and_sequential_passes() {
    let _serial = serial();
    let log = Log::default();
    let first = logging_pass("first", &log);
    let info = Pass::from(&first).info().unwrap();
    assert_eq!((info.name.as_str(), info.opt_level), ("first", 0));

    let module = empty_module();
    let result = Pass::from(&first).run(module).unwrap();
    assert_eq!(result.functions.len(), 0);
    assert_eq!(*log.lock().unwrap(), ["run first"]);

    let failing = ModulePass::new(
        |_module, _ctx| {
            Err(tvm_tirx::tvm_ffi::Error::new(tvm_tirx::tvm_ffi::VALUE_ERROR, "refused", ""))
        },
        PassInfo::new(0, "failing"),
    );
    assert!(Pass::from(&failing).run(empty_module()).is_err());

    log.lock().unwrap().clear();
    let seq = Sequential::new(
        [
            Pass::from(logging_pass("a", &log)),
            print_ir("between").unwrap(),
            logging_pass("b", &log).into(),
        ],
        0,
        "seq",
    );
    assert_eq!(seq.passes().unwrap().len(), 3);
    assert_eq!(Pass::from(&seq).info().unwrap().name.as_str(), "seq");
    let ctx = PassContext::create(2, [""; 0], ["b"], no_instruments(), None).unwrap();
    {
        let _scope = ctx.enter().unwrap();
        Pass::from(&seq).run(empty_module()).unwrap();
    }
    assert_eq!(*log.lock().unwrap(), ["run a"], "the context disables `b`");
}
