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

//! Tests of `ir::transform::instrument`: pass instruments and pass timing.

use tvm_tirx::ir::transform::*;
use tvm_tirx::ir::IRModule;

use super::{logging_pass, Log};
use crate::common::{empty_module, serial};

#[test]
fn instruments() {
    let _serial = serial();
    let log = Log::default();
    let hook = |log: &Log, entry: &'static str| {
        let log = log.clone();
        Box::new(move || {
            log.lock().unwrap().push(entry.to_owned());
            Ok(())
        })
    };
    let pass_hook = |log: &Log, prefix: &'static str| {
        let log = log.clone();
        Box::new(move |_module: IRModule, info: PassInfo| {
            log.lock().unwrap().push(format!("{prefix} {}", info.name.as_str()));
            Ok(())
        })
    };
    let should_run =
        Box::new(|_module: IRModule, info: PassInfo| Ok(info.name.as_str() != "skipped"));
    let probe = BasePassInstrument::new(
        "probe",
        PassInstrumentHooks {
            enter_pass_ctx: Some(hook(&log, "enter")),
            exit_pass_ctx: Some(hook(&log, "exit")),
            should_run: Some(should_run),
            run_before_pass: Some(pass_hook(&log, "before")),
            run_after_pass: Some(pass_hook(&log, "after")),
        },
    );
    assert_eq!(probe.name().unwrap().as_str(), "probe");
    let silent = BasePassInstrument::new("silent", PassInstrumentHooks::default());

    let ctx = PassContext::new(2, [""; 0], [""; 0], [&probe], None).unwrap();
    {
        let _scope = ctx.enter().unwrap();
        Pass::from(logging_pass("kept", &log)).run(empty_module()).unwrap();
        Pass::from(logging_pass("skipped", &log)).run(empty_module()).unwrap();
        ctx.override_instruments([&silent]).unwrap();
        Pass::from(logging_pass("quiet", &log)).run(empty_module()).unwrap();
    }
    assert_eq!(
        *log.lock().unwrap(),
        ["enter", "before kept", "run kept", "after kept", "exit", "run quiet"]
    );
}

#[test]
fn pass_timing() {
    let _serial = serial();
    let log = Log::default();
    let timing = pass_timing_instrument().unwrap();
    let ctx = PassContext::new(2, [""; 0], [""; 0], [timing], None).unwrap();
    let _scope = ctx.enter().unwrap();
    Pass::from(logging_pass("timed", &log)).run(empty_module()).unwrap();
    let profile = render_pass_timing().unwrap();
    assert!(profile.as_str().contains("timed"), "{}", profile.as_str());
}
