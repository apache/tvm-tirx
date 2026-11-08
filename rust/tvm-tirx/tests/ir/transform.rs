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

//! Tests of `ir::transform`, laid out like `src/ir/transform/`: pass contexts, passes and
//! instruments.

mod core;
mod instrument;

use std::sync::{Arc, Mutex};

use tvm_tirx::ir::transform::{ModulePass, PassInfo, PassInstrument};

fn no_instruments() -> Vec<PassInstrument> {
    Vec::new()
}

type Log = Arc<Mutex<Vec<String>>>;

/// A module pass `name` that records its runs in `log`.
fn logging_pass(name: &str, log: &Log) -> ModulePass {
    let (log, entry) = (log.clone(), format!("run {name}"));
    ModulePass::new(
        move |module, _ctx| {
            log.lock().unwrap().push(entry.clone());
            Ok(module)
        },
        PassInfo::new(0, name),
    )
}
