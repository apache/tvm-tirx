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

//! Pass infrastructure. A [`Pass`] transforms an [`IRModule`](super::IRModule) under the current
//! [`PassContext`], whose instruments observe each pass it runs.

// tvm-ffi-stubgen(file): mod(missing_docs = allow)
// An object registered under `transform` that no file has a block for yet is generated here, to be
// moved to the file of the Python module that defines it.
// tvm-ffi-stubgen(file): mod(prefix = transform)

pub mod core;
pub mod instrument;

pub use core::*;
pub use instrument::*;

// tvm-ffi-stubgen(begin): import-section
// tvm-ffi-stubgen(end)
