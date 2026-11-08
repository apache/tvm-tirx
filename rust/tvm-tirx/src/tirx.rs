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

//! Tensor programs. A [`Function`] computes on tensors, variables typed by a [`TensorType`] whose
//! [`Layout`] maps their indices to memory and threads; [`tile`] instructions are dispatched per
//! [`ThreadScope`], and [`transform`] and [`analysis`] hold the passes and analyses.

// An object registered under `tirx` that no file has a block for yet is generated here, to be moved
// to the file of the Python module that defines it.
// tvm-ffi-stubgen(file): mod(missing_docs = allow, prefix = tirx)
// Layouts are hand-written in `layout`: C++ constructs them.
// tvm-ffi-stubgen(object): tirx.Layout(skip)
// tvm-ffi-stubgen(object): tirx.TileLayout(skip)
// tvm-ffi-stubgen(object): tirx.ComposeLayout(skip)

pub mod analysis;
pub mod function;
pub mod index_map;
pub mod layout;
pub mod op;
pub mod op_attr;
pub mod structural;
pub mod structural_map_with_constraint_context;
pub mod structural_mutate_with_constraint_context;
pub mod structural_visit_with_constraint_context;
pub mod structural_walk_with_constraint_context;
pub mod tensor;
pub mod thread_scope;
pub mod tile;
pub mod transform;
pub mod r#type;

pub use function::*;
pub use index_map::*;
pub use layout::*;
pub use op::*;
pub use structural::*;
pub use tensor::*;
pub use thread_scope::*;
pub use r#type::*;

// tvm-ffi-stubgen(begin): import-section
// tvm-ffi-stubgen(end)
