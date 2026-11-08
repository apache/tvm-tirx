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

//! Symbolic integer analysis. An [`Analyzer`] proves, bounds and simplifies integer expressions
//! under the facts of a [`ConstraintContext`]; [`int_set`] and [`iter_affine_map`] describe the
//! values and the iterator structure of index expressions.

// tvm-ffi-stubgen(missing-docs): allow
// An object registered under `sym` that no file has a block for yet is generated here, to be moved
// to the file of the Python module that defines it.
// tvm-ffi-stubgen(prefix): sym
// tvm-ffi-stubgen(skip): sym.ConstraintContext
// The canonical simplifier's intermediate forms are internal to it.
// tvm-ffi-stubgen(skip): sym.CanonicalExpr
// tvm-ffi-stubgen(skip): sym.SplitExpr
// tvm-ffi-stubgen(skip): sym.SumExpr
// Integer sets and iterator sums are hand-written in their modules: C++ constructs them.
// tvm-ffi-stubgen(skip): sym.IntSet
// tvm-ffi-stubgen(skip): sym.IntervalSet
// tvm-ffi-stubgen(skip): sym.IterMapExpr
// tvm-ffi-stubgen(skip): sym.IterSplitExpr
// tvm-ffi-stubgen(skip): sym.IterSumExpr

pub mod analyzer;
pub mod constraint;
pub mod int_set;
pub mod iter_affine_map;

pub use analyzer::*;
pub use constraint::*;
pub use int_set::*;
pub use iter_affine_map::*;

// tvm-ffi-stubgen(begin): import-section
// tvm-ffi-stubgen(end)
