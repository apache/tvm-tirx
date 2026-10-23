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

/*!
 * \file min_max_utils.h
 * \brief Common utilities for lowering floating-point min and max.
 */
#ifndef TVM_CODEGEN_MIN_MAX_UTILS_H_
#define TVM_CODEGEN_MIN_MAX_UTILS_H_

#include <tvm/ir/prim/expr.h>

#include <cmath>

namespace tvm {
namespace codegen {

// Compile-time NaN information from a FloatImm or Broadcast(FloatImm).
enum class FloatNaNStatus {
  kUnknown,
  kKnownNaN,
  kKnownNotNaN,
};

inline FloatNaNStatus GetFloatNaNStatus(const PrimExpr& expr) {
  const FloatImmNode* value = expr.as<FloatImmNode>();
  if (const auto* broadcast = expr.as<prim::BroadcastNode>()) {
    // Recognize uniform vector constants, including scalable broadcasts.
    value = broadcast->value.as<FloatImmNode>();
  }
  if (value == nullptr) {
    return FloatNaNStatus::kUnknown;
  }
  return std::isnan(value->value) ? FloatNaNStatus::kKnownNaN : FloatNaNStatus::kKnownNotNaN;
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_MIN_MAX_UTILS_H_
