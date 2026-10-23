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

/*! \brief Compile-time NaN information about an expression. */
enum class FloatNaNStatus {
  /*! \brief Whether the expression is NaN has not been determined. */
  kUnknown,
  /*! \brief The scalar value or every vector lane is known to be NaN. */
  kKnownNaN,
  /*! \brief The scalar value or every vector lane is known not to be NaN. */
  kKnownNotNaN,
};

/*!
 * \brief Determine an expression's NaN status from floating-point constants.
 *
 * Recognizes a scalar FloatImm or a Broadcast of a FloatImm, including
 * fixed-length and scalable broadcasts. Non-NaN constants include infinities.
 * All other expression forms have unknown NaN status.
 *
 * \param expr The expression to classify.
 * \return FloatNaNStatus::kKnownNaN or FloatNaNStatus::kKnownNotNaN for a recognized
 *         constant; FloatNaNStatus::kUnknown otherwise.
 */
inline FloatNaNStatus GetFloatNaNStatus(const PrimExpr& expr) {
  const FloatImmObj* value = expr.as<FloatImmObj>();
  if (const auto* broadcast = expr.as<prim::BroadcastObj>()) {
    // Recognize uniform vector constants, including scalable broadcasts.
    value = broadcast->value.as<FloatImmObj>();
  }
  if (value == nullptr) {
    return FloatNaNStatus::kUnknown;
  }
  return std::isnan(value->value) ? FloatNaNStatus::kKnownNaN : FloatNaNStatus::kKnownNotNaN;
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_MIN_MAX_UTILS_H_
