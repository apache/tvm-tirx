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
 * \file tvm/tirx/op/attr.h
 * \brief Attribute keys and callback contracts for TIRx operators.
 *
 * These attributes can be set via OpDef::set_attr
 *
 * \sa tvm/ir/op.h
 */
#ifndef TVM_TIRX_OP_ATTR_H_
#define TVM_TIRX_OP_ATTR_H_

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/native_function.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/stmt.h>

namespace tvm {
namespace tirx {

class StructuralVisitorWithConstraintContextObj;
class StructuralMutatorWithConstraintContextObj;

/*!
 * \brief The intrinsic lowering function for given op.
 */
using FLowerIntrinsic = ffi::TypedFunction<PrimExpr(PrimExpr)>;

/*!
 * \brief The legalization function for given tirx op.
 */
using FLegalize = ffi::TypedFunction<PrimExpr(PrimExpr)>;

namespace op_attr {
/*!
 * \brief Global symbol of the op after lowering.
 */
inline constexpr ffi::reflection::AttrKey<ffi::String> kGlobalSymbol{"TGlobalSymbol"};
/*!
 * \brief Whether the op is overloaded for vector form.
 */
inline constexpr ffi::reflection::AttrKey<bool> kVectorizable{"TVectorizable"};
/*!
 * \brief Constraint-aware descent for a Call or RegionStmt using this Op.
 * Receives the complete containing node as value. The hook owns child descent,
 * definition regions, and constraint scopes, and propagates interrupts and errors.
 * Missing hooks use ordinary Call descent or body-scoped RegionStmt descent.
 */
inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<ffi::Optional<
    ffi::VisitInterrupt>(StructuralVisitorWithConstraintContextObj*, ffi::AnyView value)>>
    kStructuralVisitWithConstraintContext{"__s_visit_with_constraint_context__"};
/*!
 * \brief Copying mutation of a complete Call or RegionStmt using this Op.
 * The hook owns descent, definition remaps, and constraint scopes. Descend with
 * in-place mutation disallowed and return a replacement or Unchanged, or an error.
 */
inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<
    ffi::UnchangedOr<ffi::Any>(StructuralMutatorWithConstraintContextObj*, ffi::AnyView value)>>
    kStructuralMutateWithConstraintContext{"__s_mutate_with_constraint_context__"};
/*!
 * \brief Mutation when the engine permits in-place updates of the containing node.
 * Child mutation still checks ownership. Missing overrides use the Op's copying
 * hook; if both are absent, use ordinary Call or body-scoped RegionStmt descent.
 */
inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<
    ffi::UnchangedOr<ffi::Any>(StructuralMutatorWithConstraintContextObj*, ffi::AnyView value)>>
    kStructuralMaybeInplaceMutateWithConstraintContext{
        "__s_maybe_inplace_mutate_with_constraint_context__"};
}  // namespace op_attr

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_OP_ATTR_H_
