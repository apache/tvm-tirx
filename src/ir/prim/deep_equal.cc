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
 * \file ir/prim/deep_equal.cc
 * \brief Deep equality checking.
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/expr.h>

namespace tvm {
namespace prim {

#define DEFINE_DEEP_EQUAL_BIN_EXPR(OpObj)                                          \
  bool Dispatch_(const OpObj* plhs, const PrimExpr& rhs) final {                   \
    const auto* prhs = rhs.as<OpObj>();                                            \
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() && \
           Dispatch(plhs->a, prhs->a) && Dispatch(plhs->b, prhs->b);               \
  }

#define DEFINE_DEEP_EQUAL_IMM_EXPR(OpObj)                                          \
  bool Dispatch_(const OpObj* plhs, const PrimExpr& rhs) final {                   \
    const auto* prhs = rhs.as<OpObj>();                                            \
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() && \
           plhs->value == prhs->value;                                             \
  }

class ExprDeepEqualChecker : private tvm::ExprFunctor<bool(const Expr&, const PrimExpr&)> {
 public:
  static bool Check(const PrimExpr& lhs, const PrimExpr& rhs) {
    // quick path without constructing the object
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() && rhs.defined()) return false;
    if (!rhs.defined() && lhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    if (auto* plhs = lhs.as<IntImmObj>()) {
      auto* prhs = rhs.as<IntImmObj>();
      return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
             plhs->value == prhs->value;
    }
    return ExprDeepEqualChecker().Dispatch(lhs, rhs);
  }

  bool Dispatch(const Expr& expr, const PrimExpr& rhs) final {
    PrimExpr lhs = expr.as_or_throw<PrimExpr>();
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() && rhs.defined()) return false;
    if (!rhs.defined() && lhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    return ExprFunctor::Dispatch(lhs, rhs);
  }

  bool Dispatch(const Expr& lhs, const Expr& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() || !rhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    if (auto lhs_prim = lhs.as<PrimExpr>()) {
      auto rhs_prim = rhs.as<PrimExpr>();
      return rhs_prim && Dispatch(lhs_prim.value(), rhs_prim.value());
    }
    if (auto* str = lhs.as<StringImmObj>()) {
      return str->value == rhs.as<StringImmObj>()->value;
    }
    if (lhs.as<VarObj>()) {
      return false;
    }
    if (auto* lhs_tuple = lhs.as<TupleObj>()) {
      auto* rhs_tuple = rhs.as<TupleObj>();
      return ffi::StructuralEqual()(lhs_tuple->ty, rhs_tuple->ty) &&
             ArrayDeepEqual(lhs_tuple->fields, rhs_tuple->fields);
    }
    if (auto* lhs_get_item = lhs.as<TupleGetItemObj>()) {
      auto* rhs_get_item = rhs.as<TupleGetItemObj>();
      return lhs_get_item->index == rhs_get_item->index &&
             ffi::StructuralEqual()(lhs_get_item->ty, rhs_get_item->ty) &&
             Dispatch(lhs_get_item->tuple, rhs_get_item->tuple);
    }
    if (auto* lhs_call = lhs.as<CallObj>()) {
      auto* rhs_call = rhs.as<CallObj>();
      return ffi::StructuralEqual()(lhs_call->ty, rhs_call->ty) &&
             lhs_call->op.same_as(rhs_call->op) && ArrayDeepEqual(lhs_call->args, rhs_call->args) &&
             ffi::StructuralEqual()(lhs_call->attrs, rhs_call->attrs);
    }
    return false;
  }

 private:
  bool ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!Dispatch(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  bool ArrayDeepEqual(const ffi::Array<Expr>& lhs, const ffi::Array<Expr>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!Dispatch(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  bool Dispatch_(const VarObj* plhs, const PrimExpr& rhs) final {
    // for var, we require pointer equality
    return plhs == rhs.get();
  }

  bool Dispatch_(const TensorLoadObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<TensorLoadObj>();
    // we run pointer comparison of the source
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           plhs->source.same_as(prhs->source) && ArrayDeepEqual(plhs->indices, prhs->indices);
  }

  bool Dispatch_(const prim::LetObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<prim::LetObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           Dispatch(plhs->var, prhs->var) && Dispatch(plhs->value, prhs->value) &&
           Dispatch(plhs->body, prhs->body);
  }

  bool Dispatch_(const CallObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<CallObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           plhs->op.same_as(prhs->op) && ArrayDeepEqual(plhs->args, prhs->args) &&
           ffi::StructuralEqual()(plhs->attrs, prhs->attrs);
  }

  bool Dispatch_(const prim::CastObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<prim::CastObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           Dispatch(plhs->value, prhs->value);
  }

  bool Dispatch_(const prim::NotObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<prim::NotObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           Dispatch(plhs->a, prhs->a);
  }

  bool Dispatch_(const prim::BitwiseNotObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<prim::BitwiseNotObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           Dispatch(plhs->a, prhs->a);
  }

  bool Dispatch_(const prim::SelectObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<prim::SelectObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           Dispatch(plhs->condition, prhs->condition) &&
           Dispatch(plhs->true_value, prhs->true_value) &&
           Dispatch(plhs->false_value, prhs->false_value);
  }

  bool Dispatch_(const prim::RampObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<prim::RampObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           Dispatch(plhs->base, prhs->base) && Dispatch(plhs->stride, prhs->stride) &&
           Dispatch(plhs->lanes, prhs->lanes);
  }

  bool Dispatch_(const prim::ShuffleObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<prim::ShuffleObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           ArrayDeepEqual(plhs->vectors, prhs->vectors) &&
           ArrayDeepEqual(plhs->indices, prhs->indices);
  }

  bool Dispatch_(const prim::BroadcastObj* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<prim::BroadcastObj>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           Dispatch(plhs->value, prhs->value) && Dispatch(plhs->lanes, prhs->lanes);
  }

  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::AddObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::LShiftObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::RShiftObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::BitwiseAndObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::BitwiseOrObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::BitwiseXorObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::SubObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::MulObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::DivObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::ModObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::FloorDivObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::FloorModObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::MinObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::MaxObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::EQObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::NEObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::LTObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::LEObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::GTObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::GEObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::AndObj)
  DEFINE_DEEP_EQUAL_BIN_EXPR(prim::OrObj)
  DEFINE_DEEP_EQUAL_IMM_EXPR(IntImmObj)
  DEFINE_DEEP_EQUAL_IMM_EXPR(FloatImmObj)
};

bool ExprDeepEqual::operator()(const PrimExpr& lhs, const PrimExpr& rhs) const {
  return ExprDeepEqualChecker::Check(lhs, rhs);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("prim.expr_deep_equal", [](const PrimExpr& lhs, const PrimExpr& rhs) {
    return ExprDeepEqual()(lhs, rhs);
  });
}

}  // namespace prim
}  // namespace tvm
