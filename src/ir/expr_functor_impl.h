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
#ifndef TVM_IR_EXPR_FUNCTOR_IMPL_H_
#define TVM_IR_EXPR_FUNCTOR_IMPL_H_

#include <tvm/ir/expr_functor.h>

// Apply F(Node, Result) to expressions whose default traversal is shared by ordinary
// and constraint-aware native functors. Result is the mutation return category.
// Callers use this list to generate virtual hook definitions and dispatch entries.
// Structural hooks keep direct Expected implementations, separate from native throwing APIs.
#define TVM_IR_SHARED_EXPR_NODES(F) \
  F(OpaqueExprObj, Expr)            \
  F(TupleObj, Expr)                 \
  F(TupleGetItemObj, Expr)          \
  F(TensorLoadObj, PrimExpr)        \
  F(TensorRegionObj, Expr)          \
  F(GlobalVarObj, Expr)             \
  F(GenericConstObj, Expr)          \
  F(IntImmObj, PrimExpr)            \
  F(FloatImmObj, PrimExpr)          \
  F(OpObj, Expr)                    \
  F(prim::CastObj, PrimExpr)        \
  F(prim::BitwiseNotObj, PrimExpr)  \
  F(prim::NotObj, PrimExpr)         \
  F(prim::SelectObj, PrimExpr)      \
  F(prim::RampObj, PrimExpr)        \
  F(prim::BroadcastObj, PrimExpr)   \
  F(prim::ShuffleObj, PrimExpr)     \
  F(prim::AddObj, PrimExpr)         \
  F(prim::LShiftObj, PrimExpr)      \
  F(prim::RShiftObj, PrimExpr)      \
  F(prim::BitwiseAndObj, PrimExpr)  \
  F(prim::BitwiseOrObj, PrimExpr)   \
  F(prim::BitwiseXorObj, PrimExpr)  \
  F(prim::SubObj, PrimExpr)         \
  F(prim::MulObj, PrimExpr)         \
  F(prim::DivObj, PrimExpr)         \
  F(prim::ModObj, PrimExpr)         \
  F(prim::FloorDivObj, PrimExpr)    \
  F(prim::FloorModObj, PrimExpr)    \
  F(prim::MinObj, PrimExpr)         \
  F(prim::MaxObj, PrimExpr)         \
  F(prim::EQObj, PrimExpr)          \
  F(prim::NEObj, PrimExpr)          \
  F(prim::LTObj, PrimExpr)          \
  F(prim::LEObj, PrimExpr)          \
  F(prim::GTObj, PrimExpr)          \
  F(prim::GEObj, PrimExpr)          \
  F(prim::AndObj, PrimExpr)         \
  F(prim::OrObj, PrimExpr)

namespace tvm {
namespace details {

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const OpaqueExprObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->ty));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const TupleObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->fields));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const TupleGetItemObj* node) {
  // skips: index
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->tuple));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const TensorLoadObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->indices));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const TensorRegionObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->region));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const GlobalVarObj* node) {
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const GenericConstObj* node) {
  return visitor->Visit(node->ty);
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const IntImmObj* node) {
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const FloatImmObj* node) {
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const OpObj* node) {
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const prim::CastObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->value));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const prim::NotObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->a));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const prim::BitwiseNotObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->a));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const prim::SelectObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->true_value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->false_value));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const prim::RampObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->base));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->stride));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->lanes));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const prim::BroadcastObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->lanes));
  return std::nullopt;
}

template <typename Visitor>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const prim::ShuffleObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->vectors));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->indices));
  return std::nullopt;
}

template <typename Mutator>
UnchangedOr<Expr> MutateExprDefault(Mutator* mutator, const OpaqueExprObj* node,
                                    InplaceMode inplace_mode) {
  auto ty_u = mutator->Mutate(node->ty, inplace_mode).template as_or_throw<UnchangedOr<Type>>();
  if (ty_u.UnchangedOrSameAs(node->ty)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<OpaqueExprObj*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<OpaqueExprObj>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  return Expr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<Expr> MutateExprDefault(Mutator* mutator, const TupleObj* node,
                                    InplaceMode inplace_mode) {
  auto ty_u = mutator->Mutate(node->ty, inplace_mode).template as_or_throw<UnchangedOr<Type>>();
  auto fields_u = mutator->Mutate(node->fields, inplace_mode)
                      .template as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  if (ty_u.UnchangedOrSameAs(node->ty) && fields_u.UnchangedOrSameAs(node->fields))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TupleObj*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!fields_u.IsUnchanged()) writable->fields = std::move(fields_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TupleObj>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!fields_u.IsUnchanged()) copy->fields = std::move(fields_u).ValueUnchecked();
  return Expr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<Expr> MutateExprDefault(Mutator* mutator, const TupleGetItemObj* node,
                                    InplaceMode inplace_mode) {
  auto ty_u = mutator->Mutate(node->ty, inplace_mode).template as_or_throw<UnchangedOr<Type>>();
  auto tuple_u = mutator->Mutate(node->tuple, inplace_mode);
  if (ty_u.UnchangedOrSameAs(node->ty) && tuple_u.UnchangedOrSameAs(node->tuple))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TupleGetItemObj*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!tuple_u.IsUnchanged()) writable->tuple = std::move(tuple_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TupleGetItemObj>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!tuple_u.IsUnchanged()) copy->tuple = std::move(tuple_u).ValueUnchecked();
  return Expr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const TensorLoadObj* node,
                                        InplaceMode inplace_mode) {
  auto ty_u = mutator->Mutate(node->ty, inplace_mode).template as_or_throw<UnchangedOr<Type>>();
  Type result_ty = std::move(ty_u).ValueOrUnchanged(node->ty);
  TVM_FFI_CHECK(result_ty.as<PrimTypeObj>(), TypeError)
      << "TensorLoad requires a concrete PrimType";
  auto source_u = mutator->Mutate(node->source, inplace_mode);
  auto indices_u = mutator->Mutate(node->indices, inplace_mode)
                       .template as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  if (result_ty.same_as(node->ty) && source_u.UnchangedOrSameAs(node->source) &&
      indices_u.UnchangedOrSameAs(node->indices))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TensorLoadObj*>(node);
    writable->ty = std::move(result_ty);
    if (!source_u.IsUnchanged()) writable->source = std::move(source_u).ValueUnchecked();
    if (!indices_u.IsUnchanged()) writable->indices = std::move(indices_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TensorLoadObj>(*node);
  copy->ty = std::move(result_ty);
  if (!source_u.IsUnchanged()) copy->source = std::move(source_u).ValueUnchecked();
  if (!indices_u.IsUnchanged()) copy->indices = std::move(indices_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<Expr> MutateExprDefault(Mutator* mutator, const TensorRegionObj* node,
                                    InplaceMode inplace_mode) {
  auto ty_u = mutator->Mutate(node->ty, inplace_mode).template as_or_throw<UnchangedOr<Type>>();
  auto source_u = mutator->Mutate(node->source, inplace_mode);
  auto region_u = mutator->Mutate(node->region, inplace_mode)
                      .template as_or_throw<UnchangedOr<ffi::Array<Range>>>();
  if (ty_u.UnchangedOrSameAs(node->ty) && source_u.UnchangedOrSameAs(node->source) &&
      region_u.UnchangedOrSameAs(node->region))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TensorRegionObj*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!source_u.IsUnchanged()) writable->source = std::move(source_u).ValueUnchecked();
    if (!region_u.IsUnchanged()) writable->region = std::move(region_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TensorRegionObj>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!source_u.IsUnchanged()) copy->source = std::move(source_u).ValueUnchecked();
  if (!region_u.IsUnchanged()) copy->region = std::move(region_u).ValueUnchecked();
  return Expr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<Expr> MutateExprDefault(Mutator* mutator, const GlobalVarObj* node,
                                    InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

template <typename Mutator>
UnchangedOr<Expr> MutateExprDefault(Mutator* mutator, const GenericConstObj* node,
                                    InplaceMode inplace_mode) {
  auto ty = mutator->Mutate(node->ty, inplace_mode).template as_or_throw<UnchangedOr<Type>>();
  if (ty.UnchangedOrSameAs(node->ty)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<GenericConstObj*>(node)->ty = std::move(ty).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<GenericConstObj>(*node);
  copy->ty = std::move(ty).ValueUnchecked();
  return Expr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const IntImmObj* node,
                                        InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const FloatImmObj* node,
                                        InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

template <typename Mutator>
UnchangedOr<Expr> MutateExprDefault(Mutator* mutator, const OpObj* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const prim::CastObj* node,
                                        InplaceMode inplace_mode) {
  auto value_u = mutator->Mutate(node->value, inplace_mode);
  if (value_u.UnchangedOrSameAs(node->value)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::CastObj*>(node);
    if (!value_u.IsUnchanged()) writable->value = std::move(value_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::CastObj>(*node);
  if (!value_u.IsUnchanged()) copy->value = std::move(value_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const prim::NotObj* node,
                                        InplaceMode inplace_mode) {
  auto a_u = mutator->Mutate(node->a, inplace_mode);
  if (a_u.UnchangedOrSameAs(node->a)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::NotObj*>(node);
    if (!a_u.IsUnchanged()) writable->a = std::move(a_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::NotObj>(*node);
  if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const prim::BitwiseNotObj* node,
                                        InplaceMode inplace_mode) {
  auto a_u = mutator->Mutate(node->a, inplace_mode);
  if (a_u.UnchangedOrSameAs(node->a)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::BitwiseNotObj*>(node);
    if (!a_u.IsUnchanged()) writable->a = std::move(a_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::BitwiseNotObj>(*node);
  if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const prim::SelectObj* node,
                                        InplaceMode inplace_mode) {
  auto condition_u = mutator->Mutate(node->condition, inplace_mode);
  auto true_value_u = mutator->Mutate(node->true_value, inplace_mode);
  auto false_value_u = mutator->Mutate(node->false_value, inplace_mode);
  if (condition_u.UnchangedOrSameAs(node->condition) &&
      true_value_u.UnchangedOrSameAs(node->true_value) &&
      false_value_u.UnchangedOrSameAs(node->false_value))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::SelectObj*>(node);
    if (!condition_u.IsUnchanged()) writable->condition = std::move(condition_u).ValueUnchecked();
    if (!true_value_u.IsUnchanged())
      writable->true_value = std::move(true_value_u).ValueUnchecked();
    if (!false_value_u.IsUnchanged())
      writable->false_value = std::move(false_value_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::SelectObj>(*node);
  if (!condition_u.IsUnchanged()) copy->condition = std::move(condition_u).ValueUnchecked();
  if (!true_value_u.IsUnchanged()) copy->true_value = std::move(true_value_u).ValueUnchecked();
  if (!false_value_u.IsUnchanged()) copy->false_value = std::move(false_value_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const prim::RampObj* node,
                                        InplaceMode inplace_mode) {
  auto base_u = mutator->Mutate(node->base, inplace_mode);
  auto stride_u = mutator->Mutate(node->stride, inplace_mode);
  auto lanes_u = mutator->Mutate(node->lanes, inplace_mode);
  if (base_u.UnchangedOrSameAs(node->base) && stride_u.UnchangedOrSameAs(node->stride) &&
      lanes_u.UnchangedOrSameAs(node->lanes))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::RampObj*>(node);
    if (!base_u.IsUnchanged()) writable->base = std::move(base_u).ValueUnchecked();
    if (!stride_u.IsUnchanged()) writable->stride = std::move(stride_u).ValueUnchecked();
    if (!lanes_u.IsUnchanged()) writable->lanes = std::move(lanes_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::RampObj>(*node);
  if (!base_u.IsUnchanged()) copy->base = std::move(base_u).ValueUnchecked();
  if (!stride_u.IsUnchanged()) copy->stride = std::move(stride_u).ValueUnchecked();
  if (!lanes_u.IsUnchanged()) copy->lanes = std::move(lanes_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const prim::BroadcastObj* node,
                                        InplaceMode inplace_mode) {
  auto value_u = mutator->Mutate(node->value, inplace_mode);
  auto lanes_u = mutator->Mutate(node->lanes, inplace_mode);
  if (value_u.UnchangedOrSameAs(node->value) && lanes_u.UnchangedOrSameAs(node->lanes))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::BroadcastObj*>(node);
    if (!value_u.IsUnchanged()) writable->value = std::move(value_u).ValueUnchecked();
    if (!lanes_u.IsUnchanged()) writable->lanes = std::move(lanes_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::BroadcastObj>(*node);
  if (!value_u.IsUnchanged()) copy->value = std::move(value_u).ValueUnchecked();
  if (!lanes_u.IsUnchanged()) copy->lanes = std::move(lanes_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

template <typename Mutator>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const prim::ShuffleObj* node,
                                        InplaceMode inplace_mode) {
  auto vectors_u = mutator->Mutate(node->vectors, inplace_mode)
                       .template as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto indices_u = mutator->Mutate(node->indices, inplace_mode)
                       .template as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  if (vectors_u.UnchangedOrSameAs(node->vectors) && indices_u.UnchangedOrSameAs(node->indices))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::ShuffleObj*>(node);
    if (!vectors_u.IsUnchanged()) writable->vectors = std::move(vectors_u).ValueUnchecked();
    if (!indices_u.IsUnchanged()) writable->indices = std::move(indices_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::ShuffleObj>(*node);
  if (!vectors_u.IsUnchanged()) copy->vectors = std::move(vectors_u).ValueUnchecked();
  if (!indices_u.IsUnchanged()) copy->indices = std::move(indices_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

// Arithmetic, comparison, and logical binary nodes have the same two-child default.
template <typename Visitor, typename Node>
ffi::Optional<VisitInterrupt> VisitExprDefault(Visitor* visitor, const Node* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(node->b));
  return std::nullopt;
}

template <typename Mutator, typename Node>
UnchangedOr<PrimExpr> MutateExprDefault(Mutator* mutator, const Node* node,
                                        InplaceMode inplace_mode) {
  auto a_u = mutator->Mutate(node->a, inplace_mode);
  auto b_u = mutator->Mutate(node->b, inplace_mode);
  if (a_u.UnchangedOrSameAs(node->a) && b_u.UnchangedOrSameAs(node->b)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<Node*>(node);
    if (!a_u.IsUnchanged()) writable->a = std::move(a_u).ValueUnchecked();
    if (!b_u.IsUnchanged()) writable->b = std::move(b_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<Node>(*node);
  if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();
  if (!b_u.IsUnchanged()) copy->b = std::move(b_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

}  // namespace details
}  // namespace tvm
#endif  // TVM_IR_EXPR_FUNCTOR_IMPL_H_
