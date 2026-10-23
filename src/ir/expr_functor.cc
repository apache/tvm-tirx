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
#include <tvm/ir/expr_functor.h>

namespace tvm {

void ExprVisitor::InitVTable(VTable* vtable) {
  ObjectVisitor::InitVTable(vtable);
  SetDispatch<ExprVisitor, OpaqueExprObj>(vtable);
  SetDispatch<ExprVisitor, TupleObj>(vtable);
  SetDispatch<ExprVisitor, TupleGetItemObj>(vtable);
  SetDispatch<ExprVisitor, TensorLoadObj>(vtable);
  SetDispatch<ExprVisitor, TensorRegionObj>(vtable);
  SetDispatch<ExprVisitor, VarObj>(vtable);
  SetDispatch<ExprVisitor, GlobalVarObj>(vtable);
  SetDispatch<ExprVisitor, CallObj>(vtable);
  SetDispatch<ExprVisitor, GenericConstObj>(vtable);
  SetDispatch<ExprVisitor, IntImmObj>(vtable);
  SetDispatch<ExprVisitor, FloatImmObj>(vtable);
  SetDispatch<ExprVisitor, OpObj>(vtable);
  SetDispatch<ExprVisitor, StringImmObj>(vtable);
  SetDispatch<ExprVisitor, DataTypeImmObj>(vtable);
  SetDispatch<ExprVisitor, prim::CastObj>(vtable);
  SetDispatch<ExprVisitor, prim::AddObj>(vtable);
  SetDispatch<ExprVisitor, prim::LShiftObj>(vtable);
  SetDispatch<ExprVisitor, prim::RShiftObj>(vtable);
  SetDispatch<ExprVisitor, prim::BitwiseAndObj>(vtable);
  SetDispatch<ExprVisitor, prim::BitwiseOrObj>(vtable);
  SetDispatch<ExprVisitor, prim::BitwiseXorObj>(vtable);
  SetDispatch<ExprVisitor, prim::BitwiseNotObj>(vtable);
  SetDispatch<ExprVisitor, prim::SubObj>(vtable);
  SetDispatch<ExprVisitor, prim::MulObj>(vtable);
  SetDispatch<ExprVisitor, prim::DivObj>(vtable);
  SetDispatch<ExprVisitor, prim::ModObj>(vtable);
  SetDispatch<ExprVisitor, prim::FloorDivObj>(vtable);
  SetDispatch<ExprVisitor, prim::FloorModObj>(vtable);
  SetDispatch<ExprVisitor, prim::MinObj>(vtable);
  SetDispatch<ExprVisitor, prim::MaxObj>(vtable);
  SetDispatch<ExprVisitor, prim::EQObj>(vtable);
  SetDispatch<ExprVisitor, prim::NEObj>(vtable);
  SetDispatch<ExprVisitor, prim::LTObj>(vtable);
  SetDispatch<ExprVisitor, prim::LEObj>(vtable);
  SetDispatch<ExprVisitor, prim::GTObj>(vtable);
  SetDispatch<ExprVisitor, prim::GEObj>(vtable);
  SetDispatch<ExprVisitor, prim::AndObj>(vtable);
  SetDispatch<ExprVisitor, prim::OrObj>(vtable);
  SetDispatch<ExprVisitor, prim::NotObj>(vtable);
  SetDispatch<ExprVisitor, prim::SelectObj>(vtable);
  SetDispatch<ExprVisitor, prim::LetObj>(vtable);
  SetDispatch<ExprVisitor, prim::RampObj>(vtable);
  SetDispatch<ExprVisitor, prim::BroadcastObj>(vtable);
  SetDispatch<ExprVisitor, prim::ShuffleObj>(vtable);
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const OpaqueExprObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const TupleObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->fields));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const TupleGetItemObj* node) {
  // skips: index
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->tuple));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const TensorLoadObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->indices));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const TensorRegionObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->region));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const VarObj* node) {
  // Primitive types have no children; dynamic type fields are visited.
  if (!node->ty.as<PrimTypeObj>()) {
    // Clamp Simple for dynamic type fields; Pattern continues through them.
    if (this->def_region_kind() == kTVMFFIDefRegionKindSimple) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
          kTVMFFIDefRegionKindNone, [&]() { return this->Visit(node->ty); }));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
    }
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const GlobalVarObj* node) { return std::nullopt; }

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const CallObj* node) {
  // Skip constant attrs and primitive result types.
  if (!node->ty.as<PrimTypeObj>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  }
  // Interned operators have no children; function-valued operators are visited.
  if (!node->op.as<OpObj>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->op));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->args));
  // Empty type arguments are skipped, including the container callback.
  if (!node->ty_args.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty_args));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const GenericConstObj* node) {
  return this->Visit(node->ty);
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const IntImmObj* node) { return std::nullopt; }

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const FloatImmObj* node) { return std::nullopt; }

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const OpObj* node) { return std::nullopt; }

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const StringImmObj* node) { return std::nullopt; }
ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const DataTypeImmObj* node) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::CastObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->value));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::AddObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::LShiftObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::RShiftObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BitwiseAndObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BitwiseOrObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BitwiseXorObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::SubObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::MulObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::DivObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::ModObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::FloorDivObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::FloorModObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::MinObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::MaxObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::EQObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::NEObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::LTObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::LEObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::GTObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::GEObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::AndObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::OrObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::NotObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BitwiseNotObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::SelectObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->true_value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->false_value));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::LetObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(node->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->body));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::RampObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->base));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->stride));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->lanes));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BroadcastObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->lanes));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::ShuffleObj* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->vectors));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->indices));
  return std::nullopt;
}

void ExprMutator::InitVTable(VTable* vtable) {
  ObjectMutator::InitVTable(vtable);
  SetDispatch<ExprMutator, OpaqueExprObj>(vtable);
  SetDispatch<ExprMutator, TupleObj>(vtable);
  SetDispatch<ExprMutator, TupleGetItemObj>(vtable);
  SetDispatch<ExprMutator, TensorLoadObj>(vtable);
  SetDispatch<ExprMutator, TensorRegionObj>(vtable);
  SetDispatch<ExprMutator, VarObj>(vtable);
  SetDispatch<ExprMutator, GlobalVarObj>(vtable);
  SetDispatch<ExprMutator, CallObj>(vtable);
  SetDispatch<ExprMutator, GenericConstObj>(vtable);
  SetDispatch<ExprMutator, IntImmObj>(vtable);
  SetDispatch<ExprMutator, FloatImmObj>(vtable);
  SetDispatch<ExprMutator, OpObj>(vtable);
  SetDispatch<ExprMutator, StringImmObj>(vtable);
  SetDispatch<ExprMutator, DataTypeImmObj>(vtable);
  SetDispatch<ExprMutator, prim::CastObj>(vtable);
  SetDispatch<ExprMutator, prim::AddObj>(vtable);
  SetDispatch<ExprMutator, prim::LShiftObj>(vtable);
  SetDispatch<ExprMutator, prim::RShiftObj>(vtable);
  SetDispatch<ExprMutator, prim::BitwiseAndObj>(vtable);
  SetDispatch<ExprMutator, prim::BitwiseOrObj>(vtable);
  SetDispatch<ExprMutator, prim::BitwiseXorObj>(vtable);
  SetDispatch<ExprMutator, prim::BitwiseNotObj>(vtable);
  SetDispatch<ExprMutator, prim::SubObj>(vtable);
  SetDispatch<ExprMutator, prim::MulObj>(vtable);
  SetDispatch<ExprMutator, prim::DivObj>(vtable);
  SetDispatch<ExprMutator, prim::ModObj>(vtable);
  SetDispatch<ExprMutator, prim::FloorDivObj>(vtable);
  SetDispatch<ExprMutator, prim::FloorModObj>(vtable);
  SetDispatch<ExprMutator, prim::MinObj>(vtable);
  SetDispatch<ExprMutator, prim::MaxObj>(vtable);
  SetDispatch<ExprMutator, prim::EQObj>(vtable);
  SetDispatch<ExprMutator, prim::NEObj>(vtable);
  SetDispatch<ExprMutator, prim::LTObj>(vtable);
  SetDispatch<ExprMutator, prim::LEObj>(vtable);
  SetDispatch<ExprMutator, prim::GTObj>(vtable);
  SetDispatch<ExprMutator, prim::GEObj>(vtable);
  SetDispatch<ExprMutator, prim::AndObj>(vtable);
  SetDispatch<ExprMutator, prim::OrObj>(vtable);
  SetDispatch<ExprMutator, prim::NotObj>(vtable);
  SetDispatch<ExprMutator, prim::SelectObj>(vtable);
  SetDispatch<ExprMutator, prim::LetObj>(vtable);
  SetDispatch<ExprMutator, prim::RampObj>(vtable);
  SetDispatch<ExprMutator, prim::BroadcastObj>(vtable);
  SetDispatch<ExprMutator, prim::ShuffleObj>(vtable);
}

UnchangedOr<Expr> ExprMutator::Mutate_(const OpaqueExprObj* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
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

UnchangedOr<Expr> ExprMutator::Mutate_(const TupleObj* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  auto fields_u = Mutate(node->fields, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
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

UnchangedOr<Expr> ExprMutator::Mutate_(const TupleGetItemObj* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  auto tuple_u = Mutate(node->tuple, inplace_mode);
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

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const TensorLoadObj* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  Type result_ty = std::move(ty_u).ValueOrUnchanged(node->ty);
  TVM_FFI_CHECK(result_ty.as<PrimTypeObj>(), TypeError)
      << "TensorLoad requires a concrete PrimType";
  auto source_u = Mutate(node->source, inplace_mode);
  auto indices_u =
      Mutate(node->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
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

UnchangedOr<Expr> ExprMutator::Mutate_(const TensorRegionObj* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  auto source_u = Mutate(node->source, inplace_mode);
  auto region_u = Mutate(node->region, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
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

UnchangedOr<Expr> ExprMutator::Mutate_(const GlobalVarObj* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<Expr> ExprMutator::Mutate_(const CallObj* node, InplaceMode inplace_mode) {
  UnchangedOr<Type> ty_u = ffi::Unchanged();
  if (!node->ty.as<PrimTypeObj>()) {
    ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  }
  UnchangedOr<Expr> op_u = ffi::Unchanged();
  if (!node->op.as<OpObj>()) {
    op_u = Mutate(node->op, inplace_mode);
  }
  auto args_u = Mutate(node->args, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  UnchangedOr<ffi::Array<Type>> ty_args_u = ffi::Unchanged();
  if (!node->ty_args.empty()) {
    ty_args_u = Mutate(node->ty_args, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Type>>>();
  }
  if (ty_u.UnchangedOrSameAs(node->ty) && op_u.UnchangedOrSameAs(node->op) &&
      args_u.UnchangedOrSameAs(node->args) && ty_args_u.UnchangedOrSameAs(node->ty_args))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<CallObj*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!op_u.IsUnchanged()) writable->op = std::move(op_u).ValueUnchecked();
    if (!args_u.IsUnchanged()) writable->args = std::move(args_u).ValueUnchecked();
    if (!ty_args_u.IsUnchanged()) writable->ty_args = std::move(ty_args_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<CallObj>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!op_u.IsUnchanged()) copy->op = std::move(op_u).ValueUnchecked();
  if (!args_u.IsUnchanged()) copy->args = std::move(args_u).ValueUnchecked();
  if (!ty_args_u.IsUnchanged()) copy->ty_args = std::move(ty_args_u).ValueUnchecked();
  return Expr(std::move(copy));
}

UnchangedOr<Expr> ExprMutator::ReinferMutatedCallType(UnchangedOr<Expr> mutated,
                                                      const CallObj* original,
                                                      InplaceMode inplace_mode) {
  bool is_replacement = !mutated.IsUnchanged();
  const CallObj* call = is_replacement ? ffi::AnyView(mutated).as<CallObj>() : original;
  Type type = Call::ReinferType(call);
  if (call->ty.same_as(type)) return mutated;
  if (call->unique() && (is_replacement || inplace_mode == InplaceMode::kAllow)) {
    const_cast<CallObj*>(call)->ty = std::move(type);
    return mutated;
  }
  auto copy = ffi::make_object<CallObj>(*call);
  copy->ty = std::move(type);
  return Expr(std::move(copy));
}

UnchangedOr<Expr> ExprMutator::Mutate_(const GenericConstObj* node, InplaceMode inplace_mode) {
  auto ty = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  if (ty.UnchangedOrSameAs(node->ty)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<GenericConstObj*>(node)->ty = std::move(ty).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<GenericConstObj>(*node);
  copy->ty = std::move(ty).ValueUnchecked();
  return Expr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const IntImmObj* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const FloatImmObj* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<Expr> ExprMutator::Mutate_(const OpObj* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<Expr> ExprMutator::Mutate_(const StringImmObj* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}
UnchangedOr<Expr> ExprMutator::Mutate_(const DataTypeImmObj* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::CastObj* node, InplaceMode inplace_mode) {
  auto value_u = Mutate(node->value, inplace_mode);
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

#define TVM_IR_BINARY_MUTATE_IMPL(Name)                                                            \
  UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::Name##Obj* node,                          \
                                             InplaceMode inplace_mode) {                           \
    auto a_u = Mutate(node->a, inplace_mode);                                                      \
    auto b_u = Mutate(node->b, inplace_mode);                                                      \
    if (a_u.UnchangedOrSameAs(node->a) && b_u.UnchangedOrSameAs(node->b)) return ffi::Unchanged(); \
    if (inplace_mode == InplaceMode::kAllow) {                                                     \
      auto* writable = const_cast<prim::Name##Obj*>(node);                                         \
      if (!a_u.IsUnchanged()) writable->a = std::move(a_u).ValueUnchecked();                       \
      if (!b_u.IsUnchanged()) writable->b = std::move(b_u).ValueUnchecked();                       \
      return ffi::Unchanged();                                                                     \
    }                                                                                              \
    auto copy = ffi::make_object<prim::Name##Obj>(*node);                                          \
    if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();                             \
    if (!b_u.IsUnchanged()) copy->b = std::move(b_u).ValueUnchecked();                             \
    return PrimExpr(std::move(copy));                                                              \
  }
TVM_IR_BINARY_MUTATE_IMPL(Add)
TVM_IR_BINARY_MUTATE_IMPL(LShift)
TVM_IR_BINARY_MUTATE_IMPL(RShift)
TVM_IR_BINARY_MUTATE_IMPL(BitwiseAnd)
TVM_IR_BINARY_MUTATE_IMPL(BitwiseOr)
TVM_IR_BINARY_MUTATE_IMPL(BitwiseXor)
TVM_IR_BINARY_MUTATE_IMPL(Sub)
TVM_IR_BINARY_MUTATE_IMPL(Mul)
TVM_IR_BINARY_MUTATE_IMPL(Div)
TVM_IR_BINARY_MUTATE_IMPL(Mod)
TVM_IR_BINARY_MUTATE_IMPL(FloorDiv)
TVM_IR_BINARY_MUTATE_IMPL(FloorMod)
TVM_IR_BINARY_MUTATE_IMPL(Min)
TVM_IR_BINARY_MUTATE_IMPL(Max)
TVM_IR_BINARY_MUTATE_IMPL(EQ)
TVM_IR_BINARY_MUTATE_IMPL(NE)
TVM_IR_BINARY_MUTATE_IMPL(LT)
TVM_IR_BINARY_MUTATE_IMPL(LE)
TVM_IR_BINARY_MUTATE_IMPL(GT)
TVM_IR_BINARY_MUTATE_IMPL(GE)
TVM_IR_BINARY_MUTATE_IMPL(And)
TVM_IR_BINARY_MUTATE_IMPL(Or)
#undef TVM_IR_BINARY_MUTATE_IMPL

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::NotObj* node, InplaceMode inplace_mode) {
  auto a_u = Mutate(node->a, inplace_mode);
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

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::BitwiseNotObj* node,
                                           InplaceMode inplace_mode) {
  auto a_u = Mutate(node->a, inplace_mode);
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

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::SelectObj* node, InplaceMode inplace_mode) {
  auto condition_u = Mutate(node->condition, inplace_mode);
  auto true_value_u = Mutate(node->true_value, inplace_mode);
  auto false_value_u = Mutate(node->false_value, inplace_mode);
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

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::LetObj* node, InplaceMode inplace_mode) {
  Type old_value_type = node->value->ty;
  auto value_u = Mutate(node->value, inplace_mode);
  bool value_changed = !value_u.UnchangedOrSameAs(node->value);
  PrimExpr value = std::move(value_u).ValueOrUnchanged(node->value);
  value_changed |= !old_value_type.same_as(value->ty);
  Type old_var_type = node->var->ty;
  auto var_u = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                 return Mutate(node->var, inplace_mode);
               }).as_or_throw<UnchangedOr<Var>>();
  bool var_changed = !var_u.UnchangedOrSameAs(node->var);
  Var var = std::move(var_u).ValueOrUnchanged(node->var);
  var_changed |= !old_var_type.same_as(var->ty);
  if ((value_changed || var_changed) && !var->ty.same_as(value->ty)) {
    var = var.CopyWithType(value->ty);
  }
  if (!var.same_as(node->var)) VarRemapSet(node->var, var);
  auto body_u = Mutate(node->body, inplace_mode);
  PrimExpr body = std::move(body_u).ValueOrUnchanged(node->body);
  if (var.same_as(node->var) && value.same_as(node->value) && body.same_as(node->body) &&
      body->ty.same_as(node->ty))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::LetObj*>(node);
    writable->var = std::move(var);
    writable->value = std::move(value);
    writable->body = std::move(body);
    writable->ty = writable->body->ty;
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::LetObj>(*node);
  copy->var = std::move(var);
  copy->value = std::move(value);
  copy->body = std::move(body);
  copy->ty = copy->body->ty;
  return PrimExpr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::RampObj* node, InplaceMode inplace_mode) {
  auto base_u = Mutate(node->base, inplace_mode);
  auto stride_u = Mutate(node->stride, inplace_mode);
  auto lanes_u = Mutate(node->lanes, inplace_mode);
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

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::BroadcastObj* node,
                                           InplaceMode inplace_mode) {
  auto value_u = Mutate(node->value, inplace_mode);
  auto lanes_u = Mutate(node->lanes, inplace_mode);
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

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::ShuffleObj* node, InplaceMode inplace_mode) {
  auto vectors_u =
      Mutate(node->vectors, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto indices_u =
      Mutate(node->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
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

UnchangedOr<Expr> ExprMutator::Mutate_(const VarObj* node, InplaceMode inplace_mode) {
  if (TVM_FFI_PREDICT_TRUE(var_remap_.empty() && def_region_kind() == kTVMFFIDefRegionKindNone)) {
    return ffi::Unchanged();
  }
  ffi::Any remap_result = VarRemapGet(ffi::AnyView(node));
  if (remap_result.type_index() != ffi::TypeIndex::kTVMFFINone) {
    return std::move(remap_result).as_or_throw<UnchangedOr<Expr>>();
  }
  if (def_region_kind() == kTVMFFIDefRegionKindNone) return ffi::Unchanged();
  UnchangedOr<Expr> result_u = ffi::Unchanged();
  ffi::Any mapped_value = ffi::Unchanged();
  // PrimType has no children; dynamic type fields inherit Pattern but are visited outside Simple.
  if (!node->ty.as<PrimTypeObj>()) {
    UnchangedOr<ffi::Any> mapped_ty_result_u =
        def_region_kind() == kTVMFFIDefRegionKindSimple
            ? WithDefRegionKind(kTVMFFIDefRegionKindNone,
                                [&] { return Mutate(node->ty, inplace_mode); })
            : Mutate(node->ty, inplace_mode);
    auto mapped_ty_u = std::move(mapped_ty_result_u).as_or_throw<UnchangedOr<Type>>();
    if (!mapped_ty_u.UnchangedOrSameAs(node->ty)) {
      Expr mapped_expr{ffi::UnsafeInit{}};
      if (inplace_mode == InplaceMode::kAllow) {
        const_cast<VarObj*>(node)->ty = std::move(mapped_ty_u).ValueUnchecked();
        mapped_expr = ffi::GetRef<Expr>(node);
      } else {
        auto copy = ffi::make_object<VarObj>(*node);
        copy->ty = std::move(mapped_ty_u).ValueUnchecked();
        mapped_expr = Expr(std::move(copy));
      }
      result_u = mapped_expr;
      mapped_value = std::move(mapped_expr);
    }
  }
  if (!result_u.IsUnchanged() || def_region_kind() == kTVMFFIDefRegionKindPattern) {
    VarRemapSet(ffi::AnyView(node), mapped_value);
  }
  return result_u;
}

}  // namespace tvm
