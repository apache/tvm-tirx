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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/script/printer/doc_translator.h>

#include <cstring>
#include <limits>
#include <optional>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace detail {

void EmitVarDefinition(DocTranslatorObj* d, const Var& var, ExprDoc rhs) {
  // Only this type's referenced Vars must precede its declaration.
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      var->ty, [&](const Var& dependency) -> ffi::Expected<ffi::WalkResult> {
        if (d->GetImplicitDefs().count(dependency)) {
          if (auto rhs = d->Translate(dependency, dependency)) {
            d->Emit(AssignDoc(VarDoc(d, dependency), rhs.value(), std::nullopt), dependency);
          }
        }
        return ffi::WalkResult::Skip();
      });
  d->Emit(AssignDoc(VarDoc(d, var), rhs, std::nullopt), var);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<VarObj>().attr(tvm::script::printer::type_attr::kDocTranslate.name,
                                              FDocTranslate::FromNative<&VarDocTranslate>());
}

ffi::Optional<ExprDoc> LambdaExprDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* lambda =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LambdaExprObj>(input);
  return d->WithVarScope([&]() {
    ffi::Array<IdDoc> args;
    ffi::Array<ExprDoc> types;
    for (const Var& var : lambda->vars) {
      args.push_back(VarDoc(d, var));
      types.push_back(d->Translate(var->ty).value());
    }
    ExprDoc body = d->Translate(lambda->body).value();
    return NamespaceDoc("ir")->Attr("Lambda")->Call({ListDoc(types), LambdaDoc(args, body)});
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<LambdaExprObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&LambdaExprDocTranslate>());
}

ffi::Array<Doc> TensorIndices(DocTranslatorObj* d, const ffi::Array<PrimExpr>& indices,
                              bool store) {
  ffi::Array<Doc> docs;
  for (const PrimExpr& index : indices) {
    if (const auto* ramp = index.as<prim::RampObj>(); store && ramp) {
      if (const auto* stride = ramp->stride.as<IntImmObj>()) {
        ffi::Optional<ExprDoc> step = std::nullopt;
        if (stride->value != 1) step = d->Translate(ramp->stride).value();
        SliceDoc slice(d->Translate(ramp->base).value(),
                       d->Translate(ramp->base + ramp->lanes * ramp->stride).value(), step);
        d->RecordOrigin(slice, index);
        docs.push_back(slice);
        continue;
      }
    }
    docs.push_back(d->Translate(index).value());
  }
  return docs;
}

namespace {

ffi::Optional<ExprDoc> TensorRegionDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object*) {
  const auto* region =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorRegionObj>(input);
  bool indexable = region->ty.as<TensorRegionTypeObj>() && !region->region.empty();
  using Slice =
      ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>>;
  ffi::Array<ffi::Variant<Slice, PrimExpr>> indices;
  ffi::Array<Doc> slices;
  bool has_slice = false;
  if (indexable) {
    for (size_t i = 0; i < region->region.size(); ++i) {
      const Range& range = region->region[i];
      if (range->begin.ty()->dtype.lanes != 1 || range->extent.ty()->dtype.lanes != 1) {
        indexable = false;
        break;
      }
      ExprDoc start = d->Translate(range->begin).value();
      const auto* extent = range->extent.as<IntImmObj>();
      bool point =
          extent && extent->value == 1 && ffi::StructuralEqual()(extent->ty, range->begin.ty());
      if (point && (has_slice || i + 1 < region->region.size())) {
        slices.push_back(start);
        indices.push_back(range->begin);
      } else {
        try {
          PrimExpr stop = range->begin + range->extent;
          // Scalar constructors accept signed 64-bit Python literals. A computed
          // endpoint outside that range cannot reconstruct through indexing.
          if (const auto* imm = stop.as<IntImmObj>();
              imm && (imm->value > std::numeric_limits<int64_t>::max() ||
                      imm->value < std::numeric_limits<int64_t>::min())) {
            indexable = false;
            break;
          }
          slices.push_back(SliceDoc(start, d->Translate(stop).value(), std::nullopt));
          indices.push_back(Slice(range->begin, stop, std::nullopt));
          has_slice = true;
        } catch (const ffi::Error&) {
          indexable = false;
          break;
        }
      }
    }
  }
  if (indexable) {
    try {
      // Ask the existing type-directed subscription owner whether this syntax
      // reconstructs the exact region, without depending on a dialect type.
      static const auto realize = ffi::Function::GetGlobalRequired("ir.SubscriptExprRealize");
      ffi::Any restored = realize(region->source, indices, region->loc);
      if (ffi::StructuralEqual()(restored, ffi::GetRef<TensorRegion>(region))) {
        return d->Translate(region->source).value()[slices];
      }
    } catch (const ffi::Error&) {
      // Unsupported subscription or a different expression kind needs explicit construction.
    }
  }

  ffi::Array<ExprDoc> ranges;
  for (const Range& range : region->region) {
    ranges.push_back(
        NamespaceDoc("ir")
            ->Attr("Range")
            ->Attr("from_min_extent")
            ->Call({d->Translate(range->begin).value(), d->Translate(range->extent).value()}));
  }
  return NamespaceDoc("ir")
      ->Attr("TensorRegion")
      ->Call({d->Translate(region->source).value(), ListDoc(ranges)}, {"ty"},
             {d->Translate(region->ty).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TensorRegionObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&TensorRegionDocTranslate>());
}

ffi::Optional<ExprDoc> TensorLoadDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* load =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorLoadObj>(input);
  return IndexDoc(d->Translate(load->source).value(), TensorIndices(d, load->indices));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TensorLoadObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&TensorLoadDocTranslate>());
}

ffi::Optional<ExprDoc> TupleDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                         const ffi::Object*) {
  const auto* tuple =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleObj>(input);
  ffi::Array<ExprDoc> fields;
  for (const Expr& field : tuple->fields) fields.push_back(d->Translate(field).value());
  return TupleDoc(fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TupleObj>().attr(tvm::script::printer::type_attr::kDocTranslate.name,
                                                FDocTranslate::FromNative<&TupleDocTranslate>());
}

ffi::Optional<ExprDoc> TupleGetItemDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object*) {
  const auto* item =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleGetItemObj>(input);
  return d->Translate(item->tuple).value()[{LiteralDoc::Int(item->index, std::nullopt)}];
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TupleGetItemObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&TupleGetItemDocTranslate>());
}

template <typename T, OperationDocObj::Kind kind,
          PrimExpr (*operation)(PrimExpr, PrimExpr, Location)>
ffi::Optional<ExprDoc> BinaryOpDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object*) {
  const auto* node = ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const T>(input);
  ExprDoc a = d->Translate(node->a).value();
  ExprDoc b = d->Translate(node->b).value();
  // Replay operator overloads before choosing sugar. Even one constant can
  // simplify an explicit node, while two plain Python literals fold earlier.
  if (!(a.as<LiteralDocObj>() && b.as<LiteralDocObj>())) {
    try {
      PrimExpr replay = operation(node->a, node->b, UnknownLoc());
      if (const auto* result = replay.template as<T>();
          result && result->a.same_as(node->a) && result->b.same_as(node->b)) {
        return OperationDoc(kind, {a, b});
      }
    } catch (const ffi::Error&) {
      // A constructor can preserve operands rejected by a simplifying helper.
    }
  }
  return NamespaceDoc("tirx")->Attr(std::strrchr(T::_type_key, '.') + 1)->Call({a, b});
}

template <typename T, OperationDocObj::Kind kind, PrimExpr (*operation)(PrimExpr, Location)>
ExprDoc UnaryOpDocTranslate(DocTranslatorObj* d, const T* node) {
  ExprDoc value = d->Translate(node->a).value();
  if (!value.as<LiteralDocObj>()) {
    try {
      PrimExpr replay = operation(node->a, UnknownLoc());
      if (const auto* result = replay.template as<T>(); result && result->a.same_as(node->a)) {
        return OperationDoc(kind, {value});
      }
    } catch (const ffi::Error&) {
      // Preserve the explicit node even when the overloaded spelling rejects it.
    }
  }
  return NamespaceDoc("tirx")->Attr(std::strrchr(T::_type_key, '.') + 1)->Call({value});
}

template <typename T, PrimExpr (*operation)(PrimExpr, PrimExpr, Location)>
ExprDoc BinaryHelperDocTranslate(DocTranslatorObj* d, const T* node, const char* helper) {
  ExprDoc a = d->Translate(node->a).value();
  ExprDoc b = d->Translate(node->b).value();
  try {
    PrimExpr replay = operation(node->a, node->b, UnknownLoc());
    if (const auto* result = replay.template as<T>();
        result && result->a.same_as(node->a) && result->b.same_as(node->b)) {
      return NamespaceDoc("tirx")->Attr(helper)->Call({a, b});
    }
  } catch (const ffi::Error&) {
    // Keep the explicit constructor when the helper cannot preserve its inputs.
  }
  return NamespaceDoc("tirx")->Attr(std::strrchr(T::_type_key, '.') + 1)->Call({a, b});
}

ffi::Optional<ExprDoc> BitwiseNotDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::BitwiseNotObj>(input);
  return UnaryOpDocTranslate<prim::BitwiseNotObj, OperationDocObj::Kind::kInvert, tvm::bitwise_neg>(
      d, node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::BitwiseNotObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BitwiseNotDocTranslate>());
}

ffi::Optional<ExprDoc> NotDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::NotObj>(input);
  return UnaryOpDocTranslate<prim::NotObj, OperationDocObj::Kind::kNot, tvm::logical_not>(d, node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::NotObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&NotDocTranslate>());
}

ffi::Optional<ExprDoc> StringImmDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* imm =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const StringImmObj>(input);
  ExprDoc literal = LiteralDoc::Str(imm->value, std::nullopt);
  if (d->GetOrCreateExtraState<bool>("ir.explicit_value_type")) {
    return NamespaceDoc("ir")->Attr("StringImm")->Call({literal});
  }
  return literal;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<StringImmObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&StringImmDocTranslate>());
}

ffi::Optional<ExprDoc> CastDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object*) {
  const auto* cast =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::CastObj>(input);
  return NamespaceDoc("tirx")->Attr("Cast")->Call(
      {LiteralDoc::DataType(cast->ty.as_or_throw<PrimType>()->dtype, std::nullopt),
       d->Translate(cast->value).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::CastObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&CastDocTranslate>());
}

ffi::Optional<ExprDoc> SelectDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::SelectObj>(input);
  return NamespaceDoc("tirx")->Attr("Select")->Call({d->Translate(node->condition).value(),
                                                     d->Translate(node->true_value).value(),
                                                     d->Translate(node->false_value).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::SelectObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&SelectDocTranslate>());
}

ffi::Optional<ExprDoc> RampDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::RampObj>(input);
  return NamespaceDoc("tirx")->Attr("Ramp")->Call({d->Translate(node->base).value(),
                                                   d->Translate(node->stride).value(),
                                                   d->Translate(node->lanes).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::RampObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&RampDocTranslate>());
}

ffi::Optional<ExprDoc> BroadcastDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::BroadcastObj>(input);
  return NamespaceDoc("tirx")
      ->Attr("Broadcast")
      ->Call({d->Translate(node->value).value(), d->Translate(node->lanes).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::BroadcastObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BroadcastDocTranslate>());
}

ffi::Optional<ExprDoc> ShuffleDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::ShuffleObj>(input);
  ExprDoc vectors = d->Translate(node->vectors).value();
  ExprDoc indices = d->Translate(node->indices).value();
  d->RecordOrigin(vectors, node->vectors);
  d->RecordOrigin(indices, node->indices);
  return NamespaceDoc("tirx")->Attr("Shuffle")->Call({vectors, indices});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::ShuffleObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&ShuffleDocTranslate>());
}

ffi::Optional<ExprDoc> LetDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::LetObj>(input);
  ExprDoc value = d->Translate(node->value).value();
  IdDoc var = VarDoc(d, node->var);
  if (auto type = node->var->ty.as<PrimType>()) {
    ExprDoc declaration = NamespaceDoc("ir")->Attr("dynamic")->Call(
        {LiteralDoc::Str(node->var->name, std::nullopt)}, {"dtype"},
        {LiteralDoc::DataType(type.value()->dtype, std::nullopt)});
    d->Emit(AssignDoc(var, declaration, std::nullopt), node->var);
  }
  ExprDoc body = d->Translate(node->body).value();
  return NamespaceDoc("tirx")->Attr("Let")->Call({body}, {"where"}, {DictDoc({var}, {value})});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::LetObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&LetDocTranslate>());
}

ffi::Optional<ExprDoc> DivDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::DivObj>(input);
  PrimType a_type = node->a.ty();
  PrimType b_type = node->b.ty();
  if (a_type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
      b_type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return NamespaceDoc("tirx")->Attr("Div")->Call(
        {d->Translate(node->a).value(), d->Translate(node->b).value()});
  }
  return BinaryOpDocTranslate<prim::DivObj, OperationDocObj::Kind::kDiv, tvm::div>(d, input,
                                                                                   destination);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::DivObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&DivDocTranslate>());
  ffi::reflection::TypeAttrDef<prim::AddObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::AddObj, OperationDocObj::Kind::kAdd, tvm::add>>());
  ffi::reflection::TypeAttrDef<prim::SubObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::SubObj, OperationDocObj::Kind::kSub, tvm::sub>>());
  ffi::reflection::TypeAttrDef<prim::MulObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::MulObj, OperationDocObj::Kind::kMult, tvm::mul>>());
  ffi::reflection::TypeAttrDef<prim::FloorDivObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BinaryOpDocTranslate<
          prim::FloorDivObj, OperationDocObj::Kind::kFloorDiv, tvm::floordiv>>());
  ffi::reflection::TypeAttrDef<prim::FloorModObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::FloorModObj, OperationDocObj::Kind::kMod, tvm::floormod>>());
  ffi::reflection::TypeAttrDef<prim::LShiftObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BinaryOpDocTranslate<
          prim::LShiftObj, OperationDocObj::Kind::kLShift, tvm::left_shift>>());
  ffi::reflection::TypeAttrDef<prim::RShiftObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BinaryOpDocTranslate<
          prim::RShiftObj, OperationDocObj::Kind::kRShift, tvm::right_shift>>());
  ffi::reflection::TypeAttrDef<prim::BitwiseAndObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BinaryOpDocTranslate<
          prim::BitwiseAndObj, OperationDocObj::Kind::kBitAnd, tvm::bitwise_and>>());
  ffi::reflection::TypeAttrDef<prim::BitwiseOrObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BinaryOpDocTranslate<
          prim::BitwiseOrObj, OperationDocObj::Kind::kBitOr, tvm::bitwise_or>>());
  ffi::reflection::TypeAttrDef<prim::BitwiseXorObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BinaryOpDocTranslate<
          prim::BitwiseXorObj, OperationDocObj::Kind::kBitXor, tvm::bitwise_xor>>());
  ffi::reflection::TypeAttrDef<prim::LTObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::LTObj, OperationDocObj::Kind::kLt, tvm::less>>());
  ffi::reflection::TypeAttrDef<prim::LEObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::LEObj, OperationDocObj::Kind::kLtE, tvm::less_equal>>());
  ffi::reflection::TypeAttrDef<prim::EQObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::EQObj, OperationDocObj::Kind::kEq, tvm::equal>>());
  ffi::reflection::TypeAttrDef<prim::NEObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::NEObj, OperationDocObj::Kind::kNotEq, tvm::not_equal>>());
  ffi::reflection::TypeAttrDef<prim::GTObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::GTObj, OperationDocObj::Kind::kGt, tvm::greater>>());
  ffi::reflection::TypeAttrDef<prim::GEObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::GEObj, OperationDocObj::Kind::kGtE, tvm::greater_equal>>());
  ffi::reflection::TypeAttrDef<prim::AndObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::AndObj, OperationDocObj::Kind::kAnd, tvm::logical_and>>());
  ffi::reflection::TypeAttrDef<prim::OrObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::OrObj, OperationDocObj::Kind::kOr, tvm::logical_or>>());
}

ffi::Optional<ExprDoc> ModDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::ModObj>(input);
  return BinaryHelperDocTranslate<prim::ModObj, tvm::truncmod>(d, node, "truncmod");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::ModObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&ModDocTranslate>());
}

ffi::Optional<ExprDoc> MinDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::MinObj>(input);
  return BinaryHelperDocTranslate<prim::MinObj, tvm::min>(d, node, "min");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::MinObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&MinDocTranslate>());
}

ffi::Optional<ExprDoc> MaxDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::MaxObj>(input);
  return BinaryHelperDocTranslate<prim::MaxObj, tvm::max>(d, node, "max");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::MaxObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&MaxDocTranslate>());
}

}  // namespace

}  // namespace detail

ffi::Optional<ExprDoc> VarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination) {
  Var var = input.cast<Var>();
  const VarObj* node = var.get();
  IdDoc id = d->VarGetOrAllocId(var, false);
  if (destination == node && d->GetImplicitDefs().count(var)) {
    // Promote before translating the type, which may refer back to this Var.
    VarDoc(d, var);
    ExprDoc rhs(ffi::UnsafeInit{});
    if (auto primitive = var->ty.as<PrimType>()) {
      rhs = NamespaceDoc("ir")->Attr("dynamic")->Call(
          {LiteralDoc::Str(var->name, std::nullopt)}, {"dtype"},
          {LiteralDoc::DataType(primitive.value()->dtype, std::nullopt)});
    } else {
      // A type expression or annotation alone does not construct a free Var.
      rhs = NamespaceDoc("ir")->Attr("Var")->Call(
          {LiteralDoc::Str(var->name, std::nullopt), d->Translate(var->ty).value()});
    }
    detail::EmitVarDefinition(d, var, rhs);
    return std::nullopt;
  }
  return IdDoc(id->name);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
