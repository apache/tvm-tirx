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
 * \file src/ir/type.cc
 * \brief Common type system AST nodes throughout the IR.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/native_function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>

#include <cstdint>
#include <unordered_map>

namespace tvm {

namespace {

DLDataType ScalableVectorDType(DLDataTypeCode code, int bits, int lanes) {
  TVM_FFI_ICHECK_GT(lanes, 1) << "Invalid value for vscale factor " << lanes;
  TVM_FFI_ICHECK_LT(lanes, 32768);
  return DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                    static_cast<uint16_t>(-lanes)};
}

uint32_t PackDataTypeKey(DLDataType dtype) {
  return (static_cast<uint32_t>(dtype.code) << 24) | (static_cast<uint32_t>(dtype.bits) << 16) |
         static_cast<uint32_t>(dtype.lanes);
}

ffi::ObjectPtr<PrimTypeObj> GetCachedPrimTypeObj(DLDataType dtype) {
  thread_local std::unordered_map<uint32_t, ffi::ObjectPtr<PrimTypeObj>> cache;
  uint32_t key = PackDataTypeKey(dtype);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  ffi::ObjectPtr<PrimTypeObj> node = ffi::make_object<PrimTypeObj>();
  node->dtype = dtype;
  return cache.emplace(key, std::move(node)).first->second;
}

// Structural traversal hooks

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TypeVisit(
    ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // Field-less types are leaves; loc is ignored debug metadata.
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TypeMutate(ffi::StructuralMutatorObj*,
                                                                    ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> OpaqueTypeVisit(
    ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // OpaqueType is a field-less construction-time marker; loc is ignored debug metadata.
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> OpaqueTypeMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> OpaqueTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE int64_t PrimTypeAnyHash(const ffi::Any& src) {
  return static_cast<int64_t>(PackDataTypeKey(src.cast<PrimType>()->dtype));
}

TVM_FFI_INLINE bool PrimTypeAnyEqual(const ffi::Any& lhs, const ffi::Any& rhs) {
  return lhs.cast<PrimType>()->dtype == rhs.cast<PrimType>()->dtype;
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> PrimTypeVisit(
    ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // dtype is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> PrimTypeMutate(ffi::StructuralMutatorObj*,
                                                                        ffi::AnyView) noexcept {
  // dtype is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> PrimTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // dtype is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> PtrTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: scope (canonical enum identity)
  const PtrTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PtrTypeObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->element_type));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> PtrTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: scope (canonical enum identity)
  const PtrTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PtrTypeObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_element_type_u,
                                    mutator->MutateExpected(self->element_type));
  if (mapped_element_type_u.UnchangedOrSameAs(self->element_type)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<PtrTypeObj> copy = ffi::make_object<PtrTypeObj>(*self);
  if (!mapped_element_type_u.IsUnchanged())
    copy->element_type = std::move(mapped_element_type_u).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> PtrTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: scope (canonical enum identity)
  PtrTypeObj* self = const_cast<PtrTypeObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PtrTypeObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Type>, mapped_element_type_u,
      mutator->MutateExpected(self->element_type, ffi::InplaceMode::kAllow));
  if (!mapped_element_type_u.UnchangedOrSameAs(self->element_type)) {
    self->element_type = std::move(mapped_element_type_u).ValueUnchecked();
  }
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> FuncTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const FuncTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->arg_types));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ret_type));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FuncTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const FuncTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, mapped_arg_types_u,
                                    mutator->MutateExpected(self->arg_types));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret_type_u,
                                    mutator->MutateExpected(self->ret_type));
  if (mapped_arg_types_u.UnchangedOrSameAs(self->arg_types) &&
      mapped_ret_type_u.UnchangedOrSameAs(self->ret_type)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<FuncTypeObj> copy = ffi::make_object<FuncTypeObj>(*self);
  if (!mapped_arg_types_u.IsUnchanged())
    copy->arg_types = std::move(mapped_arg_types_u).ValueUnchecked();
  if (!mapped_ret_type_u.IsUnchanged())
    copy->ret_type = std::move(mapped_ret_type_u).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FuncTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  FuncTypeObj* self = const_cast<FuncTypeObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<Type>>, mapped_arg_types_u,
      mutator->MutateExpected(self->arg_types, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Type>, mapped_ret_type_u,
      mutator->MutateExpected(self->ret_type, ffi::InplaceMode::kAllow));
  if (!mapped_arg_types_u.UnchangedOrSameAs(self->arg_types)) {
    self->arg_types = std::move(mapped_arg_types_u).ValueUnchecked();
  }
  if (!mapped_ret_type_u.UnchangedOrSameAs(self->ret_type)) {
    self->ret_type = std::move(mapped_ret_type_u).ValueUnchecked();
  }
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TupleTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TupleTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleTypeObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->fields));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TupleTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TupleTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleTypeObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, mapped_fields_u,
                                    mutator->MutateExpected(self->fields));
  if (mapped_fields_u.UnchangedOrSameAs(self->fields)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<TupleTypeObj> copy = ffi::make_object<TupleTypeObj>(*self);
  if (!mapped_fields_u.IsUnchanged()) copy->fields = std::move(mapped_fields_u).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TupleTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  TupleTypeObj* self = const_cast<TupleTypeObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleTypeObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<Type>>, mapped_fields_u,
      mutator->MutateExpected(self->fields, ffi::InplaceMode::kAllow));
  if (!mapped_fields_u.UnchangedOrSameAs(self->fields)) {
    self->fields = std::move(mapped_fields_u).ValueUnchecked();
  }
  return ffi::Unchanged();
}

}  // namespace

Type Type::Missing() { return MissingType(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TypeObj::RegisterReflection();
  refl::TypeAttrDef<TypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&TypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TypeMaybeInplaceMutate>());
}

MissingType::MissingType() : Type(ffi::UnsafeInit{}) {
  static const auto singleton = ffi::make_object<MissingTypeObj>();
  data_ = singleton;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MissingTypeObj::RegisterReflection();
  refl::TypeAttrDef<MissingTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&TypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TypeMaybeInplaceMutate>());
  refl::GlobalDef().def("ir.MissingType", []() { return MissingType(); });
}

NoneType::NoneType() : Type(ffi::UnsafeInit{}) {
  static const auto singleton = ffi::make_object<NoneTypeObj>();
  data_ = singleton;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  NoneTypeObj::RegisterReflection();
  refl::TypeAttrDef<NoneTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&TypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TypeMaybeInplaceMutate>());
  refl::GlobalDef().def("ir.NoneType", []() { return NoneType(); });
}

AnyType::AnyType(Location loc) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<AnyTypeObj> n = ffi::make_object<AnyTypeObj>();
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AnyTypeObj::RegisterReflection();
  refl::TypeAttrDef<AnyTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&TypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TypeMaybeInplaceMutate>());
  refl::GlobalDef().def("ir.AnyType", [](Location loc) { return AnyType(loc); });
}

TensorRegionType::TensorRegionType() : Type(ffi::UnsafeInit{}) {
  static const auto singleton = ffi::make_object<TensorRegionTypeObj>();
  data_ = singleton;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorRegionTypeObj::RegisterReflection();
  refl::TypeAttrDef<TensorRegionTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&TypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TypeMaybeInplaceMutate>());
  refl::GlobalDef().def("ir.TensorRegionType", []() { return TensorRegionType(); });
}

OpaqueType::OpaqueType() : Type(ffi::UnsafeInit{}) { data_ = ffi::make_object<OpaqueTypeObj>(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  OpaqueTypeObj::RegisterReflection();
  refl::TypeAttrDef<OpaqueTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&OpaqueTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&OpaqueTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&OpaqueTypeMaybeInplaceMutate>());

  refl::GlobalDef().def("ir.OpaqueType", []() { return OpaqueType(); });
}

// PrimType
PrimType::PrimType(DLDataType dtype) : Type(ffi::UnsafeInit{}) {
  bool is_opaque_handle = dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLOpaqueHandle);
  bool is_void = is_opaque_handle && dtype.bits == 0 && dtype.lanes == 0;
  TVM_FFI_CHECK(!is_opaque_handle || is_void, TypeError)
      << "PrimType cannot represent an opaque pointer; use PtrType::VoidPtrType()";
  data_ = GetCachedPrimTypeObj(dtype);
}

PrimType::PrimType(DLDataTypeCode code, int bits, int lanes)
    : PrimType(DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                          static_cast<uint16_t>(lanes)}) {}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  PrimTypeObj::RegisterReflection();
  refl::TypeAttrDef<PrimTypeObj>()
      .native_fn_attr<&PrimTypeAnyHash>(ffi::type_attr::kAnyHash)
      .native_fn_attr<&PrimTypeAnyEqual>(ffi::type_attr::kAnyEqual)
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&PrimTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&PrimTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&PrimTypeMaybeInplaceMutate>());

  refl::GlobalDef().def("ir.PrimType", [](DLDataType dtype) { return PrimType(dtype); });
}

PrimType PrimType::Int(int bits, int lanes) {
  if (lanes == 1) {
    if (bits == 32) {
      thread_local PrimType i32_ty(DLDataType{kDLInt, 32, 1});
      return i32_ty;
    }
    if (bits == 64) {
      thread_local PrimType i64_ty(DLDataType{kDLInt, 64, 1});
      return i64_ty;
    }
  }
  return PrimType(DLDataType{kDLInt, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::UInt(int bits, int lanes) {
  return PrimType(DLDataType{kDLUInt, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Float(int bits, int lanes) {
  if (bits == 32 && lanes == 1) {
    thread_local PrimType f32_ty(DLDataType{kDLFloat, 32, 1});
    return f32_ty;
  }
  return PrimType(DLDataType{kDLFloat, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::BFloat(int bits, int lanes) {
  return PrimType(DLDataType{kDLBfloat, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Bool(int lanes) {
  if (lanes == 1) {
    thread_local PrimType bool_ty(DLDataType{kDLBool, 8, 1});
    return bool_ty;
  }
  return PrimType(DLDataType{kDLBool, 8, static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Void() { return PrimType(DLDataType{kDLOpaqueHandle, 0, 0}); }

PrimType PrimType::ScalableVector(DLDataTypeCode code, int bits, int lanes) {
  return PrimType(ScalableVectorDType(code, bits, lanes));
}

StringType::StringType() : Type(ffi::UnsafeInit{}) { data_ = ffi::make_object<StringTypeObj>(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  StringTypeObj::RegisterReflection();
  refl::TypeAttrDef<StringTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&TypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TypeMaybeInplaceMutate>());
  refl::GlobalDef().def("ir.StringType", []() { return StringType(); });
}

// PtrType
PtrType::PtrType(Type element_type, runtime::MemoryScope scope) : Type(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(!element_type.as<MissingType>().has_value())
      << "PtrType element_type cannot be Type::Missing()";
  ffi::ObjectPtr<PtrTypeObj> n = ffi::make_object<PtrTypeObj>();
  n->scope = std::move(scope);
  n->element_type = std::move(element_type);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  PtrTypeObj::RegisterReflection();
  refl::TypeAttrDef<PtrTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&PtrTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&PtrTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&PtrTypeMaybeInplaceMutate>());

  refl::GlobalDef().def("ir.PtrType", [](Type element_type, runtime::MemoryScope scope = {}) {
    return PtrType(element_type, scope);
  });
}

PtrType PtrType::VoidPtrType(runtime::MemoryScope scope) {
  return PtrType(PrimType::Void(), std::move(scope));
}

FuncType::FuncType(tvm::ffi::Array<Type> arg_types, Type ret_type, Location loc)
    : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<FuncTypeObj> n = ffi::make_object<FuncTypeObj>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FuncTypeObj::RegisterReflection();
  refl::TypeAttrDef<FuncTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&FuncTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&FuncTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&FuncTypeMaybeInplaceMutate>());

  refl::GlobalDef().def("ir.FuncType", [](tvm::ffi::Array<Type> arg_types, Type ret_type) {
    return FuncType(arg_types, ret_type);
  });
}

TupleType::TupleType(ffi::Array<Type> fields, Location loc) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TupleTypeObj> n = ffi::make_object<TupleTypeObj>();
  n->fields = std::move(fields);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TupleTypeObj::RegisterReflection();
  refl::TypeAttrDef<TupleTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TupleTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TupleTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TupleTypeMaybeInplaceMutate>());

  refl::GlobalDef().def(
      "ir.TupleType", [](ffi::Array<Type> fields, Location loc) { return TupleType(fields, loc); });
}

TupleType TupleType::Empty() { return TupleType(ffi::Array<Type>()); }

}  // namespace tvm
