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
 * \file tvm/ir/type.h
 * \brief IR/AST nodes for TVM types shared across IR variants.
 */
#ifndef TVM_IR_TYPE_H_
#define TVM_IR_TYPE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/location.h>
#include <tvm/runtime/memory_scope.h>

#include <string>

namespace tvm {

/*!
 * \brief Type is the base type of all types.
 *
 * TVM's type system contains following subclasses:
 *
 * - PrimType: type of primitive type values used in the low-level IR.
 * - FuncType: type of a function.
 * - TensorType: type of certain Tensor values in the expression.
 *
 * There are also advanced types to support generic(polymorphic types).
 * \sa Type
 */
class TypeObj : public ffi::Object {
 public:
  /*!
   * \brief Location that points to the original source code.
   *        Reserved debug information.
   */
  mutable Location loc = UnknownLoc();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // loc do not participate in structural equal and hash.
    refl::ObjectDef<TypeObj>().def_ro("loc", &TypeObj::loc, refl::DefaultValue(UnknownLoc()),
                                      refl::init(false), refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;

  static constexpr const uint32_t _type_child_slots = 14;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Type", TypeObj, ffi::Object);
};

/*!
 * \brief Managed reference to TypeObj.
 * \sa TypeObj
 */
class Type : public ffi::ObjectRef {
 public:
  /*!
   * \brief Request a registered IR type's constructor in its exposure destinations.
   * \param type_key The registered type key.
   * \note Call after ObjectDef finalization and all reflection and constructor registration.
   *       The first call caches the optional ir.expose_type receiver, including its absence.
   *       This permits standalone native use. Install the receiver before the first call to enable
   *       exposure; errors from an installed receiver propagate.
   */
  TVM_TIRX_DLL static void Expose(const ffi::String& type_key);

  /*!
   * \brief Request API exposure of a registered IR object type.
   * \tparam TypeObj The object class, not its reference wrapper.
   * \note Finish the ObjectDef statement and constructor registration before calling.
   *       Does nothing if the optional receiver was absent on the string helper's first call.
   * \code
   * ffi::reflection::ObjectDef<FooTypeObj>().def_ro("value", &FooTypeObj::value);
   * Type::Expose<FooTypeObj>();
   * \endcode
   */
  template <typename TypeObj>
  static inline void Expose() {
    Expose(TypeObj::_type_key);
  }

  /*! \brief Construct a MissingType for type information not yet populated. */
  TVM_TIRX_DLL static Type Missing();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Type, ffi::ObjectRef, TypeObj);
};

/*!
 * \brief Type information that has not been supplied or computed.
 *
 * MissingType is not a concrete type, wildcard, or inference variable. Unlike
 * AnyType and Void, it must be resolved before a fully typed IR boundary.
 */
class MissingTypeObj final : public TypeObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MissingTypeObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.MissingType", MissingTypeObj, TypeObj);
};

/*! \brief Managed reference to the MissingTypeObj singleton. */
class MissingType final : public Type {
 public:
  TVM_TIRX_DLL MissingType();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MissingType, Type, MissingTypeObj);
};

/*!
 * \brief Type marker for opaque construction-time expressions.
 *
 * Opaque values may be used while constructing IR, but must be lowered away
 * before the IR is considered complete.
 */
class OpaqueTypeObj final : public TypeObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OpaqueTypeObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.OpaqueType", OpaqueTypeObj, TypeObj);
};

/*! \brief Managed reference to OpaqueTypeObj. */
class OpaqueType final : public Type {
 public:
  TVM_TIRX_DLL OpaqueType();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OpaqueType, Type, OpaqueTypeObj);
};

/*!
 * \brief Primitive data types used in the low-level IR.
 *
 * PrimType represents primitive POD values and the void sentinel.
 *
 * \sa PrimType
 */
class PrimTypeObj final : public TypeObj {
 public:
  /*!
   * \brief The raw DLPack dtype represented by this primitive type.
   */
  DLDataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimTypeObj>().def_ro("dtype", &PrimTypeObj::dtype);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PrimType", PrimTypeObj, TypeObj);
};

/*
 * \brief Managed reference to PrimTypeObj.
 * \sa PrimTypeObj
 */
class PrimType final : public Type {
 public:
  /*!
   * \brief Construct from a raw DLPack dtype.
   * \param dtype The corresponding DLPack dtype.
   */
  TVM_TIRX_DLL explicit PrimType(DLDataType dtype);

  /*!
   * \brief Construct from DLPack dtype fields.
   * \param code The DLPack dtype code.
   * \param bits The scalar bit width.
   * \param lanes The fixed lane count.
   */
  TVM_TIRX_DLL PrimType(DLDataTypeCode code, int bits, int lanes = 1);

  /*! \brief Construct a signed integer type with fixed lanes. */
  TVM_TIRX_DLL static PrimType Int(int bits, int lanes = 1);
  /*! \brief Construct an unsigned integer type with fixed lanes. */
  TVM_TIRX_DLL static PrimType UInt(int bits, int lanes = 1);
  /*! \brief Construct a floating-point type with fixed lanes. */
  TVM_TIRX_DLL static PrimType Float(int bits, int lanes = 1);
  /*! \brief Construct a bfloat type with fixed lanes. */
  TVM_TIRX_DLL static PrimType BFloat(int bits, int lanes = 1);
  /*! \brief Construct a boolean type with fixed lanes. */
  TVM_TIRX_DLL static PrimType Bool(int lanes = 1);
  /*! \brief Construct the void sentinel type, encoded as handle(0, 0). */
  TVM_TIRX_DLL static PrimType Void();
  /*!
   * \brief Construct a scalable vector type.
   * \param code The DLPack dtype code.
   * \param bits The scalar bit width.
   * \param lanes The positive vscale factor to encode in the DLPack lane field.
   */
  TVM_TIRX_DLL static PrimType ScalableVector(DLDataTypeCode code, int bits, int lanes);

  /*! \return The DLPack dtype code. */
  TVM_FFI_INLINE DLDataTypeCode code() const {
    return static_cast<DLDataTypeCode>(static_cast<int>(get()->dtype.code));
  }

  /*! \return The scalar bit width. */
  TVM_FFI_INLINE int32_t bits() const { return get()->dtype.bits; }

  /*!
   * \return The fixed lane count.
   * \note Throws on scalable vector types, where the encoded lane field stores a vscale factor.
   */
  TVM_FFI_INLINE int32_t lanes() const {
    int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
    if (TVM_FFI_PREDICT_FALSE(encoded_lanes < 0)) {
      TVM_FFI_THROW(InternalError)
          << "Can't fetch the lanes of a scalable vector at a compile time.";
    }
    return encoded_lanes;
  }

  /*!
   * \brief Check the scalar element code and bit width.
   * \note Lane count and scalable-vector encoding are intentionally ignored.
   */
  TVM_FFI_INLINE bool MatchesElementType(DLDataTypeCode code, int bits) const {
    DLDataType dtype = get()->dtype;
    return dtype.code == static_cast<uint8_t>(code) && dtype.bits == bits;
  }

  /*!
   * \brief Check whether the dtype code matches any of the provided DLPack codes.
   * \note Bit width and lanes are intentionally ignored.
   */
  template <typename... Codes>
  TVM_FFI_INLINE bool MatchesCode(Codes... codes) const {
    uint8_t dtype_code = get()->dtype.code;
    return ((dtype_code == static_cast<uint8_t>(codes)) || ...);
  }

  /*! \brief Whether this type is a scalar, excluding fixed and scalable vectors. */
  TVM_FFI_INLINE bool IsScalar() const {
    int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
    return encoded_lanes == 1;
  }

  /*! \brief Whether this type is the void sentinel `handle(0, 0)`. */
  TVM_FFI_INLINE bool IsVoid() const {
    DLDataType dtype = get()->dtype;
    return dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLOpaqueHandle) && dtype.bits == 0 &&
           static_cast<int16_t>(dtype.lanes) == 0;
  }

  /*! \brief Whether this type is a scalable vector. */
  TVM_FFI_INLINE bool IsScalableVector() const {
    return static_cast<int16_t>(get()->dtype.lanes) < -1;
  }

  /*! \brief Whether this type is a fixed-length vector. */
  TVM_FFI_INLINE bool IsFixedLengthVector() const {
    return static_cast<int16_t>(get()->dtype.lanes) > 1;
  }

  /*!
   * \brief Return the number of bytes needed to store one value of this type.
   *
   * This uses the same packed sub-byte dtype sizing rule as runtime tensors.
   * Scalable vector types have no compile-time storage size and are rejected.
   */
  TVM_FFI_INLINE size_t StorageBytes() const {
    DLDataType dtype = get()->dtype;
    int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
    if (TVM_FFI_PREDICT_FALSE(encoded_lanes < 0)) {
      TVM_FFI_THROW(InternalError)
          << "Cannot compute compile-time storage bytes for non-fixed vector type " << dtype;
    }
    return static_cast<size_t>(
        (static_cast<uint64_t>(dtype.bits) * static_cast<uint64_t>(dtype.lanes) + 7) / 8);
  }

  /*! \brief Return the same type with a different dtype code, preserving bits and lanes. */
  TVM_FFI_INLINE PrimType WithCode(DLDataTypeCode code) const {
    DLDataType dtype = get()->dtype;
    int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
    if (encoded_lanes < -1) {
      return ScalableVector(code, dtype.bits, -encoded_lanes);
    }
    return PrimType(code, dtype.bits, encoded_lanes);
  }

  /*! \brief Return the same type with a different scalar bit width, preserving code and lanes. */
  TVM_FFI_INLINE PrimType WithBits(int bits) const {
    DLDataType dtype = get()->dtype;
    int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
    if (encoded_lanes < -1) {
      return ScalableVector(this->code(), bits, -encoded_lanes);
    }
    return PrimType(this->code(), bits, encoded_lanes);
  }

  /*! \brief Return the same scalar element type with a fixed lane count. */
  TVM_FFI_INLINE PrimType WithLanes(int lanes) const {
    return PrimType(this->code(), this->bits(), lanes);
  }

  /*! \return The vscale factor encoded in a scalable vector type. */
  TVM_FFI_INLINE int32_t VScaleFactor() const {
    int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
    if (encoded_lanes >= -1) {
      TVM_FFI_THROW(InternalError) << "A fixed length vector doesn't have a vscale factor.";
    }
    return -encoded_lanes;
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimType, Type, PrimTypeObj);
};

inline bool operator==(const PrimType& lhs, const PrimType& rhs) {
  return lhs->dtype == rhs->dtype;
}

inline bool operator!=(const PrimType& lhs, const PrimType& rhs) { return !(lhs == rhs); }

namespace ffi {
template <>
inline constexpr bool use_default_type_traits_v<PrimType> = false;

template <>
struct TypeTraits<PrimType> : public ObjectRefWithFallbackTraitsBase<PrimType, DLDataType> {
  TVM_FFI_INLINE static PrimType ConvertFallbackValue(DLDataType dtype) { return PrimType(dtype); }
};

}  // namespace ffi

/*! \brief The type whose only value is None. */
class NoneTypeObj : public TypeObj {
 public:
  static void RegisterReflection() { ffi::reflection::ObjectDef<NoneTypeObj>(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.NoneType", NoneTypeObj, TypeObj);
};

/*! \brief Managed reference to NoneTypeObj. */
class NoneType : public Type {
 public:
  TVM_TIRX_DLL NoneType();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NoneType, Type, NoneTypeObj);
};

/*!
 * \brief The top type, which admits any value.
 *
 * AnyType types values that have no more specific static type, such as
 * DataType-valued generic constants and opaque runtime objects.
 */
class AnyTypeObj : public TypeObj {
 public:
  static void RegisterReflection() { ffi::reflection::ObjectDef<AnyTypeObj>(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.AnyType", AnyTypeObj, TypeObj);
};

/*!
 * \brief Managed reference to AnyTypeObj.
 * \sa AnyTypeObj
 */
class AnyType : public Type {
 public:
  TVM_TIRX_DLL AnyType(Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AnyType, Type, AnyTypeObj);
};

/*! \brief The type of a multi-dimensional tensor region expression. */
class TensorRegionTypeObj : public TypeObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorRegionTypeObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorRegionType", TensorRegionTypeObj, TypeObj);
};

/*! \brief Managed reference to TensorRegionTypeObj. */
class TensorRegionType : public Type {
 public:
  TVM_TIRX_DLL TensorRegionType();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorRegionType, Type, TensorRegionTypeObj);
};

/*! \brief Semantic string type; its physical representation is context dependent. */
class StringTypeObj : public TypeObj {
 public:
  static void RegisterReflection() { ffi::reflection::ObjectDef<StringTypeObj>(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.StringType", StringTypeObj, TypeObj);
};

/*! \brief Managed reference to StringTypeObj. */
class StringType : public Type {
 public:
  TVM_TIRX_DLL StringType();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StringType, Type, StringTypeObj);
};

/*!
 * \brief Low-level raw pointer type.
 *
 *  PtrType represents type hints in the TIR to be
 *  passed to the final code generator.
 *
 *  PtrType should not occur in the high-level analysis.
 *
 * \sa PtrType
 */
class PtrTypeObj : public TypeObj {
 public:
  /*!
   * \brief The type of the element which the pointer points to.
   */
  Type element_type = PrimType::Void();
  /*!
   * \brief The storage scope of the pointer
   */
  runtime::MemoryScope scope;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PtrTypeObj>()
        .def_ro("element_type", &PtrTypeObj::element_type)
        .def_ro("scope", &PtrTypeObj::scope, refl::DefaultValue(runtime::MemoryScope::Global()));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PtrType", PtrTypeObj, TypeObj);
};

/*
 * \brief Managed reference to PtrTypeObj.
 * \sa PtrTypeObj
 */
class PtrType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param element_type The type of the element which the pointer points to.
   * \param scope The storage scope into which the pointer addresses
   */
  TVM_TIRX_DLL explicit PtrType(Type element_type, runtime::MemoryScope scope = {});

  /*! \brief Construct an opaque pointer with void element type. */
  TVM_TIRX_DLL static PtrType VoidPtrType(runtime::MemoryScope scope = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PtrType, Type, PtrTypeObj);
};

/*!
 * \brief The type of tuple values.
 * \sa TupleType
 */
class TupleTypeObj : public TypeObj {
 public:
  /*! \brief The type of each field in the tuple. */
  ffi::Array<Type> fields;

  TupleTypeObj() {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleTypeObj>().def_ro("fields", &TupleTypeObj::fields);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TupleType", TupleTypeObj, TypeObj);
};

/*!
 * \brief Managed reference to TupleTypeObj.
 * \sa TupleTypeObj.
 */
class TupleType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param fields Fields in the tuple.
   * \param loc The loc of the type.
   */
  TVM_TIRX_DLL explicit TupleType(ffi::Array<Type> fields, Location loc = UnknownLoc());

  /*!
   * \brief Create an empty tuple type that constains nothing.
   * \return A empty tuple type.
   */
  TVM_TIRX_DLL TupleType static Empty();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleType, Type, TupleTypeObj);
};

/*!
 * \return a type that represents void.
 */
inline Type VoidType() { return TupleType::Empty(); }

/*!
 * \brief Check whether the tyep represents void.
 * \return The check result.
 */
inline bool IsVoidType(const Type& type) {
  auto* n = type.as<TupleTypeObj>();
  return n && n->fields.size() == 0;
}

/*!
 * \brief Function type.
 *
 * We support polymorphic function type.
 * This can be roughly viewed as template function in C++.
 *
 * \sa FuncType
 */
class FuncTypeObj : public TypeObj {
 public:
  /*! \brief type type of arguments */
  ffi::Array<Type> arg_types;
  /*! \brief The type of return value. */
  Type ret_type = VoidType();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FuncTypeObj>()
        .def_ro("arg_types", &FuncTypeObj::arg_types)
        .def_ro("ret_type", &FuncTypeObj::ret_type);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FuncType", FuncTypeObj, TypeObj);
};

/*!
 * \brief Managed reference to FuncTypeObj.
 * \sa FuncTypeObj
 */
class FuncType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param arg_types The types of the arguments.
   * \param ret_type The type of the return value.
   * \param loc The loc information.
   * \sa FuncTypeObj for more docs about these fields.
   */
  TVM_TIRX_DLL FuncType(ffi::Array<Type> arg_types, Type ret_type, Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FuncType, Type, FuncTypeObj);
};

}  // namespace tvm
#endif  // TVM_IR_TYPE_H_
