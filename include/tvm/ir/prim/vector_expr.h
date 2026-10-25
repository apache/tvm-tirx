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
 * \file tvm/ir/prim/vector_expr.h
 * \brief Primitive vector expressions.
 */
#ifndef TVM_IR_PRIM_VECTOR_EXPR_H_
#define TVM_IR_PRIM_VECTOR_EXPR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>

namespace tvm {
namespace prim {

/*!
 * \brief Construct a vector with lanes elements
 *        where its i-th element equals base + i * stride.
 *  This is useful to construct an index for a continuous vector load.
 *  Base and stride must be scalar signed int32 or int64. The stride is cast
 *  to the base type when their widths differ. Other element types can be
 *  expressed by explicitly casting the resulting vector.
 *
 *  Examples:
 *  - ramp(0, 1, 3) = [0, 1, 2]
 *  - ramp(1, 2, 4) = [1, 3, 5, 7]
 */
class RampObj : public ExprObj {
 public:
  RampObj(PrimExpr base, PrimExpr stride, PrimExpr lanes)
      : base(std::move(base)), stride(std::move(stride)), lanes(std::move(lanes)) {}
  explicit RampObj(ffi::UnsafeInit tag) : base(tag), stride(tag), lanes(tag) {}

  /*! \brief The base value. */
  PrimExpr base;
  /*! \brief The stride of each step. */
  PrimExpr stride;
  /*! \brief Total number of lanes. */
  PrimExpr lanes;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RampObj>()
        .def_ro("base", &RampObj::base)
        .def_ro("stride", &RampObj::stride)
        .def_ro("lanes", &RampObj::lanes);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Ramp", RampObj, ExprObj);
};

/*!
 * \brief Managed reference to RampObj
 * \sa RampObj
 */
class Ramp : public PrimExpr {
 public:
  TVM_TIRX_DLL Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Location loc = UnknownLoc());
  explicit Ramp(ffi::ObjectPtr<RampObj> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Ramp, PrimExpr, RampObj);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RampObj);
};

/*! \brief Create a vector where all the elements are value. */
class BroadcastObj : public ExprObj {
 public:
  BroadcastObj(PrimExpr value, PrimExpr lanes) : value(std::move(value)), lanes(std::move(lanes)) {}
  explicit BroadcastObj(ffi::UnsafeInit tag) : value(tag), lanes(tag) {}

  /*! \brief The base value. */
  PrimExpr value;
  /*! \brief The number of lanes. */
  PrimExpr lanes;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastObj>()
        .def_ro("value", &BroadcastObj::value)
        .def_ro("lanes", &BroadcastObj::lanes);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Broadcast", BroadcastObj, ExprObj);
};

/*!
 * \brief Managed reference to BroadcastObj
 * \sa BroadcastObj
 */
class Broadcast : public PrimExpr {
 public:
  TVM_TIRX_DLL Broadcast(PrimExpr value, PrimExpr lanes, Location loc = UnknownLoc());
  explicit Broadcast(ffi::ObjectPtr<BroadcastObj> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Broadcast, PrimExpr, BroadcastObj);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BroadcastObj);
};

/*!
 * \brief Shuffle instruction.
 *  vec = concat(vectors)
 *  result = (vec[indices[0]], vec[indices[1]] ...)
 */
class ShuffleObj : public ExprObj {
 public:
  /*! \brief the input vectors. */
  ffi::Array<PrimExpr> vectors;
  /*! \brief The indices of each element. */
  ffi::Array<PrimExpr> indices;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShuffleObj>()
        .def_ro("vectors", &ShuffleObj::vectors)
        .def_ro("indices", &ShuffleObj::indices);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Shuffle", ShuffleObj, ExprObj);
};

/*!
 * \brief Managed reference to ShuffleObj
 * \sa ShuffleObj
 */
class Shuffle : public PrimExpr {
 public:
  TVM_TIRX_DLL Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices,
                       Location loc = UnknownLoc());
  TVM_TIRX_DLL static PrimExpr Concat(ffi::Array<PrimExpr> vectors, Location loc = UnknownLoc());
  TVM_TIRX_DLL static PrimExpr ExtractElement(PrimExpr vector, int index,
                                              Location loc = UnknownLoc());
  explicit Shuffle(ffi::ObjectPtr<ShuffleObj> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Shuffle, PrimExpr, ShuffleObj);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShuffleObj);
};

}  // namespace prim
}  // namespace tvm
#endif  // TVM_IR_PRIM_VECTOR_EXPR_H_
