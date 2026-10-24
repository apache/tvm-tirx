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
 * \file tvm/ir/expr.h
 * \brief Base expr nodes in TVM.
 */
#ifndef TVM_IR_EXPR_H_
#define TVM_IR_EXPR_H_

#include <tvm/ffi/big_int.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/native_function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/location.h>
#include <tvm/ir/type.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace tvm {

/*!
 * \brief Base type of all the expressions.
 * \sa Expr
 */
class ExprObj : public ffi::Object {
 public:
  /*!
   * \brief Location that points to the original source code.
   *        Reserved debug information.
   */
  mutable Location loc = UnknownLoc();

  /*!
   * \brief The deduced or annotated type of the expression.
   *
   * MissingType() denotes type information that will be populated by
   * later analysis passes instead of expression constructors.
   */
  mutable Type ty = MissingType();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // loc does not participate in structural equal and hash.
    refl::ObjectDef<ExprObj>()
        .def_ro("loc", &ExprObj::loc, refl::DefaultValue(UnknownLoc()),
                refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("ty", &ExprObj::ty, refl::DefaultValue(MissingType()));
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  static constexpr const uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Expr", ExprObj, ffi::Object);
};

/*!
 * \brief Managed reference to ExprObj.
 * \sa ExprObj
 */
class Expr : public ffi::ObjectRef {
 public:
  // Expressions do not implicitly compare by object identity or address.  Callers must name
  // whether they intend object identity, structural equality, or primitive symbolic comparison.
  bool operator==(const Expr& other) const = delete;
  bool operator!=(const Expr& other) const = delete;
  bool operator<(const Expr& other) const = delete;

  explicit Expr(ffi::ObjectPtr<ExprObj> node) : ffi::ObjectRef(std::move(node)) {
    TVM_FFI_CHECK(defined(), ValueError) << "Expr expects a defined node";
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Expr, ffi::ObjectRef, ExprObj);
};

/*! \brief Base node for traversable expressions eliminated during compilation. */
class StagingExprObj : public ExprObj {
 public:
  static void RegisterReflection() { ffi::reflection::ObjectDef<StagingExprObj>(); }
  static constexpr const uint32_t _type_child_slots = 4;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.StagingExpr", StagingExprObj, ExprObj);
};

/*! \brief Managed reference to a staging expression. */
class StagingExpr : public Expr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StagingExpr, Expr, StagingExprObj);
};

/*!
 * \brief Base node for opaque construction-time expressions.
 *
 * Subclasses are passed through by generic expression visitors and mutators.
 * They must not remain in finished IR.
 */
class OpaqueExprObj : public ExprObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OpaqueExprObj>();
  }

  static constexpr const uint32_t _type_child_slots = 2;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.OpaqueExpr", OpaqueExprObj, ExprObj);
};

/*! \brief Managed reference to OpaqueExprObj. */
class OpaqueExpr : public Expr {
 public:
  explicit OpaqueExpr(ffi::ObjectPtr<OpaqueExprObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OpaqueExpr, Expr, OpaqueExprObj);
};

/*!
 * \brief Typed reference/view over an expression whose result type is a
 * specific Type subtype.
 * \tparam ExpectedType The expected expression result type.
 */
template <typename ExpectedType>
class TypedExpr : public Expr {
 public:
  /*! \return the typed result of this expression. */
  ExpectedType ty() const {
    const auto* node = get();
    TVM_FFI_DCHECK(node != nullptr);
    const auto* ty_node = node->ExprObj::ty.template as<typename ExpectedType::ContainerType>();
    TVM_FFI_DCHECK(ty_node != nullptr);
    return ffi::GetRef<ExpectedType>(ty_node);
  }

  explicit TypedExpr(ffi::ObjectPtr<ExprObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TypedExpr, Expr, ExprObj);
  static constexpr bool _type_container_is_exact = false;
};

/*!
 * \brief Typed reference/view over any Expr whose `ExprObj::ty` is PrimType.
 *
 * PrimExpr is a type category rather than a dedicated runtime node category.
 * It can contain intrinsic primitive nodes such as IntImmObj and FloatImmObj,
 * or a general ExprObj such as CallObj, when that expression's `ty` field is
 * a PrimType. This keeps primitive-only APIs explicit while allowing shared
 * Expr nodes for cross-dialect values with richer result types when needed.
 */
class PrimExpr : public TypedExpr<PrimType> {
 public:
  using TypedExpr<PrimType>::ty;

  /*!
   * \brief construct from integer.
   * \param value The value to be constructed.
   */
  TVM_TIRX_DLL PrimExpr(int32_t value);  // NOLINT(*)
  /*!
   * \brief construct from float.
   * \param value The value to be constructed.
   */
  TVM_TIRX_DLL PrimExpr(float value);  // NOLINT(*)

  explicit PrimExpr(ffi::ObjectPtr<ExprObj> node) : TypedExpr<PrimType>(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimExpr, TypedExpr<PrimType>, ExprObj);
  static constexpr bool _type_container_is_exact = false;
};

/*!
 * \brief View over a scalar signed or unsigned integer expression of any bit width.
 *
 * IntExpr validates integer operands in operator signatures without changing the
 * underlying expression node or its PrimExpr implementation API.
 */
class IntExpr : public PrimExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntExpr, PrimExpr, ExprObj);
  static constexpr bool _type_container_is_exact = false;
};

namespace ffi::detail {
// Array members require storage availability before the expression traits below are complete.
template <>
inline constexpr bool storage_enabled_v<Expr> = true;
template <>
inline constexpr bool storage_enabled_v<PrimExpr> = true;
}  // namespace ffi::detail

/*! \brief Base node for literal constants. */
class ConstantObj : public ExprObj {
 public:
  static constexpr uint32_t _type_child_slots = 5;
  static void RegisterReflection() { ffi::reflection::ObjectDef<ConstantObj>(); }
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Constant", ConstantObj, ExprObj);
};

/*! \brief Managed reference to a literal constant. */
class Constant : public Expr {
 public:
  explicit Constant(ffi::ObjectPtr<ConstantObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Constant, Expr, ConstantObj);
};

/*! \brief A constant whose payload is separate from its expression type. */
class GenericConstObj : public ConstantObj {
 public:
  ffi::Any value;

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<GenericConstObj>().def_ro("value", &GenericConstObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GenericConst", GenericConstObj, ConstantObj);
};

/*! \brief Managed reference to a generic constant. */
class GenericConst : public Constant {
 public:
  TVM_TIRX_DLL GenericConst(Type ty, ffi::Any value, Location loc = UnknownLoc());

  explicit GenericConst(ffi::ObjectPtr<GenericConstObj> node) : Constant(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GenericConst, Constant, GenericConstObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GenericConstObj);
};

/*! \brief A string literal with shared semantic StringType. */
class StringImmObj : public ConstantObj {
 public:
  ffi::String value;

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<StringImmObj>().def_ro("value", &StringImmObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.StringImm", StringImmObj, ConstantObj);
};

/*! \brief Managed reference to a string literal. */
class StringImm : public Constant {
 public:
  TVM_TIRX_DLL explicit StringImm(ffi::String value, Location loc = UnknownLoc());

  explicit StringImm(ffi::ObjectPtr<StringImmObj> node) : Constant(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StringImm, Constant, StringImmObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmObj);
};

/*! \brief A data type literal with AnyType as its expression type. */
class DataTypeImmObj : public ConstantObj {
 public:
  DLDataType value;

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<DataTypeImmObj>().def_ro("value", &DataTypeImmObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.DataTypeImm", DataTypeImmObj, ConstantObj);
};

/*! \brief Managed reference to a data type literal. */
class DataTypeImm : public Constant {
 public:
  TVM_TIRX_DLL explicit DataTypeImm(DLDataType value, Location loc = UnknownLoc());

  explicit DataTypeImm(ffi::ObjectPtr<DataTypeImmObj> node) : Constant(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DataTypeImm, Constant, DataTypeImmObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataTypeImmObj);
};

/*!
 * \brief Constant integer literals in the program.
 * \sa IntImm
 */
class IntImmObj : public ConstantObj {
 public:
  /*! \brief the Internal value. */
  ffi::BigInt value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntImmObj>().def_ro("value", &IntImmObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.IntImm", IntImmObj, ConstantObj);
};

/*!
 * \brief Managed reference class to IntImmObj.
 *
 * \sa IntImmObj
 */
class IntImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor.
   * \param value_ty The primitive type of the value.
   * \param value The internal value.
   * \param loc The location of this object in the source code.
   */
  TVM_TIRX_DLL IntImm(PrimType value_ty, ffi::BigInt value, Location loc = UnknownLoc());

  template <typename Enum, std::enable_if_t<std::is_enum_v<Enum>, int> = 0>
  IntImm(PrimType value_ty, Enum value, Location loc = UnknownLoc())
      : IntImm(std::move(value_ty), ffi::BigInt(static_cast<std::underlying_type_t<Enum>>(value)),
               std::move(loc)) {}

  /*!
   * \brief Construct a scalar boolean constant.
   * \param value The boolean value.
   * \param loc The location of this object in the source code.
   */
  static IntImm Bool(bool value, Location loc = UnknownLoc()) {
    return IntImm(PrimType::Bool(), value, loc);
  }

  /*!
   * \brief Construct a scalar int32 constant.
   * \param value The integer value.
   * \param loc The location of this object in the source code.
   */
  static IntImm Int32(ffi::BigInt value, Location loc = UnknownLoc()) {
    return IntImm(PrimType::Int(32), std::move(value), loc);
  }

  template <typename Enum, std::enable_if_t<std::is_enum_v<Enum>, int> = 0>
  static IntImm Int32(Enum value, Location loc = UnknownLoc()) {
    return IntImm(PrimType::Int(32), value, std::move(loc));
  }

  /*!
   * \brief Construct a scalar int64 constant.
   * \param value The integer value.
   * \param loc The location of this object in the source code.
   */
  static IntImm Int64(ffi::BigInt value, Location loc = UnknownLoc()) {
    return IntImm(PrimType::Int(64), std::move(value), loc);
  }

  template <typename Enum, std::enable_if_t<std::is_enum_v<Enum>, int> = 0>
  static IntImm Int64(Enum value, Location loc = UnknownLoc()) {
    return IntImm(PrimType::Int(64), value, std::move(loc));
  }

  explicit IntImm(ffi::ObjectPtr<IntImmObj> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntImm, PrimExpr, IntImmObj);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IntImmObj);
};

/*!
 * \brief Constant floating point literals in the program.
 * \sa FloatImm
 */
class FloatImmObj : public ConstantObj {
 public:
  /*! \brief The constant value content. */
  double value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FloatImmObj>().def_ro("value", &FloatImmObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FloatImm", FloatImmObj, ConstantObj);
};

/*!
 * \brief Managed reference class to FloatImmObj.
 *
 * \sa FloatImmObj
 */
class FloatImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor.
   * \param value_ty The primitive type of the value.
   * \param value The internal value.
   * \param loc The location in the source code.
   */
  TVM_TIRX_DLL FloatImm(PrimType value_ty, double value, Location loc = UnknownLoc());

  explicit FloatImm(ffi::ObjectPtr<FloatImmObj> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloatImm, PrimExpr, FloatImmObj);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloatImmObj);
};

/*! \brief Tuple container */
class TupleObj : public ExprObj {
 public:
  /*! \brief The fields of the tuple. */
  ffi::Array<Expr> fields;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleObj>().def_ro("fields", &TupleObj::fields);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Tuple", TupleObj, ExprObj);
};

/*! \brief Managed reference to TupleObj. */
class Tuple : public Expr {
 public:
  /*!
   * \brief Construct a tuple from its fields.
   * \param fields The fields of the tuple.
   * \param loc The source location of the expression.
   */
  TVM_TIRX_DLL explicit Tuple(ffi::Array<Expr> fields, Location loc = UnknownLoc());

  explicit Tuple(ffi::ObjectPtr<TupleObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Tuple, Expr, TupleObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleObj);
};

/*! \brief Get the index-th field out of a tuple. */
class TupleGetItemObj : public ExprObj {
 public:
  explicit TupleGetItemObj(Expr tuple) : tuple(std::move(tuple)) {}
  explicit TupleGetItemObj(ffi::UnsafeInit tag) : tuple(tag) {}

  /*! \brief The tuple expression. */
  Expr tuple;
  /*! \brief The field index. */
  int index;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleGetItemObj>()
        .def_ro("tuple_value", &TupleGetItemObj::tuple)
        .def_ro("index", &TupleGetItemObj::index);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TupleGetItem", TupleGetItemObj, ExprObj);
};

/*! \brief Managed reference to TupleGetItemObj. */
class TupleGetItem : public Expr {
 public:
  /*!
   * \brief Construct a tuple field projection.
   * \param tuple The tuple to get an element from.
   * \param index The field index.
   * \param loc The source location of the expression.
   */
  TVM_TIRX_DLL TupleGetItem(Expr tuple, int index, Location loc = UnknownLoc());

  explicit TupleGetItem(ffi::ObjectPtr<TupleGetItemObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleGetItem, Expr, TupleGetItemObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleGetItemObj);
};

/*! \brief Load a value from an indexed expression source. */
class TensorLoadObj : public ExprObj {
 public:
  explicit TensorLoadObj(Expr source) : source(std::move(source)) {}
  explicit TensorLoadObj(ffi::UnsafeInit tag) : source(tag) {}

  /*! \brief The indexed source expression. */
  Expr source;
  /*! \brief The indices at which the source is loaded. */
  ffi::Array<PrimExpr> indices;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorLoadObj>()
        .def_ro("source", &TensorLoadObj::source)
        .def_ro("indices", &TensorLoadObj::indices);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorLoad", TensorLoadObj, ExprObj);
};

namespace type_attr {
/*!
 * \brief Infer a load's primitive result type from its populated source, indices and location.
 * \note The callback must derive the result from those inputs, not the unpublished result type.
 */
inline constexpr ffi::reflection::AttrKey<
    ffi::reflection::NativeFunctionView<PrimType(const TensorLoadObj*)>>
    kTensorLoadInferType{"__tvm_tensor_load_infer_type__"};
}  // namespace type_attr

/*! \brief Managed reference to TensorLoadObj. */
class TensorLoad : public PrimExpr {
 public:
  /*! \brief Construct a load using the inference hook on the runtime class of source->ty. */
  TVM_TIRX_DLL TensorLoad(Expr source, ffi::Array<PrimExpr> indices, Location loc = UnknownLoc());

  /*! \brief Construct with an explicit primitive result type, bypassing source-type inference. */
  TVM_TIRX_DLL static TensorLoad Unchecked(PrimType ty, Expr source, ffi::Array<PrimExpr> indices,
                                           Location loc = UnknownLoc());

  /*! \brief Wrap an existing node, requiring a concrete primitive result type. */
  TVM_TIRX_DLL explicit TensorLoad(ffi::ObjectPtr<TensorLoadObj> node);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorLoad, PrimExpr, TensorLoadObj);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorLoadObj);
};

/*!
 * \brief A local variable in the IR.
 *
 * Variables are uniquely identified by their object identity.  The name is a
 * hint for printing and does not participate in structural equality or
 * hashing.
 */
class VarObj : public ExprObj {
 public:
  /*! \brief The variable name. */
  ffi::String name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VarObj>().def_ro("name", &VarObj::name, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  static constexpr const uint32_t _type_child_slots = 1;
  // VarObj reserves its sole child slot for the final relax::DataflowVarNode subtype, so its
  // descendant type-index range cannot overflow.
  static constexpr bool _type_child_slots_can_overflow = false;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Var", VarObj, ExprObj);
};

/*! \brief Managed reference to VarObj. */
class Var : public Expr {
 public:
  TVM_TIRX_DLL explicit Var(ffi::String name, Type ty, Location loc = UnknownLoc());

  /*! \brief Return a fresh ordinary Var with the same type and a new name. */
  TVM_TIRX_DLL Var CopyWithName(const ffi::String& name) const;

  /*! \brief Return a fresh ordinary Var with a suffix appended to its name. */
  TVM_TIRX_DLL Var CopyWithSuffix(const ffi::String& suffix) const;

  /*! \brief Return a fresh ordinary Var with a new primitive type. */
  TVM_TIRX_DLL Var CopyWithDType(PrimType dtype) const;

  /*! \brief Return a fresh ordinary Var with a new type, retaining its metadata. */
  TVM_TIRX_DLL Var CopyWithType(Type type) const;

  explicit Var(ffi::ObjectPtr<VarObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Var, Expr, VarObj);
};

/*!
 * \brief Checked scalar view over a VarObj.
 *
 * PrimVar is a zero-state reference view over the same VarObj as Var.  It additionally
 * guarantees that the inherited ExprObj::ty is PrimType.
 */
class PrimVar : public PrimExpr {
 public:
  /*! \brief Construct a scalar variable directly from a primitive type. */
  explicit PrimVar(ffi::String name, PrimType dtype = PrimType::Int(32),
                   Location loc = UnknownLoc())
      : PrimExpr(Var(std::move(name), std::move(dtype), std::move(loc)).as_or_throw<PrimExpr>()) {}

  /*! \brief Construct a scalar variable directly from a checked type annotation. */
  explicit PrimVar(ffi::String name, Type type_annotation, Location loc = UnknownLoc())
      : PrimExpr(Var(std::move(name), std::move(type_annotation), std::move(loc))
                     .as_or_throw<PrimExpr>()) {}

  /*! \brief Safe widening to a general Var view over the same node. */
  operator Var() const { return this->as_or_throw<Var>(); }

  PrimVar CopyWithSuffix(const ffi::String& suffix) const {
    return this->as_or_throw<Var>().CopyWithSuffix(suffix).as_or_throw<PrimVar>();
  }
  PrimVar CopyWithDType(PrimType dtype) const {
    return this->as_or_throw<Var>().CopyWithDType(dtype).as_or_throw<PrimVar>();
  }

  explicit PrimVar(ffi::ObjectPtr<VarObj> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimVar, PrimExpr, VarObj);
  static constexpr bool _type_container_is_exact = false;
};

/*!
 * \brief A typed staging expression representing a lambda computation.
 *
 * LambdaExpr records computations such as reduction combiners and predication
 * rules. Its body may describe computations on runtime values.
 *
 * Parameters are bound within the expression body, which may produce a scalar
 * or tuple result. The lambda has a FuncType describing its parameter and
 * return types.
 *
 * As a StagingExpr, LambdaExpr is eliminated during compilation and does not
 * remain in executable IR.
 */
class LambdaExprObj : public StagingExprObj {
 public:
  /*! \brief Lambda-local parameter definitions. */
  ffi::Array<Var> vars;
  /*! \brief Computation over the parameters and captured expressions. */
  Expr body;

  explicit LambdaExprObj(ffi::UnsafeInit tag) : body(tag) {}
  explicit LambdaExprObj(Expr body) : body(std::move(body)) {}
  /*! \brief Simultaneously substitute arguments for the bound parameters. */
  TVM_TIRX_DLL Expr Apply(const ffi::Array<Expr>& arguments) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LambdaExprObj>()
        .def_ro("vars", &LambdaExprObj::vars, refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("body", &LambdaExprObj::body);
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.LambdaExpr", LambdaExprObj, StagingExprObj);
};

/*! \brief Managed reference to a typed staging lambda. */
class LambdaExpr : public StagingExpr {
 public:
  TVM_TIRX_DLL explicit LambdaExpr(ffi::Array<Var> vars, Expr body, Location loc = UnknownLoc());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LambdaExpr, StagingExpr, LambdaExprObj);
};

class GlobalVar;
/*!
 * \brief Global variable that lives in the top-level module.
 *
 * A GlobalVar only refers to function definitions.
 * This is used to enable recursive calls between function.
 *
 * \sa GlobalVarObj
 */
class GlobalVarObj : public ExprObj {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  ffi::String name_hint;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GlobalVarObj>().def_ro("name_hint", &GlobalVarObj::name_hint);
    // A GlobalVar identifies a module-level symbol.  Its type is derived from the
    // corresponding function definition and is not part of the symbol identity.
    refl::TypeAttrDef<GlobalVarObj>()
        .def(tvm::ffi::type_attr::kSEqual.name, &GlobalVarObj::SEqual)
        .def(tvm::ffi::type_attr::kSHash.name, &GlobalVarObj::SHash);
  }

  bool SEqual(const GlobalVarObj* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
    return equal(name_hint, other->name_hint, false, "name_hint");
  }

  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
    return hash(name_hint, init_hash, false);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GlobalVar", GlobalVarObj, ExprObj);
};

/*!
 * \brief Managed reference to GlobalVarObj.
 * \sa GlobalVarObj
 */
class GlobalVar : public Expr {
 public:
  TVM_TIRX_DLL explicit GlobalVar(ffi::String name_hint, Location loc = UnknownLoc());

  explicit GlobalVar(ffi::ObjectPtr<GlobalVarObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GlobalVar, Expr, GlobalVarObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GlobalVarObj);
};

/*!
 * \brief The effect type of the call.
 */
enum class CallEffectKind : int {
  /*! \brief Function corresponds to an annotation(e.g. likely) and can translate to identity. */
  kExprAnnotation = 0,
  /*!
   * \brief Pure function that do not interacts
   *        with any external state.
   */
  kPure = 1,
  /*!
   * \brief Function's that may read from states(e.g. RAM)
   */
  kReadState = 2,
  /*!
   * \brief Function that may read/write from states(e.g. RAM).
   */
  kUpdateState = 3,
  /*!
   * \brief Opaque function, cannot make any assumption
   */
  kOpaque = kUpdateState,
  /*!
   * \brief Special intrinsic to annotate call arguments info
   *        only valid as a direct argument to a call.
   */
  kSpecialCallArg = 4,
  /*!
   * \brief Embed opaque information in the Expr, cannot be codegen.
   */
  kEmbedInfo = 5,
  /*!
   * \brief Function that changes control flow
   */
  kControlJump = 6,
};

inline std::ostream& operator<<(std::ostream& os, CallEffectKind side_effect) {
  switch (side_effect) {
    case CallEffectKind::kExprAnnotation:
      return os << "kExprAnnotation";

    case CallEffectKind::kPure:
      return os << "kPure";

    case CallEffectKind::kReadState:
      return os << "kReadState";

    case CallEffectKind::kUpdateState:
      return os << "kUpdateState";

    case CallEffectKind::kSpecialCallArg:
      return os << "kSpecialCallArg";

    case CallEffectKind::kEmbedInfo:
      return os << "kEmbedInfo";

    case CallEffectKind::kControlJump:
      return os << "kControlJump";

    default:
      TVM_FFI_THROW(InternalError) << "Unknown CallEffectKind: " << static_cast<int>(side_effect);
  }
}

/*!
 * \brief Call corresponds to callable invocation.
 */
class CallObj : public ExprObj {
 public:
  explicit CallObj(Expr op) : op(std::move(op)) {}
  explicit CallObj(ffi::UnsafeInit tag) : op(tag) {}

  /*!
   * \brief The operator/function being invoked.
   *
   * It can be an Op, a GlobalVar, a local function value, or another callable
   * expression.
   */
  Expr op;

  /*! \brief The arguments of the call. */
  ffi::Array<Expr> args;

  /*! \brief The additional attributes. */
  ffi::Optional<Attrs> attrs;

  /*! \brief The type information arguments passed to the callee. */
  ffi::Array<Type> ty_args;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallObj>()
        .def_ro("op", &CallObj::op)
        .def_ro("args", &CallObj::args)
        .def_ro("attrs", &CallObj::attrs)
        .def_ro("ty_args", &CallObj::ty_args);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Call", CallObj, ExprObj);
};

/*!
 * \brief Managed reference to CallObj.
 */
class Call : public Expr {
 public:
  /*! \brief Construct a Call, inferring only when ret_ty is absent.
   * Construction preserves provisional IR; invoke Validate separately when needed.
   */
  TVM_TIRX_DLL Call(ffi::Optional<Type> ret_ty, Expr op, ffi::Array<Expr> args,
                    ffi::Optional<Attrs> attrs = std::nullopt,
                    ffi::Array<Type> ty_args = ffi::Array<Type>(), Location loc = UnknownLoc());

  /*! \brief Check this Call against its registered operator contract. */
  TVM_TIRX_DLL void Validate() const;
  /*! \brief Recompute a result type from the Call's current explicit inputs.
   *
   * This ignores the Call's stored result type and does not mutate the Call.
   * Returns Missing when no deduction is available; inference errors propagate.
   * This does not invoke the operator's validator.
   */
  TVM_TIRX_DLL static Type ReinferType(const CallObj* call);

  explicit Call(ffi::ObjectPtr<CallObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Call, Expr, CallObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CallObj);
};

namespace op_attr {
/*! \brief The runtime effect of an operator call. */
inline constexpr ffi::reflection::AttrKey<CallEffectKind> kCallEffectKind{"TCallEffectKind"};
}  // namespace op_attr

/*! \brief range over one dimension */
class RangeObj : public ffi::Object {
 public:
  /*! \brief Inclusive beginning of the range. */
  PrimExpr begin;
  /*! \brief Extent of the range. */
  PrimExpr extent;
  /*! \brief the location of this range in the source */
  mutable Location loc = UnknownLoc();
  /*! \brief constructor */
  explicit RangeObj(ffi::UnsafeInit tag) : begin(tag), extent(tag) {}
  RangeObj(PrimExpr begin, PrimExpr extent, Location loc = UnknownLoc())
      : begin(begin), extent(extent), loc(loc) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RangeObj>()
        .def_ro("begin", &RangeObj::begin)
        .def_ro("extent", &RangeObj::extent)
        .def_ro("loc", &RangeObj::loc, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Range", RangeObj, ffi::Object);
};

/*! \brief Range container  */
class Range : public ffi::ObjectRef {
 public:
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   * \param loc The location of the Range in the source.
   */
  TVM_TIRX_DLL Range(PrimExpr begin, PrimExpr end, Location loc = UnknownLoc());
  /*!
   * \brief construct a new range with min and extent
   *  The corresponding constructor is removed,
   *  because that is counter convention of tradition meaning
   *  of range(begin, end)
   *
   * \param min The minimum range.
   * \param extent The extent of the range.
   * \param loc The location of the Range in the source.
   */
  TVM_TIRX_DLL static Range FromMinExtent(PrimExpr min, PrimExpr extent,
                                          Location loc = UnknownLoc());
  /*!
   * \brief Extract the interval width from its endpoints without an analyzer.
   *
   * For begin + extent or extent + begin, return the original extent operand.
   * Otherwise form end - begin using primitive arithmetic.
   */
  TVM_TIRX_DLL static PrimExpr GetExtentFromBeginEnd(const PrimExpr& begin, const PrimExpr& end);
  // declare range.
  explicit Range(ffi::UnsafeInit tag, ffi::ObjectPtr<RangeObj> n) : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Range, ffi::ObjectRef, RangeObj);
};

/*! \brief A region of an indexed expression source. */
class TensorRegionObj : public ExprObj {
 public:
  explicit TensorRegionObj(Expr source) : source(std::move(source)) {}
  explicit TensorRegionObj(ffi::UnsafeInit tag) : source(tag) {}

  /*! \brief The indexed source expression. */
  Expr source;
  /*! \brief The ranges selected from the source. */
  ffi::Array<Range> region;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // A region's source and symbolic source type form a definition pattern;
    // ranges then refer to those same identities.
    refl::ObjectDef<TensorRegionObj>()
        .def_ro("source", &TensorRegionObj::source, refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("region", &TensorRegionObj::region);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorRegion", TensorRegionObj, ExprObj);
};

/*! \brief Managed reference to TensorRegionObj. */
class TensorRegion : public Expr {
 public:
  TVM_TIRX_DLL TensorRegion(Expr source, ffi::Array<Range> region, Type ty,
                            Location loc = UnknownLoc());

  /*!
   * \brief Convert a tensor load to a region with unit extent at each index.
   * \param value The tensor load whose source, indices and location are retained.
   * \return The region, requiring the source to be a TIRx TensorVar.
   */
  TVM_TIRX_DLL static TensorRegion FromTensorLoad(TensorLoad value);

  explicit TensorRegion(ffi::ObjectPtr<TensorRegionObj> node) : Expr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorRegion, Expr, TensorRegionObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorRegionObj);
};

/*!
 * \brief Analyze the runtime effects of an expression.
 *
 * Accumulates effects of evaluated expression children, excluding type and vector
 * lane-count metadata. Non-operator callees are conservatively opaque. Missing
 * operator effect attributes are errors unless an earlier update effect stops
 * traversal. Returns kPure, kReadState, or kUpdateState.
 *
 * Shared IR contains only minimal analyses of expression properties. Arithmetic
 * reasoning, constraint solving, and target- or dialect-specific analyses belong
 * in their respective modules.
 * \param expr The expression to inspect.
 * \return The strongest runtime effect of the expression.
 */
TVM_TIRX_DLL CallEffectKind SideEffect(const Expr& expr);

namespace ffi {

template <>
inline constexpr bool use_default_type_traits_v<tvm::StringImm> = false;

template <>
struct TypeTraits<tvm::StringImm>
    : public ObjectRefWithFallbackTraitsBase<tvm::StringImm, ffi::String> {
  TVM_FFI_INLINE static tvm::StringImm ConvertFallbackValue(ffi::String value) {
    return tvm::StringImm(value);
  }
};

template <>
inline constexpr bool use_default_type_traits_v<IntImm> = false;

template <>
struct TypeTraits<IntImm> : public ObjectRefWithFallbackTraitsBase<IntImm, int64_t> {
  TVM_FFI_INLINE static IntImm ConvertFallbackValue(int64_t value) {
    auto value_ty =
        (value > std::numeric_limits<int>::max() || value < std::numeric_limits<int>::min())
            ? PrimType::Int(64)
            : PrimType::Int(32);
    return IntImm(value_ty, value);
  }
};

template <>
inline constexpr bool object_ref_contains_v<PrimExpr, IntImmObj> = true;

template <>
inline constexpr bool use_default_type_traits_v<FloatImm> = false;

template <>
struct TypeTraits<FloatImm> : public ObjectRefWithFallbackTraitsBase<FloatImm, double> {
  TVM_FFI_INLINE static FloatImm ConvertFallbackValue(double value) {
    return FloatImm(PrimType::Float(32), value);
  }
};

template <>
inline constexpr bool object_ref_contains_v<PrimExpr, FloatImmObj> = true;

template <>
inline constexpr bool object_ref_contains_v<PrimExpr, TensorLoadObj> = true;

template <typename ExpectedType>
inline constexpr bool use_default_type_traits_v<TypedExpr<ExpectedType>> = false;

template <typename ExpectedType>
struct TypeTraits<TypedExpr<ExpectedType>>
    : public ObjectRefTypeTraitsBase<TypedExpr<ExpectedType>> {
  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
        !detail::IsObjectInstance<ExprObj>(src->type_index)) {
      return false;
    }
    // Non-owning: this only reads `ty`, and the owning form's incref/decref pair costs two
    // atomics per check on a path every typed field assignment takes.
    const auto* expr = detail::ObjectUnsafe::RawObjectPtrFromUnowned<ExprObj>(src->v_obj);
    return detail::AnyUnsafe::CheckAnyViewStrict<ExpectedType>(expr->ty);
  }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    if (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin && src->v_obj != nullptr &&
        detail::IsObjectInstance<ExprObj>(src->type_index)) {
      const auto* expr = detail::ObjectUnsafe::RawObjectPtrFromUnowned<ExprObj>(src->v_obj);
      if (expr->ty.defined()) {
        const auto* prim_ty = expr->ty.as<PrimTypeObj>();
        std::string ty =
            prim_ty ? std::string(ffi::DLDataTypeToString(prim_ty->dtype)) : expr->ty.GetTypeKey();
        return TypeIndexToTypeKey(src->type_index) + "[ty=" + ty + "]";
      }
    }
    return TypeTraitsBase::GetMismatchTypeInfo(src);
  }
};

// Primitive expression arguments also accept bool, integer, and floating-point literals.
template <>
inline constexpr bool use_default_type_traits_v<PrimExpr> = false;

template <>
struct TypeTraits<PrimExpr>
    : public ObjectRefWithFallbackTraitsBase<PrimExpr, StrictBool, int64_t, double> {
  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<TypedExpr<PrimType>>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() { return "ir.PrimExpr"; }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraits<TypedExpr<PrimType>>::GetMismatchTypeInfo(src);
  }

  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(StrictBool value) {
    return IntImm::Bool(value);
  }
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(int64_t value) {
    return TypeTraits<IntImm>::ConvertFallbackValue(value);
  }
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(double value) {
    return TypeTraits<FloatImm>::ConvertFallbackValue(value);
  }
};

// Generic expression arguments accept primitive/string literals and arrays of expressions.
template <>
inline constexpr bool use_default_type_traits_v<Expr> = false;

template <>
struct TypeTraits<Expr> : public ObjectRefWithFallbackTraitsBase<Expr, PrimExpr, ffi::String> {
  TVM_FFI_INLINE static Expr ConvertFallbackValue(PrimExpr value) { return value; }
  TVM_FFI_INLINE static Expr ConvertFallbackValue(ffi::String value) {
    return StringImm(std::move(value));
  }
  static std::optional<Expr> TryCastFromAnyView(const TVMFFIAny* src) {
    if (auto value =
            ObjectRefWithFallbackTraitsBase<Expr, PrimExpr, ffi::String>::TryCastFromAnyView(src)) {
      return value;
    }
    if (auto fields = TypeTraits<ffi::Array<Expr>>::TryCastFromAnyView(src)) {
      return tvm::Tuple(std::move(*fields));
    }
    return std::nullopt;
  }
};

template <>
inline constexpr bool use_default_type_traits_v<IntExpr> = false;

template <>
struct TypeTraits<IntExpr> : public ObjectRefWithFallbackTraitsBase<IntExpr, int64_t> {
  using Base = ObjectRefWithFallbackTraitsBase<IntExpr, int64_t>;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (!Base::CheckAnyStrict(src)) return false;
    const auto* expr = detail::ObjectUnsafe::RawObjectPtrFromUnowned<ExprObj>(src->v_obj);
    const auto* ty = expr->ty.as<PrimTypeObj>();
    return ty != nullptr && ty->dtype.lanes == 1 &&
           (ty->dtype.code == kDLInt || ty->dtype.code == kDLUInt);
  }

  TVM_FFI_INLINE static IntExpr ConvertFallbackValue(int64_t value) {
    return TypeTraits<PrimExpr>::ConvertFallbackValue(value).as_or_throw<IntExpr>();
  }

  TVM_FFI_INLINE static std::string TypeStr() { return "ir.IntExpr"; }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraits<TypedExpr<PrimType>>::GetMismatchTypeInfo(src);
  }
};

template <>
inline constexpr bool use_default_type_traits_v<PrimVar> = false;

template <>
struct TypeTraits<PrimVar> : public ObjectRefTypeTraitsBase<PrimVar> {
  using Base = ObjectRefTypeTraitsBase<PrimVar>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeSchema;
  using Base::TypeStr;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != VarObj::RuntimeTypeIndex()) {
      return false;
    }
    const auto* var = static_cast<const VarObj*>(
        detail::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
    return detail::AnyUnsafe::CheckAnyStrict<PrimType>(var->ExprObj::ty);
  }

  TVM_FFI_INLINE static std::optional<PrimVar> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) {
      return detail::ObjectUnsafe::ObjectRefFromObjectPtr<PrimVar>(
          detail::ObjectUnsafe::ObjectPtrFromUnowned<VarObj>(src->v_obj));
    }
    return std::nullopt;
  }
};

template <>
inline constexpr bool use_default_type_traits_v<TensorRegion> = false;

template <>
struct TypeTraits<TensorRegion> : public ObjectRefWithFallbackTraitsBase<TensorRegion, TensorLoad> {
  TVM_FFI_INLINE static TensorRegion ConvertFallbackValue(TensorLoad value) {
    return TensorRegion::FromTensorLoad(std::move(value));
  }
};

}  // namespace ffi

/*! \brief Expression subscripts: point indices or (start, stop, step) slices. */
using SubscriptSlice = ffi::Array<ffi::Variant<
    ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>>,
    PrimExpr>>;

namespace type_attr {
/*!
 * \brief Realize a subscript of an expression, dispatched on the type of the expression.
 *
 * The callback receives the expression, its point indices or slices, and the source location.
 * It returns a defined object representing the realized subscription.
 */
inline constexpr ffi::reflection::AttrKey<
    ffi::TypedFunction<ffi::ObjectRef(Expr, SubscriptSlice, Location)>>
    kExprSubscriptRealizeByTy{"__tvm_expr_subscript_realize__"};
}  // namespace type_attr

}  // namespace tvm

/* \brief Allow tvm.Var and tvm.GlobalVar as keys in STL tables
 *
 * For most IR expressions, it would be ambiguous whether the
 * expression should follow reference equality or structural equality.
 * This is not the case for variables, which do not contain nested
 * internal structure, and are frequently used as keys in lookup
 * tables.
 *
 * Providing `std::hash` and `std::equal_to` specializations for
 * `tvm::Var` and `tvm::GlobalVar` allows them to be used as keys in STL tables.  For
 * other IR expressions, the user must specify the type of equality
 * used (e.g. `std::unordered_set<T, StructuralHash, StructuralEqual>`
 * or `std::unordered_set<T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>`).
 */
template <>
struct std::hash<tvm::Var> {
  std::size_t operator()(const tvm::Var& var) const { return tvm::ffi::ObjectPtrHash()(var); }
};

template <>
struct std::equal_to<tvm::Var> {
  bool operator()(const tvm::Var& var_a, const tvm::Var& var_b) const {
    return tvm::ffi::ObjectPtrEqual()(var_a, var_b);
  }
};

template <>
struct std::hash<tvm::GlobalVar> {
  std::size_t operator()(const tvm::GlobalVar& var) const { return tvm::ffi::ObjectPtrHash()(var); }
};

template <>
struct std::equal_to<tvm::GlobalVar> {
  bool operator()(const tvm::GlobalVar& var_a, const tvm::GlobalVar& var_b) const {
    return tvm::ffi::ObjectPtrEqual()(var_a, var_b);
  }
};
#endif  // TVM_IR_EXPR_H_
