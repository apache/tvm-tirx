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
/*! \file tvm/ir/stmt.h
 * \brief Statements shared across IR dialects.
 */
#ifndef TVM_IR_STMT_H_
#define TVM_IR_STMT_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <utility>

namespace tvm {

/*! \brief Base node of all statements. */
class StmtObj : public ffi::Object {
 public:
  /*!
   * \brief Location that points to the original source code.
   *        Reserved debug information.
   */
  mutable Location loc = UnknownLoc();

  StmtObj() = default;
  explicit StmtObj(Location loc) : loc(loc) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StmtObj>().def_ro("loc", &StmtObj::loc, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;

  static constexpr const uint32_t _type_child_slots = 15;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Stmt", StmtObj, ffi::Object);
};

/*! \brief Container of all statements */
class Stmt : public ffi::ObjectRef {
 public:
  explicit Stmt(ffi::ObjectPtr<StmtObj> node) : ffi::ObjectRef(std::move(node)) {
    TVM_FFI_CHECK(defined(), ValueError) << "Stmt expects a defined node";
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Stmt, ffi::ObjectRef, StmtObj);
};

/*!
 * \brief The container of seq statement.
 *        Represent a sequence of statements.
 */
class SeqStmtObj : public StmtObj {
 public:
  /*! \brief internal sequence content. */
  ffi::Array<Stmt> seq;

  /*! \return get the size of the sequence */
  size_t size() const { return seq.size(); }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const { return seq[index]; }

  TVM_TIRX_DLL static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SeqStmt", SeqStmtObj, StmtObj);
};

/*! \brief Sequence statement. */
class SeqStmt : public Stmt {
 public:
  /*!
   * \brief Construct SeqStmt.
   * \param seq The sequence.
   * \param loc The location of this object in the source code.
   */
  TVM_TIRX_DLL explicit SeqStmt(ffi::Array<Stmt> seq, Location loc = UnknownLoc());
  /*! \brief Wrap a statement, reusing an existing sequence when possible. */
  TVM_TIRX_DLL SeqStmt(Stmt stmt, ffi::Optional<Location> loc = std::nullopt);
  SeqStmt(std::initializer_list<Stmt> seq, Location loc = UnknownLoc())
      : SeqStmt(ffi::Array<Stmt>(seq), std::move(loc)) {}

  /*! \return get the size of the sequence */
  size_t size() const { return operator->()->size(); }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const { return (*(operator->()))[index]; }
  explicit SeqStmt(ffi::ObjectPtr<SeqStmtObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SeqStmt, Stmt, SeqStmtObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqStmtObj);
};

namespace ffi {
template <>
inline constexpr bool use_default_type_traits_v<SeqStmt> = false;

template <>
struct TypeTraits<SeqStmt> : public ObjectRefWithFallbackTraitsBase<SeqStmt, Stmt, Array<Stmt>> {
  static SeqStmt ConvertFallbackValue(Stmt stmt) { return SeqStmt(std::move(stmt)); }
  static SeqStmt ConvertFallbackValue(Array<Stmt> seq) { return SeqStmt(std::move(seq)); }
};
}  // namespace ffi

/*!
 * \brief Bind a variable to a value in the enclosing scope.
 *
 * BindObj has no body field. The bound variable is visible
 * in all subsequent statements within the same enclosing scope (SeqStmt,
 * ForObj.body, etc.). This enables flat (non-nested) IR sequences.
 */
class BindObj : public StmtObj {
 public:
  explicit BindObj(ffi::UnsafeInit tag) : var(tag), value(tag) {}

  BindObj(Var var, Expr value) : var(std::move(var)), value(std::move(value)) {}

  /*! \brief The variable being bound. */
  Var var;
  /*! \brief The value to bind to the variable. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BindObj>()
        .def_ro("var", &BindObj::var, refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("value", &BindObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Bind", BindObj, StmtObj);
};

/*!
 * \brief Managed reference to BindObj.
 * \sa BindObj
 */
class Bind : public Stmt {
 public:
  TVM_TIRX_DLL Bind(Var var, Expr value, Location loc = UnknownLoc());

  explicit Bind(ffi::ObjectPtr<BindObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Bind, Stmt, BindObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BindObj);
};

/*!
 * \brief Check FRegionGetBodyParams registration without invoking it or creating variables.
 */
TVM_TIRX_DLL bool IsRegionOp(const Op& op);

/*!
 * \brief Validate region operands and construct fresh typed body parameters.
 * \param op The region operation.
 * \param args Operands evaluated in the enclosing scope.
 * \param attrs Attributes evaluated in the enclosing scope.
 * \return Parameters from the required FRegionGetBodyParams hook.
 * \throws ValueError if the operation has no registered region hook.
 */
TVM_TIRX_DLL ffi::Array<Var> GetRegionBodyParams(Op op, ffi::Array<Expr> args, DictAttrs attrs);

/*!
 * \brief A single-body statement whose semantics are defined by an operator.
 * Operands are evaluated in the enclosing scope. Body parameters are definitions
 * at body entry; result variables are definitions following the region.
 */
class RegionStmtObj : public StmtObj {
 public:
  /*! \brief Operator defining the region's semantics. */
  Op op;
  /*! \brief Operands evaluated in the enclosing scope. */
  ffi::Array<Expr> args;
  /*! \brief Variables defined at body entry, visible only within the body. */
  ffi::Array<Var> body_params;
  /*! \brief Attributes evaluated in the enclosing scope. */
  DictAttrs attrs;
  /*! \brief Body evaluated with the body parameters in scope. */
  SeqStmt body;
  /*! \brief Variables defined after the region in the enclosing sequence. */
  ffi::Array<Var> result_vars;

  explicit RegionStmtObj(ffi::UnsafeInit tag) : op(tag), body(tag) {}

  RegionStmtObj(Op op, SeqStmt body) : op(std::move(op)), body(std::move(body)) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RegionStmtObj>()
        .def_ro("op", &RegionStmtObj::op)
        .def_ro("attrs", &RegionStmtObj::attrs)
        .def_ro("args", &RegionStmtObj::args)
        .def_ro("body_params", &RegionStmtObj::body_params,
                refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("body", &RegionStmtObj::body)
        .def_ro("result_vars", &RegionStmtObj::result_vars,
                refl::AttachFieldFlag::SEqHashDefSimple());
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.RegionStmt", RegionStmtObj, StmtObj);
};

/*! \brief Managed reference to RegionStmtObj. */
class RegionStmt : public Stmt {
 public:
  TVM_TIRX_DLL RegionStmt(Op op, ffi::Array<Expr> args, ffi::Array<Var> body_params,
                          DictAttrs attrs, SeqStmt body, ffi::Array<Var> result_vars = {},
                          Location loc = UnknownLoc());

  explicit RegionStmt(ffi::ObjectPtr<RegionStmtObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RegionStmt, Stmt, RegionStmtObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RegionStmtObj);
};

namespace op_attr {
/*!
 * \brief Construct fresh typed variables for a region's lexical body parameters.
 *
 * The input carries only the operation, operands and attributes, without a body
 * or builder state. Parameters are ordered, distinct definitions and may have
 * name hints. Every region operation must register this hook, returning an empty
 * array when it has no body parameters. Presence of the attribute identifies
 * region support without invoking the hook or allocating variables.
 */
inline constexpr ffi::reflection::AttrKey<
    ffi::reflection::NativeFunctionView<ffi::Array<Var>(const CallObj* call)>>
    kRegionGetBodyParams{"FRegionGetBodyParams"};
/*! \brief Optional dialect validation of a complete region statement. */
inline constexpr ffi::reflection::AttrKey<
    ffi::reflection::NativeFunctionView<void(const RegionStmtObj*)>>
    kRegionValidate{"FRegionValidate"};
}  // namespace op_attr

// Compatibility names; new callers can deduce the value type from the key.
using FRegionGetBodyParams = decltype(op_attr::kRegionGetBodyParams)::ValueType;
using FRegionValidate = decltype(op_attr::kRegionValidate)::ValueType;

/*!
 * \brief Assert condition, if an error occurs, return the error message.
 *
 * The error is described by:
 * - \p error_kind: the error kind (e.g. "RuntimeError", "TypeError", "ValueError")
 * - \p message_parts: an array of string fragments that are concatenated at runtime
 *   via TVMFFIErrorSetRaisedFromCStrParts. This enables string fragment reuse
 *   across multiple assertions to reduce binary size.
 */
class AssertStmtObj : public StmtObj {
 public:
  explicit AssertStmtObj(ffi::UnsafeInit tag) : condition(tag), error_kind(tag) {}

  AssertStmtObj(PrimExpr condition, StringImm error_kind)
      : condition(std::move(condition)), error_kind(std::move(error_kind)) {}

  /*! \brief Condition to be checked. */
  PrimExpr condition;
  /*! \brief The error kind, e.g. "RuntimeError", "TypeError", "ValueError". */
  StringImm error_kind;
  /*! \brief Error message fragments, concatenated at runtime when assertion fails. */
  ffi::Array<StringImm> message_parts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AssertStmtObj>()
        .def_ro("condition", &AssertStmtObj::condition)
        .def_ro("error_kind", &AssertStmtObj::error_kind)
        .def_ro("message_parts", &AssertStmtObj::message_parts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.AssertStmt", AssertStmtObj, StmtObj);
};

/*!
 * \brief Managed reference to AssertStmtObj.
 * \sa AssertStmtObj
 */
class AssertStmt : public Stmt {
 public:
  TVM_TIRX_DLL AssertStmt(PrimExpr condition, StringImm error_kind,
                          ffi::Array<StringImm> message_parts, Location loc = UnknownLoc());

  explicit AssertStmt(ffi::ObjectPtr<AssertStmtObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertStmt, Stmt, AssertStmtObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AssertStmtObj);
};

/*!
 * \brief Store a primitive value to an indexed destination expression.
 *
 * \code
 *
 *  buffer[i, j] = value;
 *
 * \endcode
 */
class TensorStoreObj : public StmtObj {
 public:
  explicit TensorStoreObj(ffi::UnsafeInit tag) : dest(tag), value(tag) {}

  TensorStoreObj(Expr dest, PrimExpr value) : dest(std::move(dest)), value(std::move(value)) {}

  /*! \brief The indexed destination expression. */
  Expr dest;
  /*! \brief The indices location to be stored. */
  ffi::Array<PrimExpr> indices;
  /*! \brief The value to be stored. */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorStoreObj>()
        .def_ro("dest", &TensorStoreObj::dest)
        .def_ro("indices", &TensorStoreObj::indices)
        .def_ro("value", &TensorStoreObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorStore", TensorStoreObj, StmtObj);
};

/*!
 * \brief Managed reference to TensorStoreObj.
 * \sa TensorStoreObj
 */
class TensorStore : public Stmt {
 public:
  TVM_TIRX_DLL explicit TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value,
                                    Location loc = UnknownLoc());

  explicit TensorStore(ffi::ObjectPtr<TensorStoreObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorStore, Stmt, TensorStoreObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorStoreObj);
};

namespace type_attr {
/*! \brief Validate a tensor store by its destination type before constructing the statement. */
inline constexpr ffi::reflection::AttrKey<ffi::reflection::NativeFunctionView<void(
    const Expr&, const ffi::Array<PrimExpr>&, const PrimExpr&)>>
    kTensorStoreValidateByTy{"__tvm_tensor_store_validate__"};
}  // namespace type_attr

/*!
 * \brief Evaluates an expression.
 *  This is mostly used for putting a Call node into Stmt.
 *
 *  If value do not have side-effect, this node can be safely removed.
 */
class EvaluateObj : public StmtObj {
 public:
  explicit EvaluateObj(ffi::UnsafeInit tag) : value(tag) {}

  explicit EvaluateObj(Expr value) : value(std::move(value)) {}

  /*! \brief The expression to be evaluated. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EvaluateObj>().def_ro("value", &EvaluateObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Evaluate", EvaluateObj, StmtObj);
};

/*!
 * \brief Managed reference to EvaluateObj.
 * \sa EvaluateObj
 */
class Evaluate : public Stmt {
 public:
  TVM_TIRX_DLL explicit Evaluate(Expr value, Location loc = UnknownLoc());

  explicit Evaluate(int value, Location loc = UnknownLoc()) : Evaluate(PrimExpr(value), loc) {}

  explicit Evaluate(ffi::ObjectPtr<EvaluateObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Evaluate, Stmt, EvaluateObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EvaluateObj);
};

/*!
 * \brief If statement.
 */
class IfObj : public StmtObj {
 public:
  explicit IfObj(ffi::UnsafeInit tag) : condition(tag), then_case(tag) {}

  IfObj(PrimExpr condition, SeqStmt then_case)
      : condition(std::move(condition)), then_case(std::move(then_case)) {}

  /*! \brief The condition. */
  PrimExpr condition;
  /*! \brief The branch to be executed when condition is true. */
  SeqStmt then_case;
  /*! \brief The branch to be executed when condition is false, can be null. */
  ffi::Optional<SeqStmt> else_case;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfObj>()
        .def_ro("condition", &IfObj::condition)
        .def_ro("then_case", &IfObj::then_case)
        .def_ro("else_case", &IfObj::else_case);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.If", IfObj, StmtObj);
};

/*!
 * \brief Managed reference to IfObj.
 * \sa IfObj
 */
class If : public Stmt {
 public:
  TVM_TIRX_DLL If(PrimExpr condition, SeqStmt then_case,
                  ffi::Optional<SeqStmt> else_case = std::nullopt, Location loc = UnknownLoc());

  explicit If(ffi::ObjectPtr<IfObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(If, Stmt, IfObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IfObj);
};

/*!
 * \brief The kind of the loop.
 *
 *  ForKind can change the control flow semantics
 *  of the loop. So the kind field needs to be considered
 *  in all TIR passes.
 */
enum class ForKind : int {
  /*! \brief Ordinary loop with sequential iteration semantics. */
  kDefault = 0,
  /*! \brief Parallel execution. */
  kParallel = 1,
  /*!
   * \brief Vector SIMD loop.
   *  The loop body will be vectorized.
   */
  kVectorized = 2,
  /*! \brief The loop body must be unrolled. */
  kUnrolled = 3,
};

/*!
 * \brief A for loop, with possible type annotations.
 *
 * \code
 *
 *  for (var = begin; var < end; var += step) {
 *    // body
 *  }
 * \endcode
 */
class ForObj : public StmtObj {
 public:
  explicit ForObj(ffi::UnsafeInit tag) : var(tag), begin(tag), end(tag), step(tag), body(tag) {}

  ForObj(PrimVar var, PrimExpr begin, PrimExpr end, PrimExpr step, ForKind kind, SeqStmt body)
      : var(std::move(var)),
        begin(std::move(begin)),
        end(std::move(end)),
        step(std::move(step)),
        kind(kind),
        body(std::move(body)) {}

  /*! \brief The loop variable. */
  PrimVar var;
  /*! \brief The inclusive beginning of iteration. */
  PrimExpr begin;
  /*! \brief The exclusive end of iteration. */
  PrimExpr end;
  /*! \brief The required loop step, including an explicit typed one for unit stride. */
  PrimExpr step;
  /*! \brief The kind of the for loop. */
  ForKind kind;
  /*! \brief The body of the for loop. */
  SeqStmt body;
  /*!
   * \brief Additional annotations about the loop.
   *
   * Entries are interpreted by the consuming dialect. Transformations preserve
   * annotations unless they explicitly handle the corresponding semantics.
   */
  ffi::Map<ffi::String, ffi::Any> annotations;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ForObj>()
        .def_ro("var", &ForObj::var, refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("begin", &ForObj::begin)
        .def_ro("end", &ForObj::end)
        .def_ro("step", &ForObj::step)
        .def_ro("kind", &ForObj::kind)
        .def_ro("body", &ForObj::body)
        .def_ro("annotations", &ForObj::annotations);
  }

  /*! \brief Check it is a loop without nontrivial loop step. */
  TVM_TIRX_DLL bool HasTrivialStep() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.For", ForObj, StmtObj);
};

/*!
 * \brief Managed reference to ForObj.
 * \sa ForObj
 */
class For : public Stmt {
 public:
  TVM_TIRX_DLL For(PrimVar var, PrimExpr begin, PrimExpr end, PrimExpr step, ForKind kind,
                   SeqStmt body, ffi::Map<ffi::String, ffi::Any> annotations = {},
                   Location loc = UnknownLoc());

  explicit For(ffi::ObjectPtr<ForObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(For, Stmt, ForObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ForObj);
};

/*!
 * \brief A While loop
 *
 * \code
 *
 *  while (condition)
 *    body
 *
 * \endcode
 */
class WhileObj : public StmtObj {
 public:
  explicit WhileObj(ffi::UnsafeInit tag) : condition(tag), body(tag) {}

  WhileObj(PrimExpr condition, SeqStmt body)
      : condition(std::move(condition)), body(std::move(body)) {}

  /*! \brief The termination condition. */
  PrimExpr condition;
  /*! \brief The body of the while loop. */
  SeqStmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WhileObj>()
        .def_ro("condition", &WhileObj::condition)
        .def_ro("body", &WhileObj::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.While", WhileObj, StmtObj);
};

/*!
 * \brief Managed reference to WhileObj.
 * \sa WhileObj
 */
class While : public Stmt {
 public:
  TVM_TIRX_DLL While(PrimExpr condition, SeqStmt body, Location loc = UnknownLoc());

  explicit While(ffi::ObjectPtr<WhileObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(While, Stmt, WhileObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(WhileObj);
};

/*!
 * \brief A return from the current function.
 */
class ReturnObj : public StmtObj {
 public:
  explicit ReturnObj(ffi::UnsafeInit tag) : value(tag) {}

  explicit ReturnObj(Expr value) : value(std::move(value)) {}

  /*! \brief The value to return. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReturnObj>().def_ro("value", &ReturnObj::value);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Return", ReturnObj, StmtObj);
};

/*!
 * \brief Managed reference to ReturnObj.
 * \sa ReturnObj
 */
class Return : public Stmt {
 public:
  TVM_TIRX_DLL explicit Return(Expr value, Location loc = UnknownLoc());

  explicit Return(ffi::ObjectPtr<ReturnObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Return, Stmt, ReturnObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ReturnObj);
};

/*!
 * \brief A Break in control flow.
 */
class BreakObj : public StmtObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BreakObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Break", BreakObj, StmtObj);
};

/*!
 * \brief Managed reference to BreakObj.
 * \sa BreakObj
 */
class Break : public Stmt {
 public:
  TVM_TIRX_DLL explicit Break(Location loc);

  explicit Break(ffi::ObjectPtr<BreakObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Break, Stmt, BreakObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BreakObj);
};

/*!
 * \brief A Continue in control flow.
 */
class ContinueObj : public StmtObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ContinueObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Continue", ContinueObj, StmtObj);
};

/*!
 * \brief Managed reference to ContinueObj.
 * \sa ContinueObj
 */
class Continue : public Stmt {
 public:
  TVM_TIRX_DLL explicit Continue(Location loc);

  explicit Continue(ffi::ObjectPtr<ContinueObj> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Continue, Stmt, ContinueObj);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ContinueObj);
};

// overload printing of for type.
TVM_TIRX_DLL std::ostream& operator<<(std::ostream& os, ForKind kind);

// inline implementations
inline const char* ForKind2String(ForKind t) {
  switch (t) {
    case ForKind::kDefault:
      return "serial";
    case ForKind::kParallel:
      return "parallel";
    case ForKind::kVectorized:
      return "vectorized";
    case ForKind::kUnrolled:
      return "unroll";
  }
  TVM_FFI_THROW(InternalError) << "Unknown ForKind" << t;
  TVM_FFI_UNREACHABLE();
}

}  // namespace tvm
#endif  // TVM_IR_STMT_H_
