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
 * \file tvm/tirx/stmt_functor.h
 *
 * \brief Functors for tirx stmts
 *        utility functions to call common functors.
 */
#ifndef TVM_TIRX_STMT_FUNCTOR_H_
#define TVM_TIRX_STMT_FUNCTOR_H_

#include <tvm/ir/expr_functor.h>
#include <tvm/ir/object_functor.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace tirx {
/*!
 * \brief Type-dispatched statement functor with a caller-selected signature.
 *
 * Override Dispatch_ for a node type or DispatchDefault_ for default behavior.
 * This functor does not traverse children automatically. Dispatch may use a
 * registered ancestor. Derived extensions can initialize a fresh inherited
 * table with InitVTable and register additional hooks with SetDispatch.
 * \tparam FType The function signature.
 * \sa ExprFunctor
 */
template <typename FType>
class StmtFunctor;

template <typename R, typename... Args>
class StmtFunctor<R(const Stmt&, Args...)> {
 private:
  using TSelf = StmtFunctor<R(const Stmt&, Args...)>;

 public:
  /*! \brief The result type of this functor. */
  using result_type = R;
  /*! \brief Construct a functor with the TIRx statement hooks. */
  StmtFunctor() : StmtFunctor(GlobalVTable()) {}
  /*! \brief Destroy through the statement functor base. */
  virtual ~StmtFunctor() = default;
  /*! \brief Dispatch a statement, forwarding additional arguments to its hook. */
  TVM_FFI_INLINE R operator()(const Stmt& node, Args... args) {
    return Dispatch(node, std::forward<Args>(args)...);
  }
  /*! \brief Dispatch to a node hook, including registered ancestor hooks. */
  TVM_FFI_INLINE virtual R Dispatch(const Stmt& node, Args... args) {
    TVM_FFI_ICHECK(node.defined()) << "Cannot dispatch a null statement";
    return (*vtable_)(node, this, std::forward<Args>(args)...);
  }

  virtual R Dispatch_(const BindObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const RegionStmtObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const IfObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const ForObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const WhileObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const ReturnObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const BreakObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const ContinueObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const TensorStoreObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const AssertStmtObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const SeqStmtObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const EvaluateObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*! \brief Default behavior for statement hooks not overridden by a subclass. */
  virtual R DispatchDefault_(const ffi::Object* node, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << node->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

 protected:
  /*! \brief Dispatch table shared by this signature and its subclasses. */
  using VTable = ObjectFunctor<R(const ffi::ObjectRef&, TSelf*, Args...)>;
  /*! \brief Construct with a finalized table that outlives the functor. */
  explicit StmtFunctor(const VTable* vtable) : vtable_(vtable) {}
  /*! \brief Register statement hooks in a fresh mutable table. */
  static void InitVTable(VTable* vtable) {
    vtable->template SetDispatch<StmtObj>(
        [](const ffi::ObjectRef& node, TSelf* self, Args... args) -> R {
          return self->DispatchDefault_(node.get(), std::forward<Args>(args)...);
        });
    SetDispatch<TSelf, BindObj>(vtable);
    SetDispatch<TSelf, RegionStmtObj>(vtable);
    SetDispatch<TSelf, IfObj>(vtable);
    SetDispatch<TSelf, ForObj>(vtable);
    SetDispatch<TSelf, WhileObj>(vtable);
    SetDispatch<TSelf, ReturnObj>(vtable);
    SetDispatch<TSelf, BreakObj>(vtable);
    SetDispatch<TSelf, ContinueObj>(vtable);
    SetDispatch<TSelf, TensorStoreObj>(vtable);
    SetDispatch<TSelf, AssertStmtObj>(vtable);
    SetDispatch<TSelf, SeqStmtObj>(vtable);
    SetDispatch<TSelf, EvaluateObj>(vtable);
  }
  /*! \brief Register an additional node hook implemented by Self. */
  template <typename Self, typename Node>
  static void SetDispatch(VTable* vtable) {
    vtable->template SetDispatch<Node>(
        [](const ffi::ObjectRef& node, TSelf* self, Args... args) -> R {
          return static_cast<Self*>(self)->Dispatch_(static_cast<const Node*>(node.get()),
                                                     std::forward<Args>(args)...);
        });
  }

 private:
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
  const VTable* vtable_;
};

/*!
 * \brief Native visitor for TIRx statements and their expression operands.
 *
 * Inherits core expression dispatch and preserves TIRx traversal order and
 * tensor definition/use boundaries. Allocate visitors with ffi::make_object;
 * hooks return the first interrupt or throw on failure.
 * To preserve thrown ffi::Error subclasses, keep child traversal native:
 * Visit(array) crosses structural callbacks that may erase the C++ subtype.
 *
 * Native hooks match exact types. Unregistered OpaqueExprObj subclasses use
 * structural traversal; leaf types register non-descending structural hooks.
 */
class TVM_TIRX_DLL StmtExprVisitor : public tvm::ExprVisitor {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitor, tvm::ExprVisitor)

  using tvm::ExprVisitor::Visit;
  using tvm::ExprVisitor::Visit_;

  virtual ffi::Optional<VisitInterrupt> Visit_(const BindObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const IfObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const ForObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const WhileObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const ReturnObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const BreakObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const ContinueObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const AssertStmtObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const EvaluateObj* op);

  // Preserve TIRx operand traversal where it differs from the shared defaults.
  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const OpaqueExprObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TupleObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TupleGetItemObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::LetObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::RampObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleObj* op) override;

 protected:
  explicit StmtExprVisitor(const VTable* vtable) : tvm::ExprVisitor(vtable) {}
  static void InitVTable(VTable* vtable);
};

/*!
 * \brief Mutator that recursively mutates stmts and exprs on them.
 *
 * Base hooks preserve stored types and derived fields. They rewrite structural
 * children without re-inferring types or repeating constructor validation,
 * including on in-place writes. Passes that change dtypes or index lanes must
 * provide the corresponding TensorLoad, TensorStore, or TensorRegion inference.
 */
class TVM_TIRX_DLL StmtExprMutator : public tvm::ExprMutator {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprMutator, tvm::ExprMutator)
  using tvm::ExprMutator::Mutate;
  using tvm::ExprMutator::Mutate_;

  /*! \brief Mutate a borrowed statement through the virtual generic entry.
   * \param stmt The borrowed statement.
   * \param inplace_mode Inherited permission along the complete ownership chain.
   * \return Unchanged or an owning statement replacement.
   * \note Entry overrides must preserve the statement category. Result storage
   *       is transferred without a runtime category check, as for Expr/PrimExpr.
   */
  TVM_FFI_INLINE UnchangedOr<Stmt> Mutate(const Stmt& stmt,
                                          InplaceMode inplace_mode = InplaceMode::kDisallow) {
    return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<Stmt>(
        ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(stmt), inplace_mode)));
  }

  /*! \brief Mutate a scope body without collapsing its sequence container. */
  TVM_FFI_INLINE UnchangedOr<SeqStmt> Mutate(const SeqStmt& stmt,
                                             InplaceMode inplace_mode = InplaceMode::kDisallow) {
    auto result = Mutate(static_cast<const Stmt&>(stmt), inplace_mode);
    if (result.UnchangedOrSameAs(stmt)) return ffi::Unchanged();
    return SeqStmt(std::move(result).ValueUnchecked());
  }

  TVM_FFI_INLINE UnchangedOr<ffi::Optional<SeqStmt>> Mutate(
      const ffi::Optional<SeqStmt>& stmt, InplaceMode inplace_mode = InplaceMode::kDisallow) {
    if (!stmt.has_value()) return ffi::Unchanged();
    auto result = Mutate(stmt.value(), inplace_mode);
    if (result.IsUnchanged()) return ffi::Unchanged();
    return ffi::Optional<SeqStmt>(std::move(result).ValueUnchecked());
  }

  /*! \brief Mutate an optional statement, preserving an absent function body. */
  TVM_FFI_INLINE UnchangedOr<ffi::Optional<Stmt>> Mutate(
      const ffi::Optional<Stmt>& stmt, InplaceMode inplace_mode = InplaceMode::kDisallow) {
    if (!stmt.has_value()) return ffi::Unchanged();
    auto result = Mutate(stmt.value(), inplace_mode);
    if (result.IsUnchanged()) return ffi::Unchanged();
    return ffi::Optional<Stmt>(std::move(result).ValueUnchecked());
  }

  virtual UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const IfObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const WhileObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const ReturnObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const BreakObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const ContinueObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const AssertStmtObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const SeqStmtObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const EvaluateObj* op, InplaceMode inplace_mode);

 protected:
  explicit StmtExprMutator(const VTable* vtable) : tvm::ExprMutator(vtable) {}
  static void InitVTable(VTable* vtable);
};

/*!
 * \brief Substitute the var specified by vmap and legalize data types after substitution.
 * \param stmt The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 *
 * This statement overload legalizes only core TIRX nodes.  Dtype-changing
 * substitutions must be applied after lowering dialect blocks: structural
 * traversal of extension statements does not legalize their iterator domains or
 * bindings.  Type-preserving mappings continue to traverse schedulable blocks
 * structurally before lowering.
 *
 * \return The result.
 */
TVM_TIRX_DLL Stmt SubstituteWithDataTypeLegalization(
    Stmt stmt, ffi::TypedFunction<ffi::Optional<PrimExpr>(const Var&)> vmap);

/*!
 * \brief Substitute the var specified by vmap and legalize data types after substitution.
 * \param expr The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 *
 * Substitution may change the data type of the expression.
 *
 * \return The result.
 */
TVM_TIRX_DLL PrimExpr SubstituteWithDataTypeLegalization(
    PrimExpr expr, ffi::TypedFunction<ffi::Optional<PrimExpr>(const Var&)> vmap);

/*!
 * \brief Check if the statement contains the specified node type.
 *
 * This utility potentially walks the entire statement, and should
 * therefore not be used if it could otherwise be merged with another
 * pass.
 *
 * \param stmt The statement to be searched
 * \return Whether stmt contains Node
 */
template <typename Node, typename = std::enable_if_t<std::is_base_of_v<StmtObj, Node>>>
bool ContainsNode(const Stmt& stmt) {
  struct Visitor : StmtExprVisitor {
    // Early bail-out, if we already found the node. Skip expression operands.
    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
      if (contains_node || value.as<ExprObj>()) {
        return std::nullopt;
      }
      if (value.as<Node>()) {
        contains_node = true;
        return std::nullopt;
      }
      return StmtExprVisitor::Visit(value);
    }

    bool contains_node{false};
  };

  auto visitor = ffi::make_object<Visitor>();
  visitor->Visit(stmt);
  return visitor->contains_node;
}

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_STMT_FUNCTOR_H_
