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
 * \file tvm/ir/expr_functor.h
 * \brief Native visiting and mutation of core IR expressions.
 */
#ifndef TVM_IR_EXPR_FUNCTOR_H_
#define TVM_IR_EXPR_FUNCTOR_H_
#include <tvm/ir/expr.h>
#include <tvm/ir/object_functor.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/vector_expr.h>

namespace tvm {

/*!
 * \brief Type-dispatched expression functor with a caller-selected signature.
 * \tparam FType A function signature of the form R(const Expr&, Args...).
 *
 * Override Dispatch_ for a node type, or DispatchDefault_ for the default
 * behavior. Unlike structural visitors and mutators, this functor does not
 * traverse children automatically. Dispatch may use a registered ancestor.
 * A derived class can add node types with a fresh inherited table.
 * Use tvm::ExprFunctor explicitly when dialect functors are also in scope.
 */
template <typename FType>
class ExprFunctor;

template <typename R, typename... Args>
class ExprFunctor<R(const Expr&, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr&, Args...)>;

 public:
  /*! \brief The result type of this functor. */
  using result_type = R;
  /*! \brief Construct a functor with the core expression hooks. */
  ExprFunctor() : ExprFunctor(GlobalVTable()) {}
  /*!
   * \brief Dispatch an expression through Dispatch.
   * \param node The borrowed expression.
   * \param args Additional arguments forwarded to the selected hook.
   * \return The hook result.
   */
  TVM_FFI_INLINE R operator()(const Expr& node, Args... args) {
    return Dispatch(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch to a node hook, including registered ancestor hooks.
   * \param node The non-null borrowed expression.
   * \param args Additional arguments forwarded to the selected hook.
   * \return The hook result.
   */
  TVM_FFI_INLINE virtual R Dispatch(const Expr& node, Args... args) {
    TVM_FFI_ICHECK(node.defined()) << "Cannot dispatch a null expression";
    return (*vtable_)(node, this, std::forward<Args>(args)...);
  }

  virtual R Dispatch_(const LambdaExprObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const OpaqueExprObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const TupleObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const TupleGetItemObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const TensorRegionObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const TensorLoadObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const VarObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const GlobalVarObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const CallObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const GenericConstObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const DataTypeImmObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const IntImmObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const FloatImmObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const OpObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const StringImmObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::CastObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::AddObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::LShiftObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::RShiftObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::BitwiseAndObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::BitwiseOrObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::BitwiseXorObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::BitwiseNotObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::SubObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::MulObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::DivObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::ModObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::FloorDivObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::FloorModObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::MinObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::MaxObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::EQObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::NEObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::LTObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::LEObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::GTObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::GEObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::AndObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::OrObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::NotObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::SelectObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::LetObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::RampObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::BroadcastObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::ShuffleObj* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }

  /*!
   * \brief Default node behavior, overridden by subclasses that handle arbitrary core nodes.
   * \param node The borrowed node.
   * \param args Additional arguments supplied to the functor.
   * \return The hook result.
   */
  virtual R DispatchDefault_(const ffi::Object* node, Args... args) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << node->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

 protected:
  /*! \brief Dispatch table shared by this signature and its subclasses. */
  using VTable = ObjectFunctor<R(const ffi::ObjectRef&, TSelf*, Args...)>;
  /*!
   * \brief Construct a functor with an extended finalized table.
   * \param vtable The table, which must outlive the functor.
   */
  explicit ExprFunctor(const VTable* vtable) : vtable_(vtable) {}
  /*!
   * \brief Register core expression hooks in a fresh mutable table.
   * \param vtable The table to initialize before adding derived registrations.
   */
  static void InitVTable(VTable* vtable) {
    SetDispatch<TSelf, LambdaExprObj>(vtable);
    SetDispatch<TSelf, OpaqueExprObj>(vtable);
    SetDispatch<TSelf, TupleObj>(vtable);
    SetDispatch<TSelf, TupleGetItemObj>(vtable);
    SetDispatch<TSelf, TensorLoadObj>(vtable);
    SetDispatch<TSelf, TensorRegionObj>(vtable);
    SetDispatch<TSelf, VarObj>(vtable);
    SetDispatch<TSelf, GlobalVarObj>(vtable);
    SetDispatch<TSelf, CallObj>(vtable);
    SetDispatch<TSelf, GenericConstObj>(vtable);
    SetDispatch<TSelf, DataTypeImmObj>(vtable);
    SetDispatch<TSelf, IntImmObj>(vtable);
    SetDispatch<TSelf, FloatImmObj>(vtable);
    SetDispatch<TSelf, OpObj>(vtable);
    SetDispatch<TSelf, StringImmObj>(vtable);
    SetDispatch<TSelf, prim::CastObj>(vtable);
    SetDispatch<TSelf, prim::AddObj>(vtable);
    SetDispatch<TSelf, prim::LShiftObj>(vtable);
    SetDispatch<TSelf, prim::RShiftObj>(vtable);
    SetDispatch<TSelf, prim::BitwiseAndObj>(vtable);
    SetDispatch<TSelf, prim::BitwiseOrObj>(vtable);
    SetDispatch<TSelf, prim::BitwiseXorObj>(vtable);
    SetDispatch<TSelf, prim::BitwiseNotObj>(vtable);
    SetDispatch<TSelf, prim::SubObj>(vtable);
    SetDispatch<TSelf, prim::MulObj>(vtable);
    SetDispatch<TSelf, prim::DivObj>(vtable);
    SetDispatch<TSelf, prim::ModObj>(vtable);
    SetDispatch<TSelf, prim::FloorDivObj>(vtable);
    SetDispatch<TSelf, prim::FloorModObj>(vtable);
    SetDispatch<TSelf, prim::MinObj>(vtable);
    SetDispatch<TSelf, prim::MaxObj>(vtable);
    SetDispatch<TSelf, prim::EQObj>(vtable);
    SetDispatch<TSelf, prim::NEObj>(vtable);
    SetDispatch<TSelf, prim::LTObj>(vtable);
    SetDispatch<TSelf, prim::LEObj>(vtable);
    SetDispatch<TSelf, prim::GTObj>(vtable);
    SetDispatch<TSelf, prim::GEObj>(vtable);
    SetDispatch<TSelf, prim::AndObj>(vtable);
    SetDispatch<TSelf, prim::OrObj>(vtable);
    SetDispatch<TSelf, prim::NotObj>(vtable);
    SetDispatch<TSelf, prim::SelectObj>(vtable);
    SetDispatch<TSelf, prim::LetObj>(vtable);
    SetDispatch<TSelf, prim::RampObj>(vtable);
    SetDispatch<TSelf, prim::BroadcastObj>(vtable);
    SetDispatch<TSelf, prim::ShuffleObj>(vtable);
  }
  /*!
   * \brief Register a hook for an additional expression type.
   * \tparam Self The subclass implementing the hook.
   * \tparam Node The node type handled by the hook.
   * \param vtable The mutable table to receive the registration.
   */
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
 * \brief Structural Expr visitor with native overrides for core expressions.
 *
 * Prefer StructuralVisit or StructuralWalk for common cases; use ExprVisitor
 * for extensive per-kind customization or optimization. Default hooks follow
 * the registered structural child order and skip primitive metadata.
 * Native hooks match exact node types; derived node types use structural fallback
 * unless separately registered in a fresh inherited table with SetDispatch.
 * Existing hooks only require overriding. For an extra MyExprObj derived from
 * ExprObj, initialize an inherited table and register the new virtual hook:
 * \code
 * class MyExprVisitor : public ExprVisitor {
 *  public:
 *   TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(MyExprVisitor, ExprVisitor)
 *   using ExprVisitor::Visit_;
 *   virtual ffi::Optional<VisitInterrupt> Visit_(const MyExprObj* node);
 *
 *  protected:
 *   static void InitVTable(VTable* vtable) {
 *     ExprVisitor::InitVTable(vtable);
 *     SetDispatch<MyExprVisitor, MyExprObj>(vtable);
 *   }
 * };
 * \endcode
 * Use tvm::ExprVisitor explicitly when dialect visitors are also in scope.
 */
class TVM_TIRX_DLL ExprVisitor : public ObjectVisitor {
 public:
  /*! \brief Construct a visitor with the core expression hooks. */
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(ExprVisitor, ObjectVisitor)

  using ObjectVisitor::Visit;

  // Override existing hooks directly. Extra types need a fresh inherited table.
  // Hooks borrow the node and return None or an owning interrupt, and throw on failure.
  virtual ffi::Optional<VisitInterrupt> Visit_(const OpaqueExprObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const TupleObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const TupleGetItemObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const TensorRegionObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const VarObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const GlobalVarObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const CallObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const GenericConstObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const DataTypeImmObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const IntImmObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const FloatImmObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const OpObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const StringImmObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::CastObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::AddObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LShiftObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::RShiftObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseAndObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseOrObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseXorObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BitwiseNotObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::SubObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MulObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::DivObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::ModObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::FloorDivObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::FloorModObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MinObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MaxObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::EQObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::NEObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LTObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LEObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::GTObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::GEObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::AndObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::OrObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::NotObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::SelectObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LetObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::RampObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastObj* node);
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleObj* node);

 protected:
  /*!
   * \brief Construct a visitor with an extended native dispatch table.
   * \param vtable The finalized table, which must outlive this visitor.
   */
  explicit ExprVisitor(const VTable* vtable) : ObjectVisitor(vtable) {}
  /*!
   * \brief Register core expression hooks in a fresh mutable table.
   * \param vtable The table to initialize before adding derived registrations.
   */
  static void InitVTable(VTable* vtable);
};

/*!
 * \brief Structural Expr mutator with native overrides for core expressions
 * and structural fallback for other objects.
 *
 * Prefer StructuralMap for common cases; use ExprMutator for extensive per-kind
 * customization or optimization.
 * The default Var hook consults the inherited remap before visiting definition types.
 *
 * Native hooks match exact node types; derived node types need their own registrations.
 * Existing hooks only require overriding. For an extra MyExprObj derived from
 * ExprObj, initialize an inherited table and register the new virtual hook:
 * \code
 * class MyExprMutator : public ExprMutator {
 *  public:
 *   TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(MyExprMutator, ExprMutator)
 *   using ExprMutator::Mutate_;
 *   virtual UnchangedOr<Expr> Mutate_(const MyExprObj* node, InplaceMode
 * inplace_mode);
 *
 *  protected:
 *   static void InitVTable(VTable* vtable) {
 *     ExprMutator::InitVTable(vtable);
 *     SetDispatch<MyExprMutator, MyExprObj>(vtable);
 *   }
 * };
 * \endcode
 * Use tvm::ExprMutator explicitly when dialect mutators are also in scope.
 */
class TVM_TIRX_DLL ExprMutator : public ObjectMutator {
 public:
  /*! \brief Construct a mutator with the core expression hooks. */
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(ExprMutator, ObjectMutator)

  using ObjectMutator::Mutate;

  /*!
   * \brief Mutate a primitive expression, preserving its expression category.
   * \param expr The borrowed primitive expression.
   * \param inplace_mode Inherited mutation permission.
   * \return Unchanged or an owning primitive expression replacement.
   * \note Calls the virtual AnyView entry. Overrides must preserve PrimExpr;
   *       the result storage is transferred without a runtime type check.
   */
  TVM_FFI_INLINE UnchangedOr<PrimExpr> Mutate(const PrimExpr& expr,
                                              InplaceMode inplace_mode = InplaceMode::kDisallow) {
    return ffi::detail::UnchangedOrUnsafe::MoveFromTVMFFIAny<PrimExpr>(
        ffi::detail::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(expr), inplace_mode)));
  }

  /*!
   * \brief Mutate an expression, preserving its expression category.
   * \param expr The borrowed expression.
   * \param inplace_mode Inherited mutation permission.
   * \return Unchanged or an owning expression replacement.
   * \note Calls the virtual AnyView entry. Overrides must preserve Expr;
   *       the result storage is transferred without a runtime type check.
   */
  TVM_FFI_INLINE UnchangedOr<Expr> Mutate(const Expr& expr,
                                          InplaceMode inplace_mode = InplaceMode::kDisallow) {
    return ffi::detail::UnchangedOrUnsafe::MoveFromTVMFFIAny<Expr>(
        ffi::detail::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(expr), inplace_mode)));
  }

  // A downstream class overrides any existing hook without rebuilding the table.
  // Extra node types use a fresh inherited table and SetDispatch<Self, ExtraObj>.
  // Hooks borrow the node and return Unchanged or an owning replacement in its expression
  // category, throwing on failure. Narrower field types are checked separately. Forward
  // inplace_mode on every child edge.
  virtual UnchangedOr<Expr> Mutate_(const OpaqueExprObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const TupleObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const TupleGetItemObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const TensorRegionObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const VarObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const GlobalVarObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const CallObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const GenericConstObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const DataTypeImmObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const IntImmObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const FloatImmObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const OpObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<Expr> Mutate_(const StringImmObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::CastObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::AddObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::LShiftObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::RShiftObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::SubObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::MulObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::DivObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::ModObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::FloorModObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::MinObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::MaxObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::EQObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::NEObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::LTObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::LEObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::GTObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::GEObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::AndObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::OrObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::NotObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::SelectObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::LetObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::RampObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastObj* node, InplaceMode inplace_mode);
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleObj* node, InplaceMode inplace_mode);

 protected:
  /*!
   * \brief Reinfer the result type of a Call after ordinary mutation.
   * \param mutated The ordinary mutation result, or Unchanged for the original Call.
   * \param original The borrowed Call before ordinary mutation.
   * \param inplace_mode Permission to modify the original Call in place.
   * \return The combined mutation result, preserving Unchanged when possible.
   * \note A unique replacement Call may be updated in place even when the original
   *       could not be. Inference errors propagate to the caller.
   */
  static UnchangedOr<Expr> ReinferMutatedCallType(UnchangedOr<Expr> mutated,
                                                  const CallObj* original,
                                                  InplaceMode inplace_mode);

  /*!
   * \brief Construct a mutator with an extended native dispatch table.
   * \param vtable The finalized table, which must outlive this mutator.
   */
  explicit ExprMutator(const VTable* vtable) : ObjectMutator(vtable) {}
  /*!
   * \brief Register core expression hooks in a fresh mutable table.
   * \param vtable The table to initialize before adding derived registrations.
   */
  static void InitVTable(VTable* vtable);
};

}  // namespace tvm
#endif  // TVM_IR_EXPR_FUNCTOR_H_
