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
 * \file tvm/tirx/stmt_functor_with_constraint_context.h
 * \brief Native subclassing adapters over the structural constraint engines.
 */
#ifndef TVM_TIRX_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
#define TVM_TIRX_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/object_functor_with_constraint_context.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tirx {
/*!
 * \brief Native virtual node hooks using one constraint-aware structural visitor.
 * Allocate with ffi::make_object<Derived>(). Override Visit_ for a known node or
 * Visit for general dispatch. Defaults share matching native expression bodies
 * and delegate scope-sensitive nodes to structural constraint hooks. A hook owns
 * its descent; calling the qualified parent hook runs it once with the same context.
 * Unknown types use structural fallback, which can reach specialized descendants.
 */
class TVM_TIRX_DLL StmtExprVisitorWithConstraintContext
    : public ObjectVisitorWithConstraintContext {
 public:
  StmtExprVisitorWithConstraintContext() : ObjectVisitorWithConstraintContext(GlobalVTable()) {}

  /*!
   * \brief Default constraint-aware descent; override to customize LambdaExprObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const LambdaExprObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize OpaqueExprObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const OpaqueExprObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize TupleObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TupleObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize TupleGetItemObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TupleGetItemObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize TensorLoadObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorLoadObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize TensorRegionObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorRegionObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize VarObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const VarObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize GlobalVarObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const GlobalVarObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize CallObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize GenericConstObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const GenericConstObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize DataTypeImmObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const DataTypeImmObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize IntImmObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const IntImmObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize FloatImmObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const FloatImmObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize OpObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const OpObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize StringImmObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const StringImmObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::CastObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::CastObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::AddObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::AddObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::LShiftObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::LShiftObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::RShiftObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::RShiftObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BitwiseAndObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BitwiseAndObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BitwiseOrObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BitwiseOrObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BitwiseXorObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BitwiseXorObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BitwiseNotObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BitwiseNotObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::SubObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::SubObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::MulObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::MulObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::DivObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::DivObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::ModObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::ModObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::FloorDivObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::FloorDivObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::FloorModObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::FloorModObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::MinObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::MinObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::MaxObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::MaxObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::EQObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::EQObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::NEObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::NEObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::LTObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::LTObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::LEObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::LEObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::GTObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::GTObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::GEObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::GEObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::AndObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::AndObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::OrObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::OrObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::NotObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::NotObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::SelectObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::SelectObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::LetObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::LetObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::RampObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::RampObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BroadcastObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::BroadcastObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::ShuffleObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const prim::ShuffleObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize BindObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const BindObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize RegionStmtObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const RegionStmtObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize IfObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const IfObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize ForObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const ForObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize WhileObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const WhileObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize ReturnObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const ReturnObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize BreakObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const BreakObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize ContinueObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const ContinueObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize TensorStoreObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorStoreObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize AssertStmtObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const AssertStmtObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize SeqStmtObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const SeqStmtObj* node);
  /*!
   * \brief Default constraint-aware descent; override to customize EvaluateObj.
   */
  virtual ffi::Optional<ffi::VisitInterrupt> Visit_(const EvaluateObj* node);

 protected:
  /*!
   * \brief Construct with a finalized table extended by a native subclass.
   */
  explicit StmtExprVisitorWithConstraintContext(const VTable* table)
      : ObjectVisitorWithConstraintContext(table) {}

  static void InitVTable(VTable* table);

 private:
  static const VTable* GlobalVTable();
};

/*!
 * \brief Native virtual mutation hooks over one structural constraint engine.
 *
 * Override Mutate_ for known nodes or Mutate for general dispatch. Allocate with
 * ffi::make_object; typed entry points preserve expression/statement categories.
 * Defaults share matching native expression bodies and delegate scope-sensitive
 * nodes to the same hooks as StructuralMapWithConstraintContext and
 * StructuralMutateWithConstraintContext, with one context and one variable-remap
 * environment. Unknown node types use structural fallback.
 *
 * Forward the inherited mode to child mutations. Entry overrides must preserve
 * inherited denial and check uniqueness before writing. The default entry does
 * so before calling a node hook. Errors propagate without restoring constraints,
 * in-place IR changes, or identity remaps; discard the context after failure.
 * \code
 * struct Rewrite : StmtExprMutatorWithConstraintContext {
 *   using StmtExprMutatorWithConstraintContext::Mutate_;
 *   ffi::UnchangedOr<PrimExpr> Mutate_(const prim::AddObj* node,
 *                                    ffi::InplaceMode mode) override {
 *     // Inspect constraints here; delegate to retain default recursive behavior.
 *     return StmtExprMutatorWithConstraintContext::Mutate_(node, mode);
 *   }
 * };
 * auto mutator = ffi::make_object<Rewrite>();
 * auto result = mutator->Mutate(stmt);
 * \endcode
 */
class TVM_TIRX_DLL StmtExprMutatorWithConstraintContext
    : public ObjectMutatorWithConstraintContext {
 public:
  StmtExprMutatorWithConstraintContext() : ObjectMutatorWithConstraintContext(GlobalVTable()) {}
  /*! \brief Capture the supplied environment; the mutator owns its analyzer and local facts. */
  explicit StmtExprMutatorWithConstraintContext(const sym::ConstraintContext& context)
      : StmtExprMutatorWithConstraintContext() {
    constraints = context->Snapshot();
  }
  using ObjectMutatorWithConstraintContext::Mutate;
  /*!
   * \brief Typed entry preserving the expression or statement category.
   */
  ffi::UnchangedOr<Expr> Mutate(const Expr& value,
                                ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) {
    return Mutate(ffi::AnyView(value), mode).as_or_throw<ffi::UnchangedOr<Expr>>();
  }
  /*!
   * \brief Mutate a primitive expression, allowing a different primitive node kind.
   */
  ffi::UnchangedOr<PrimExpr> Mutate(const PrimExpr& value,
                                    ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) {
    return Mutate(ffi::AnyView(value), mode).as_or_throw<ffi::UnchangedOr<PrimExpr>>();
  }
  /*!
   * \brief Mutate a statement, allowing a different statement kind.
   */
  ffi::UnchangedOr<Stmt> Mutate(const Stmt& value,
                                ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) {
    return Mutate(ffi::AnyView(value), mode).as_or_throw<ffi::UnchangedOr<Stmt>>();
  }
  /*!
   * \brief Keep a sequence entry point canonical when its callback returns one statement.
   */
  ffi::UnchangedOr<SeqStmt> Mutate(const SeqStmt& value,
                                   ffi::InplaceMode mode = ffi::InplaceMode::kDisallow) {
    auto result = Mutate(ffi::AnyView(value), mode);
    if (result.IsUnchanged()) return ffi::Unchanged();
    return SeqStmt(std::move(result).ValueUnchecked().cast<Stmt>());
  }
  /*!
   * \brief Default constraint-aware descent; override to customize LambdaExprObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const LambdaExprObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize OpaqueExprObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const OpaqueExprObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize TupleObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const TupleObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize TupleGetItemObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const TupleGetItemObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize TensorLoadObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize TensorRegionObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const TensorRegionObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize VarObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const VarObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize GlobalVarObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const GlobalVarObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize CallObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const CallObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize GenericConstObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const GenericConstObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize DataTypeImmObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const DataTypeImmObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize IntImmObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const IntImmObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize FloatImmObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const FloatImmObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize OpObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const OpObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize StringImmObj.
   */
  virtual ffi::UnchangedOr<Expr> Mutate_(const StringImmObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::CastObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::CastObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::AddObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::AddObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::LShiftObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LShiftObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::RShiftObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::RShiftObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BitwiseAndObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndObj* node,
                                             ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BitwiseOrObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BitwiseXorObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorObj* node,
                                             ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BitwiseNotObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotObj* node,
                                             ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::SubObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::SubObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::MulObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::MulObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::DivObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::DivObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::ModObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::ModObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::FloorDivObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::FloorModObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::FloorModObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::MinObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::MinObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::MaxObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::MaxObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::EQObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::EQObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::NEObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::NEObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::LTObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LTObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::LEObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LEObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::GTObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::GTObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::GEObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::GEObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::AndObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::AndObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::OrObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::OrObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::NotObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::NotObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::SelectObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::SelectObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::LetObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LetObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::RampObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::RampObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::BroadcastObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize prim::ShuffleObj.
   */
  virtual ffi::UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize BindObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const BindObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize RegionStmtObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize IfObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const IfObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize ForObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const ForObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize WhileObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const WhileObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize ReturnObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const ReturnObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize BreakObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const BreakObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize ContinueObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const ContinueObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize TensorStoreObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize AssertStmtObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const AssertStmtObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize SeqStmtObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj* node, ffi::InplaceMode mode);
  /*!
   * \brief Default constraint-aware descent; override to customize EvaluateObj.
   */
  virtual ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj* node, ffi::InplaceMode mode);

 protected:
  /*!
   * \brief Synchronize before every analysis, including after child traversal.
   * The borrowed pointer is not a live view: reacquire it after changing context.
   */
  sym::AnalyzerObj* GetSyncedAnalyzer() {
    analyzer_->SyncConstraintContext(constraints);
    return analyzer_.get();
  }

  /*! \brief Add global nonnegative tensor-parameter shape premises before traversal. */
  void AddParamConstraints(const Function& func);

  /*! \brief Simplify indices using the active collector's iterator facts. */
  ffi::Array<PrimExpr> IterMapSimplifyWithContext(const ffi::Array<PrimExpr>& indices,
                                                  bool non_trivial_only);

  /*! \brief Construct with a finalized table extended by a native subclass. */
  explicit StmtExprMutatorWithConstraintContext(const VTable* table)
      : ObjectMutatorWithConstraintContext(table) {}

  static void InitVTable(VTable* table);

 private:
  static const VTable* GlobalVTable();
  sym::Analyzer analyzer_;
};
}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
