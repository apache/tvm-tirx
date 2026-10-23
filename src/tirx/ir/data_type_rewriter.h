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
 * \file data_type_rewriter.h
 * \brief Rewrite the data type of expressions.
 */
#ifndef TVM_TIRX_IR_DATA_TYPE_REWRITER_H_
#define TVM_TIRX_IR_DATA_TYPE_REWRITER_H_

#include <tvm/tirx/stmt_functor.h>

#include <unordered_map>

namespace tvm {
namespace tirx {

/*!
 * \brief Legalize the data types of expressions to make sure they are consistent with other
 * parts of the program.
 *
 * It enforces the following rules:
 * - The data type of the index variable in a loop must be consistent with the data type of the loop
 *  bounds.
 * - The data type of the binary and ternary expressions must be consistent with the data types of
 * each of their operands.
 *
 * Usually we enforce the consistency of data types when constructing the IR nodes. However, such
 * inconsistency may happen as a result of IR mutation in some passes. This class can be used as
 * base class of such passes to ensure the consistency of data types.
 */
class DataTypeLegalizer : public StmtExprMutator {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(DataTypeLegalizer, StmtExprMutator)
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

 protected:
  explicit DataTypeLegalizer(const VTable* vtable) : StmtExprMutator(vtable) {}
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SelectObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::RampObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LShiftObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::RShiftObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::AddObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SubObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MulObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::DivObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::ModObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MinObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MaxObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::EQObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::NEObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LTObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LEObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GTObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GEObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetObj* op, InplaceMode inplace_mode) override;

  /*! \brief Whether to clamp shift amounts after narrowing signed integers. */
  virtual bool ShouldClampShiftAmounts() const { return false; }
};

/*!
 * \brief Data type rewriter for tensor indices.
 *
 * Detect the components of tensor indices that should be considered for data type rewriting.
 * This class doesn't perform actual rewriting of data types. During recursive visiting, the
 * internal flags `is_enabled_` and `is_conditional_` are used to indicate whether the current
 * expression is a tensor index or a conditional expression, which can be used in the sub-classes to
 * implement different rewriting rules.
 */
class IndexDataTypeRewriter : public DataTypeLegalizer {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(IndexDataTypeRewriter, DataTypeLegalizer)
  using DataTypeLegalizer::Mutate;
  using DataTypeLegalizer::Mutate_;

 protected:
  explicit IndexDataTypeRewriter(const VTable* vtable) : DataTypeLegalizer(vtable) {}
  using Parent = DataTypeLegalizer;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) override;
  ffi::Array<PrimExpr> VisitIndices(const ffi::Array<PrimExpr>& indices, InplaceMode inplace_mode);
  UnchangedOr<Stmt> Mutate_(const IfObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::EQObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::NEObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LTObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LEObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GTObj* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GEObj* op, InplaceMode inplace_mode) override;

  UnchangedOr<PrimExpr> Mutate_(const prim::SelectObj* op, InplaceMode inplace_mode) override;

  // indicator of index expr to rewrite
  bool is_enabled_{false};
  // indicator of condition
  bool is_condition_{false};
};

/*!
 * \brief Normalize the data types of tensor shapes and indices to the same data type.
 *
 * This pass rewrites the data types of tensor shapes and indices to the specified data type. It
 * assumes the specified data type is large enough to hold the original ranges of tensor shapes and
 * indices.
 */
class IndexDataTypeNormalizer : public IndexDataTypeRewriter {
 public:
  using IndexDataTypeRewriter::Mutate;
  using IndexDataTypeRewriter::Mutate_;
  explicit IndexDataTypeNormalizer(PrimType target_data_type);
  Function Rewrite(Function func);

 protected:
  IndexDataTypeNormalizer(PrimType target_data_type, const VTable* vtable);
  using Parent = IndexDataTypeRewriter;

  UnchangedOr<PrimExpr> Mutate_(const IntImmObj* op, InplaceMode inplace_mode) override;

  UnchangedOr<PrimExpr> Mutate_(const prim::CastObj* op, InplaceMode inplace_mode) override;

  /*! \brief Specifies which data type we can rewrite */
  virtual bool CanRewriteDType(PrimType dtype) const;

  PrimType target_data_type_ = PrimType::Int(64);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_IR_DATA_TYPE_REWRITER_H_
