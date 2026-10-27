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
 * \file vectorize_loop.cc
 * \brief Analyze original scalar loops before constructing vector IR.
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {
int GetLanesOrVScaleFactor(const PrimType& ty) {
  if (ty.IsScalableVector()) {
    return ty.VScaleFactor();
  }
  return ty.lanes();
}

// File-local helper: true if `expr` is a call to prim::vscale_op().
bool IsVScaleCall(const PrimExpr& expr) {
  if (const auto* call = expr.as<CallObj>()) {
    return call->op.same_as(prim::vscale_op());
  }
  return false;
}

bool TargetHasRVV(ffi::Optional<Target> target) {
  if (!target.has_value()) return false;
  static auto target_has_feature_fn = tvm::ffi::Function::GetGlobal("codegen.target_has_feature");
  return target_has_feature_fn.has_value() && (*target_has_feature_fn)("v", target).cast<bool>();
}

// File-local helper: true if the target supports Variable-Length Array extensions
// (AArch64 SVE or RISC-V V).
bool TargetHasVLA(ffi::Optional<Target> target) {
  if (!target.has_value()) return false;
  bool has_vla = target.value()->GetAttr<bool>("feature.has_sve").value_or(false);
  if (!has_vla) {
    if (auto mattr = target.value()->GetAttr<ffi::Array<ffi::String>>("mattr")) {
      for (const ffi::String& attr : mattr.value()) {
        if (attr == "+sve") {
          has_vla = true;
          break;
        }
      }
    }
  }
  has_vla |= TargetHasRVV(target);
  return has_vla;
}

}  // namespace

inline PrimExpr CreateNewLanes(bool is_scalable, int lanes_or_vscale_factor) {
  if (is_scalable) {
    return prim::Mul(Call(PrimType::Int(32), prim::vscale_op(), {}).as_or_throw<PrimExpr>(),
                     lanes_or_vscale_factor);
  } else {
    return lanes_or_vscale_factor;
  }
}

inline PrimExpr BroadcastTo(PrimExpr e, int lanes, bool is_scalable) {
  // Check if e is already in the expected form
  if (GetLanesOrVScaleFactor(e.ty()) == lanes && e.ty().IsScalableVector() == is_scalable) return e;

  if (const prim::BroadcastObj* op = e.as<prim::BroadcastObj>()) {
    TVM_FFI_ICHECK(op->ty.as_or_throw<PrimType>().IsScalableVector() == is_scalable)
        << "Can't broadcast between scalable and fixed length vectors.";
    int e_lanes = GetLanesOrVScaleFactor(op->ty.as_or_throw<PrimType>());

    if (lanes % e_lanes == 0) {
      return prim::Broadcast(op->value, CreateNewLanes(is_scalable, lanes));
    }
  }

  TVM_FFI_ICHECK(e.ty().IsScalar())
      << "Cannot broadcast lanes=" << GetLanesOrVScaleFactor(e.ty())
      << " is_scalable=" << e.ty().IsScalableVector() << " to " << lanes;

  return prim::Broadcast(e, CreateNewLanes(is_scalable, lanes));
}

// Admission is intentionally a small, closed set. Analyze the original whole loop:
// no speculative rewriting, no statement splitting, and no inferred call support.
class VectorizationAnalysis : private tvm::ExprFunctor<bool(const Expr&)>,
                              private StmtFunctor<bool(const Stmt&)> {
 private:
  enum class Variation { kUniform, kVarying };
  enum class AccessKind { kRead, kWrite };

  struct MemoryAccess {
    PrimExpr index;
    AccessKind kind;
  };

 public:
  // Construction can query approved expressions, but cannot change the plan.
  class Plan {
   public:
    bool IsUniform(const PrimExpr& expr) const {
      auto it = variation_.find(expr.get());
      TVM_FFI_ICHECK(it != variation_.end()) << "Expression missing from vectorization plan";
      return it->second == Variation::kUniform;
    }

   private:
    friend class VectorizationAnalysis;
    std::unordered_map<const ExprObj*, Variation> variation_;
  };

  VectorizationAnalysis(PrimVar var, const PrimExpr& extent, bool noalias,
                        const std::unordered_map<const VarObj*, const VarObj*>& roots,
                        bool allow_rvv_calls = false)
      : ExprFunctor(ExpressionVTable()),
        var_(var),
        noalias_(noalias),
        scalable_(!extent.as<IntImmObj>()),
        allow_rvv_calls_(allow_rvv_calls),
        roots_(roots) {
    var_variation_[var.get()] = Variation::kVarying;
  }

  std::optional<Plan> Run(const Stmt& body) {
    if (!StmtFunctor::Dispatch(body)) return std::nullopt;
    // Without tirx.noalias, distinct parameter storage may overlap, so admit
    // only one storage root. For a written root, every access must use the same
    // index expression so lanes cannot read or write each other's elements.
    if (!noalias_ && accesses_.size() > 1) return std::nullopt;
    for (const auto& [tensor, accesses] : accesses_) {
      bool written = std::any_of(accesses.begin(), accesses.end(), [](const MemoryAccess& access) {
        return access.kind == AccessKind::kWrite;
      });
      if (!written) continue;
      for (const MemoryAccess& access : accesses) {
        if (!ffi::StructuralEqual()(access.index, accesses.front().index)) return std::nullopt;
      }
    }
    return std::move(plan_);
  }

 private:
  static const ExprFunctor::VTable* ExpressionVTable() {
    static const ExprFunctor::VTable table = [] {
      ExprFunctor::VTable table;
      ExprFunctor::InitVTable(&table);
      // ExprFunctor, unlike StmtFunctor, throws on nodes with no registered hook.
      // Route those expressions to DispatchDefault_ so unknown nodes reject the loop.
      table.SetDispatch<ExprObj>([](const ffi::ObjectRef& node, ExprFunctor* self) {
        return self->DispatchDefault_(node.get());
      });
      table.Finalize();
      return table;
    }();
    return &table;
  }

  bool IsVectorValueType(PrimType type) const {
    if (!type.IsScalar()) return false;
    if (type == PrimType::Bool()) return true;
    if (type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
      return type.bits() == 8 || type.bits() == 16 || type.bits() == 32 || type.bits() == 64;
    }
    return type.MatchesCode(DLDataTypeCode::kDLFloat) &&
           (type.bits() == 16 || type.bits() == 32 || type.bits() == 64);
  }

  bool IsVectorStorageType(PrimType type) const {
    // Mask support alone does not establish a vector storage representation for bool.
    return type != PrimType::Bool() && IsVectorValueType(type);
  }

  bool AnalyzeExpr(const PrimExpr& expr) {
    return IsVectorValueType(expr.ty()) && ExprFunctor::Dispatch(expr);
  }

  Variation VariationOf(const PrimExpr& expr) const { return plan_.variation_.at(expr.get()); }

  Variation Join(std::initializer_list<PrimExpr> operands) const {
    for (const PrimExpr& operand : operands) {
      if (VariationOf(operand) == Variation::kVarying) return Variation::kVarying;
    }
    return Variation::kUniform;
  }

  bool Record(const ExprObj* expr, Variation variation) {
    plan_.variation_[expr] = variation;
    return true;
  }

  // The functors only dispatch: they never inherit recursive traversal that
  // could admit unsupported nodes. AssertStmt, calls, existing vector forms,
  // control flow and all other unhandled nodes retain the original scalar loop.
  bool DispatchDefault_(const ffi::Object* op) final { return false; }

  bool Dispatch_(const IntImmObj* op) final { return Record(op, Variation::kUniform); }
  bool Dispatch_(const FloatImmObj* op) final { return Record(op, Variation::kUniform); }

  bool Dispatch_(const VarObj* op) final {
    auto it = var_variation_.find(op);
    return Record(op, it == var_variation_.end() ? Variation::kUniform : it->second);
  }

  bool Dispatch_(const CallObj* op) final {
    // These unary operations are lane-wise and pure. Keep arbitrary calls and
    // scalable call lowering outside the admitted subset.
    if (scalable_ || op->args.size() != 1) return false;
    auto arg = op->args[0].as<PrimExpr>();
    auto result_type = op->ty.as<PrimType>();
    if (!arg || !result_type || !AnalyzeExpr(*arg)) return false;
    PrimType input_type = arg->ty();
    bool supported = false;
    if (op->op.same_as(prim::popcount_op())) {
      supported = input_type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
                  input_type == *result_type;
    } else if (allow_rvv_calls_ && op->op.same_as(prim::floor_op())) {
      supported = input_type.MatchesCode(DLDataTypeCode::kDLFloat) && input_type == *result_type;
    } else if (allow_rvv_calls_ && op->op.same_as(tirx::reinterpret_op())) {
      supported = input_type.bits() >= 8 && input_type.bits() == result_type->bits();
    }
    return supported && Record(op, VariationOf(*arg));
  }

  bool Dispatch_(const TensorLoadObj* op) final {
    if (!AnalyzeAccess(op->source, op->indices, AccessKind::kRead)) return false;
    return Record(op, VariationOf(op->indices[0]));
  }

  bool Dispatch_(const prim::CastObj* op) final {
    // Scalar conversions between the supported numeric types (including bool
    // masks) widen to the same fixed or scalable shape.
    if (!AnalyzeExpr(op->value)) return false;
    return Record(op, VariationOf(op->value));
  }

  bool Dispatch_(const prim::SelectObj* op) final {
    if (!AnalyzeExpr(op->condition) || !AnalyzeExpr(op->true_value) ||
        !AnalyzeExpr(op->false_value))
      return false;
    return Record(op, Join({op->condition, op->true_value, op->false_value}));
  }

  bool Dispatch_(const prim::NotObj* op) final {
    if (!AnalyzeExpr(op->a)) return false;
    return Record(op, VariationOf(op->a));
  }

  bool Dispatch_(const prim::BitwiseNotObj* op) final {
    if (!AnalyzeExpr(op->a)) return false;
    return Record(op, VariationOf(op->a));
  }

  template <typename T>
  bool AnalyzeBinary(const T* op) {
    if (!AnalyzeExpr(op->a) || !AnalyzeExpr(op->b)) return false;
    return Record(op, Join({op->a, op->b}));
  }

#define TVM_VECTOR_BINARY_DISPATCH(Node) \
  bool Dispatch_(const prim::Node##Obj* op) final { return AnalyzeBinary(op); }
  TVM_VECTOR_BINARY_DISPATCH(Add)
  TVM_VECTOR_BINARY_DISPATCH(Sub)
  TVM_VECTOR_BINARY_DISPATCH(Mul)
  TVM_VECTOR_BINARY_DISPATCH(Div)
  TVM_VECTOR_BINARY_DISPATCH(Mod)
  TVM_VECTOR_BINARY_DISPATCH(FloorDiv)
  TVM_VECTOR_BINARY_DISPATCH(FloorMod)
  TVM_VECTOR_BINARY_DISPATCH(Min)
  TVM_VECTOR_BINARY_DISPATCH(Max)
  TVM_VECTOR_BINARY_DISPATCH(EQ)
  TVM_VECTOR_BINARY_DISPATCH(NE)
  TVM_VECTOR_BINARY_DISPATCH(LT)
  TVM_VECTOR_BINARY_DISPATCH(LE)
  TVM_VECTOR_BINARY_DISPATCH(GT)
  TVM_VECTOR_BINARY_DISPATCH(GE)
  TVM_VECTOR_BINARY_DISPATCH(And)
  TVM_VECTOR_BINARY_DISPATCH(Or)
  TVM_VECTOR_BINARY_DISPATCH(LShift)
  TVM_VECTOR_BINARY_DISPATCH(RShift)
  TVM_VECTOR_BINARY_DISPATCH(BitwiseAnd)
  TVM_VECTOR_BINARY_DISPATCH(BitwiseOr)
  TVM_VECTOR_BINARY_DISPATCH(BitwiseXor)
#undef TVM_VECTOR_BINARY_DISPATCH

  bool AnalyzeAccess(const Expr& tensor_expr, const ffi::Array<PrimExpr>& indices,
                     AccessKind kind) {
    auto tensor = tensor_expr.as<TensorVar>();
    if (!tensor || !roots_.count(tensor->get()) || tensor->ty()->layout.has_value() ||
        !IsVectorStorageType(tensor->dtype()) || indices.size() != 1 || !AnalyzeExpr(indices[0]))
      return false;
    bool varying = VariationOf(indices[0]) == Variation::kVarying;
    if (kind == AccessKind::kWrite || (scalable_ && varying)) {
      auto coefficients = DetectLinearEquation(indices[0], {var_});
      if (coefficients.size() != 2) return false;
      const auto* stride = coefficients[0].as<IntImmObj>();
      if (!stride) return false;
      // Writes must hit a distinct element per lane: stride * var + base, stride != 0.
      if (kind == AccessKind::kWrite && stride->value == 0) return false;
      // This vector construction path requires contiguous scalable accesses.
      if (scalable_ && stride->value != 1) return false;
    }
    accesses_[roots_.at(tensor->get())].push_back({indices[0], kind});
    return true;
  }

  bool Dispatch_(const SeqStmtObj* op) final {
    for (const Stmt& child : op->seq) {
      if (!StmtFunctor::Dispatch(child)) return false;
    }
    return true;
  }

  bool Dispatch_(const BindObj* op) final {
    auto value = op->value.as<PrimExpr>();
    if (!value || !AnalyzeExpr(*value)) return false;
    var_variation_[op->var.get()] = VariationOf(*value);
    return true;
  }

  bool Dispatch_(const TensorStoreObj* op) final {
    return AnalyzeExpr(op->value) && AnalyzeAccess(op->dest, op->indices, AccessKind::kWrite);
  }

  bool Dispatch_(const EvaluateObj* op) final {
    auto value = op->value.as<PrimExpr>();
    return value && AnalyzeExpr(*value);
  }

  PrimVar var_;
  bool noalias_;
  bool scalable_;
  bool allow_rvv_calls_;
  const std::unordered_map<const VarObj*, const VarObj*>& roots_;
  Plan plan_;
  std::unordered_map<const VarObj*, Variation> var_variation_;
  std::unordered_map<const VarObj*, std::vector<MemoryAccess>> accesses_;
};

// Construction consumes a completed plan. There is no scalarization path here.
class Vectorizer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  Vectorizer(PrimVar var, PrimExpr lanes, const VectorizationAnalysis::Plan& plan) : plan_(plan) {
    VarRemapSet(var, prim::Ramp(IntImm(var.ty(), 0), IntImm(var.ty(), 1), lanes));
  }

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (auto expr = value.as<PrimExpr>()) {
      if (plan_.IsUniform(*expr)) return ffi::Unchanged();
    }
    return StmtExprMutator::Mutate(value, InplaceMode::kDisallow);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::AddObj* op, InplaceMode inplace_mode) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a + b; }, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::SubObj* op, InplaceMode inplace_mode) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a - b; }, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::MulObj* op, InplaceMode inplace_mode) final {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::Unchanged();
    } else {
      bool is_vec_a = a.ty().IsScalableVector() || a.ty().IsFixedLengthVector();
      bool is_vec_b = b.ty().IsScalableVector() || b.ty().IsFixedLengthVector();
      if (is_vec_a && is_vec_b) {
        // Let's not multiply scalable and fixed length vectors
        TVM_FFI_ICHECK(a.ty().IsScalableVector() == b.ty().IsScalableVector())
            << "Fixed length and scalable vectors can't be mixed in multiplication.";
      }
      if (is_vec_a || is_vec_b) {
        const prim::RampObj* b_ramp = b.as<prim::RampObj>();
        const prim::RampObj* a_ramp = a.as<prim::RampObj>();
        if (a_ramp && b.ty().IsScalar() && analyzer_->CanProve(b > 0)) {
          PrimExpr lanes = a_ramp->lanes;
          return prim::Ramp(a_ramp->base * b, a_ramp->stride * b, lanes);
        }
        if (b_ramp && a.ty().IsScalar() && analyzer_->CanProve(a > 0)) {
          PrimExpr lanes = b_ramp->lanes;
          return prim::Ramp(b_ramp->base * a, b_ramp->stride * a, lanes);
        }
        int a_lanes = GetLanesOrVScaleFactor(a.ty());
        int b_lanes = GetLanesOrVScaleFactor(b.ty());
        int max_lanes = std::max(a_lanes, b_lanes);
        bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
        return prim::Mul(BroadcastTo(a, max_lanes, is_scalable),
                         BroadcastTo(b, max_lanes, is_scalable));
      }
    }
    return BinaryVec<prim::Mul>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::DivObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Div>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::ModObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Mod>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::FloorDiv>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::FloorMod>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::MinObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Min>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::MaxObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Max>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::EQObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::EQ>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::NEObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::NE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LTObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::LT>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LEObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::LE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::GTObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::GT>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::GEObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::GE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::AndObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::And>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::OrObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Or>(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::LShiftObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::LShift>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::RShiftObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::RShift>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseAnd>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseOr>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorObj* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseXor>(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotObj* op, InplaceMode inplace_mode) final {
    auto a = this->Mutate(op->a, inplace_mode);
    if (a.UnchangedOrSameAs(op->a)) return ffi::Unchanged();
    return prim::BitwiseNot(std::move(a).ValueOrUnchanged(op->a), op->loc);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::NotObj* op, InplaceMode inplace_mode) final {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    if (a_unchanged) {
      return ffi::Unchanged();
    } else {
      return !(a);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::SelectObj* op, InplaceMode inplace_mode) final {
    auto cond_update = this->Mutate(op->condition, inplace_mode);
    bool cond_unchanged = cond_update.UnchangedOrSameAs(op->condition);
    PrimExpr cond = std::move(cond_update).ValueOrUnchanged(op->condition);
    auto t_update = this->Mutate(op->true_value, inplace_mode);
    bool t_unchanged = t_update.UnchangedOrSameAs(op->true_value);
    PrimExpr t = std::move(t_update).ValueOrUnchanged(op->true_value);
    auto f_update = this->Mutate(op->false_value, inplace_mode);
    bool f_unchanged = f_update.UnchangedOrSameAs(op->false_value);
    PrimExpr f = std::move(f_update).ValueOrUnchanged(op->false_value);
    if (cond_unchanged && t_unchanged && f_unchanged) {
      return ffi::Unchanged();
    } else {
      int cond_lanes = GetLanesOrVScaleFactor(cond.ty());
      int t_lanes = GetLanesOrVScaleFactor(t.ty());
      int f_lanes = GetLanesOrVScaleFactor(f.ty());
      int lanes = std::max(std::max(cond_lanes, t_lanes), f_lanes);
      bool is_scalable =
          cond.ty().IsScalableVector() || t.ty().IsScalableVector() || f.ty().IsScalableVector();
      return prim::Select(BroadcastTo(cond, lanes, is_scalable), BroadcastTo(t, lanes, is_scalable),
                          BroadcastTo(f, lanes, is_scalable));
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::CastObj* op, InplaceMode inplace_mode) final {
    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);
    if (value_unchanged) {
      return ffi::Unchanged();
    } else {
      if (value.ty().IsScalableVector()) {
        return prim::Cast(PrimType::ScalableVector(op->ty.as_or_throw<PrimType>().code(),
                                                   op->ty.as_or_throw<PrimType>().bits(),
                                                   value.ty().VScaleFactor()),
                          value);
      } else {
        return prim::Cast(op->ty.as_or_throw<PrimType>().WithLanes(value.ty().lanes()), value);
      }
    }
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    PrimExpr original = op->args[0].as_or_throw<PrimExpr>();
    PrimExpr value = Mutate(original, inplace_mode).ValueOrUnchanged(original);
    if (value.same_as(original)) return ffi::Unchanged();
    PrimType result_type = op->ty.as_or_throw<PrimType>().WithLanes(value.ty().lanes());
    return Call(result_type, op->op, {value}, op->attrs, op->ty_args, op->loc);
  }

  // TensorLoad
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    auto load = ffi::GetRef<TensorLoad>(op);

    auto fmutate = [this](const PrimExpr& index) {
      return this->Mutate(index).ValueOrUnchanged(index);
    };
    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);

    if (!indices.same_as(op->indices)) {
      return TensorLoad(op->source.as_or_throw<tvm::tirx::TensorVar>(), indices, op->loc);
    }

    return load;
  }

  // TensorStore
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    auto store = ffi::GetRef<TensorStore>(op);

    auto fmutate = [this](const PrimExpr& index) {
      return this->Mutate(index).ValueOrUnchanged(index);
    };
    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);

    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);

    if (!indices.same_as(op->indices) || !value_unchanged) {
      TVM_FFI_ICHECK(!op->dest.as_or_throw<TensorVar>().dtype().IsScalableVector())
          << "Vectorizing over scalable tensor elements is not supported in vectorizer.";
      // How many lanes of indexing are present in the index and
      // tensor element type, excluding the last index.
      int other_index_lanes = op->dest.as_or_throw<TensorVar>().dtype().lanes();
      for (size_t i = 0; i < indices.size() - 1; i++) {
        other_index_lanes *= indices[i].ty().lanes();
        // Only allow the last index to be scalable
        TVM_FFI_ICHECK(!indices[i].ty().IsScalableVector())
            << "Only the last index can be scalable.";
      }

      // The total number of lanes of indexing, including the last index.
      PrimType last_index_dtype = indices[indices.size() - 1].ty();
      int lanes_in_last_index = GetLanesOrVScaleFactor(last_index_dtype);
      int index_lanes = other_index_lanes * lanes_in_last_index;

      // The total number of lanes in this store operation.  Either
      // the index or the value will be broadcast out to this number
      // of lanes, depending on which has more lanes.
      int value_dtype_lanes = GetLanesOrVScaleFactor(value.ty());
      bool is_last_index_scalable = last_index_dtype.IsScalableVector();
      int total_lanes = std::max(index_lanes, value_dtype_lanes);

      TVM_FFI_ICHECK_EQ(total_lanes % other_index_lanes, 0)
          << "When storing to tensor " << op->dest.as_or_throw<TensorVar>()->name
          << ", cannot produce " << total_lanes
          << " lanes of storage location by changing the last index.";
      int last_index_lanes = total_lanes / other_index_lanes;

      // Broadcast the last index such that the total number of index
      // lanes matches the desired number.
      indices.Set(indices.size() - 1, BroadcastTo(indices[indices.size() - 1], last_index_lanes,
                                                  is_last_index_scalable));

      auto writer = store.CopyOnWrite();
      writer->indices = indices;
      writer->value = BroadcastTo(value, total_lanes, is_last_index_scalable);
    }

    return store;
  }

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    PrimExpr original = op->value.as_or_throw<PrimExpr>();
    PrimExpr value = Mutate(original).ValueOrUnchanged(original);
    TVM_FFI_ICHECK(VarRemapGet(op->var) == nullptr) << "SSA violation, a single var is bound twice";
    if (value.ty() != original.ty()) {
      Var new_var(op->var->name, value.ty(), op->var->loc);
      VarRemapSet(op->var, new_var);
      return Bind(new_var, value, op->loc);
    } else {
      VarRemapSet(op->var, op->var);
      if (value.same_as(op->value)) {
        return ffi::Unchanged();
      } else {
        return Bind(op->var, value, op->loc);
      }
    }
  }

 private:
  const VectorizationAnalysis::Plan& plan_;
  sym::Analyzer analyzer_;
  template <typename TOp, typename T>
  PrimExpr BinaryVec(const T* op, InplaceMode inplace_mode) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int a_lanes = GetLanesOrVScaleFactor(a.ty());
      int b_lanes = GetLanesOrVScaleFactor(b.ty());
      int lanes = std::max(a_lanes, b_lanes);
      bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
      return TOp(BroadcastTo(a, lanes, is_scalable), BroadcastTo(b, lanes, is_scalable), op->loc);
    }
  }
  template <typename T, typename FCompute>
  PrimExpr AddSubVec(const T* op, FCompute fcompute, InplaceMode inplace_mode) {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int a_lanes = GetLanesOrVScaleFactor(a.ty());
      int b_lanes = GetLanesOrVScaleFactor(b.ty());
      int lanes = std::max(a_lanes, b_lanes);
      if (lanes != 1) {
        const prim::RampObj* b_ramp = b.as<prim::RampObj>();
        const prim::RampObj* a_ramp = a.as<prim::RampObj>();
        if (a.ty().IsScalar() && b_ramp) {
          return prim::Ramp(fcompute(a, b_ramp->base),
                            fcompute(IntImm(b_ramp->stride.ty(), 0), b_ramp->stride),
                            b_ramp->lanes);
        }
        if (b.ty().IsScalar() && a_ramp) {
          return prim::Ramp(fcompute(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
      return fcompute(BroadcastTo(a, lanes, is_scalable), BroadcastTo(b, lanes, is_scalable));
    }
  }
};

class VectorizeSkipper : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<ForObj>();
    if (op->kind == ForKind::kVectorized) {
      auto copy = ffi::make_object<ForObj>(*op);
      copy->kind = ForKind::kDefault;
      return For(copy);
    } else {
      return stmt;
    }
  }
};

Stmt SkipVectorize(Stmt stmt) {
  return ffi::make_object<VectorizeSkipper>()
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(stmt));
}

class LoopVectorizer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  explicit LoopVectorizer(const Function& func)
      : target_(func->attrs.GetAttr(tvm::attr::kTarget)),
        noalias_(func->attrs.GetAttr<bool>("tirx.noalias").value_or(false)) {
    for (const Var& param : func->params) {
      if (param.as<TensorVar>()) roots_[param.get()] = param.get();
    }
  }

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    TrackStorageRoot(op);
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    if (op->kind != ForKind::kVectorized) return StmtExprMutator::Mutate_(op, inplace_mode);
    if (IsSupportedLoop(op)) {
      PrimExpr extent = Range::GetExtentFromBeginEnd(op->begin, op->end);
      bool rvv = TargetHasRVV(target_);
      if (auto plan = VectorizationAnalysis(op->var, extent, noalias_, roots_, rvv).Run(op->body)) {
        return ffi::make_object<Vectorizer>(op->var, extent, *plan)
            ->Mutate(op->body, InplaceMode::kDisallow)
            .ValueOrUnchanged(op->body);
      }
      if (rvv) {
        if (auto versioned = VectorizeFullChunks(op, extent)) return *versioned;
      }
    }
    // Preserve the complete original loop, including bindings, order, effects,
    // multiplicity and annotations. Nested candidates in this rejected region
    // also stay scalar; never split off or retry individual statements.
    return SkipVectorize(ffi::GetRef<Stmt>(op));
  }

 private:
  void TrackStorageRoot(const BindObj* op) {
    // FlattenTensor introduces layout-free views of parameter storage. Only
    // identical element types and an unadjusted data pointer preserve the
    // physical index map used by the dependence checks.
    auto tensor = op->var.as<TensorVar>();
    if (!tensor || tensor->ty()->layout.has_value()) return;
    const auto* decl = op->value.as<CallObj>();
    if (!decl) return;
    if (decl->op.same_as(alloc_tensor_op())) {
      // Ordinary allocations outside the candidate loop are distinct storage
      // roots. Explicit placement and allocations inside the candidate remain
      // outside this analysis.
      if (decl->args.empty()) roots_[tensor->get()] = tensor->get();
      return;
    }
    if (!decl->op.same_as(decl_tensor_op()) || decl->args.size() != 1) return;
    const auto* data = decl->args[0].as<CallObj>();
    if (!data || !data->op.same_as(tensor_data_ptr_op()) || data->args.size() != 1) return;
    auto source = data->args[0].as<TensorVar>();
    if (!source || source->dtype() != tensor->dtype()) return;
    auto root = roots_.find(source->get());
    if (root != roots_.end()) roots_[tensor->get()] = root->second;
  }

  // RVV tail loops generated by split contain one guarded store. Version the
  // whole chunk instead of speculating any memory access across its guard. The
  // original guarded scalar loop handles partial and entirely inactive chunks.
  ffi::Optional<Stmt> VectorizeFullChunks(const ForObj* op, const PrimExpr& extent) {
    const auto* count = extent.as<IntImmObj>();
    if (!count || op->body->seq.size() != 1) return std::nullopt;
    const auto* guard = op->body->seq[0].as<IfObj>();
    if (!guard || guard->else_case || guard->then_case->seq.size() != 1 ||
        !guard->then_case->seq[0].as<TensorStoreObj>())
      return std::nullopt;
    const auto* condition = guard->condition.as<prim::LTObj>();
    if (!condition || SideEffect(guard->condition) != CallEffectKind::kPure) return std::nullopt;
    auto coefficients = DetectLinearEquation(condition->a - condition->b, {op->var});
    if (coefficients.size() != 2 || !IsOne(coefficients[0])) return std::nullopt;
    auto plan =
        VectorizationAnalysis(op->var, extent, noalias_, roots_, true).Run(guard->then_case);
    if (!plan) return std::nullopt;
    // Check every original lane, rather than relying on a no-overflow
    // assumption when replacing an affine predicate by its last-lane value.
    PrimExpr full_chunk = IntImm(PrimType::Bool(), 1);
    for (int64_t lane = 0; lane < count->value; ++lane) {
      auto substitute = ffi::make_object<StmtExprMutator>();
      substitute->VarRemapSet(op->var, IntImm(op->var.ty(), lane));
      PrimExpr lane_condition = substitute->Mutate(guard->condition, InplaceMode::kDisallow)
                                    .ValueOrUnchanged(guard->condition);
      full_chunk = full_chunk && lane_condition;
    }
    Stmt vector_body = ffi::make_object<Vectorizer>(op->var, extent, *plan)
                           ->Mutate(guard->then_case, InplaceMode::kDisallow)
                           .ValueOrUnchanged(guard->then_case);
    return If(full_chunk, SeqStmt({vector_body}), SeqStmt({SkipVectorize(ffi::GetRef<Stmt>(op))}),
              op->loc);
  }

  bool IsSupportedLoop(const ForObj* op) const {
    if (!IsZero(op->begin) || !op->HasTrivialStep()) return false;
    PrimType index_type = op->var.ty();
    if (!index_type.IsScalar() || !index_type.MatchesCode(DLDataTypeCode::kDLInt) ||
        (index_type.bits() != 32 && index_type.bits() != 64))
      return false;
    PrimExpr lanes = Range::GetExtentFromBeginEnd(op->begin, op->end);
    if (const auto* count = lanes.as<IntImmObj>()) {
      const std::initializer_list<int64_t> supported = {2, 4, 8, 16, 32, 64};
      return std::find(supported.begin(), supported.end(), count->value) != supported.end();
    }
    if (!TargetHasVLA(target_)) return false;
    const auto* mul = lanes.as<prim::MulObj>();
    if (!mul) return false;
    const auto* count = mul->a.as<IntImmObj>();
    if (!count || !IsVScaleCall(mul->b)) {
      count = mul->b.as<IntImmObj>();
      if (!count || !IsVScaleCall(mul->a)) return false;
    }
    const std::initializer_list<int64_t> supported = {2, 4, 8, 16};
    return std::find(supported.begin(), supported.end(), count->value) != supported.end();
  }

  ffi::Optional<Target> target_;
  bool noalias_;
  std::unordered_map<const VarObj*, const VarObj*> roots_;
};

namespace transform {

// TODO(tvm-team): Make it as a target property.
Pass VectorizeLoop(bool enable_vectorize) {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    if (enable_vectorize) {
      n->body = ffi::make_object<LoopVectorizer>(f)
                    ->Mutate(n->body, InplaceMode::kAllow)
                    .ValueOrUnchanged(n->body);
    } else {
      n->body = ffi::make_object<VectorizeSkipper>()
                    ->Mutate(n->body, InplaceMode::kAllow)
                    .ValueOrUnchanged(n->body);
    }
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.VectorizeLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.VectorizeLoop", VectorizeLoop);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
