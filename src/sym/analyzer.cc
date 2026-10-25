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
 * \file tvm/sym/analyzer.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>

#include <algorithm>
#include <unordered_set>

#include "const_fold.h"
#include "constraint_helpers.h"
#include "product_normal_form.h"

namespace tvm {
namespace sym {

thread_local int AllowUIntAsIndex::depth_ = 0;

bool AllowUIntAsIndex::Enabled() { return depth_ > 0; }

void AllowUIntAsIndex::EnterWithScope() { ++depth_; }

void AllowUIntAsIndex::ExitWithScope() {
  TVM_FFI_ICHECK(depth_ > 0) << "AllowUIntAsIndex exit without a matching enter";
  --depth_;
}

AnalyzerObj::AnalyzerObj()
    : const_int_bound(this),
      modular_set(this),
      rewrite_simplify(this),
      canonical_simplify(this),
      int_set(this),
      known_comparisons(this),
      z3_prover(this) {}

ConstraintContext AnalyzerObj::GetConstraintContext() const {
  // Never return the sealed internal position: callers may extend this child.
  return last_snapshot_->Snapshot();
}

namespace detail {
namespace {
// Normalize empty captured children to their source and captured prefix. This
// token is private to the coordinator; backends only see ordered facts.
struct ContextPosition {
  const ConstraintContextObj* source;
  int64_t prefix;

  bool operator==(const ContextPosition& other) const {
    return source == other.source && prefix == other.prefix;
  }
};
}  // namespace
}  // namespace detail

void AnalyzerObj::SyncConstraintContext(const ConstraintContext& context) {
  TVM_FFI_CHECK(!synchronizing_ && active_queries_ == 0, RuntimeError)
      << "Cannot synchronize an Analyzer during its active query or synchronization";
  auto position = [](const ConstraintContextObj* source, int64_t prefix) {
    while (prefix == 0 && source->parent_) {
      prefix = source->parent_prefix_size_;
      source = source->parent_.get();
    }
    return detail::ContextPosition{source, prefix};
  };
  auto next_position = position(context.get(), context->LocalSize());
  auto old_position = position(last_snapshot_.get(), last_snapshot_->LocalSize());
  if (next_position == old_position) return;

  // Stable Var identities have one definition in a premise environment.
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> definitions;
  context->ForEach([&](const Constraint& fact) {
    ffi::Optional<Var> var;
    if (const auto* binding = fact.as<BindingConstraintObj>()) var = binding->var;
    if (const auto* range = fact.as<RangeConstraintObj>()) var = range->var;
    if (var)
      TVM_FFI_CHECK(definitions.insert(*var).second, ValueError)
          << "A constraint context cannot redefine variable " << *var;
  });

  ConstraintContext next = context->Snapshot();  // One shared capture per change.
  synchronizing_ = true;
  bool additive = next_position.source == old_position.source
                      ? next_position.prefix >= old_position.prefix
                      : next_position.source->parent_ &&
                            position(next_position.source->parent_.get(),
                                     next_position.source->parent_prefix_size_) == old_position;
  if (additive) {
    int64_t begin = next_position.source == old_position.source ? old_position.prefix : 0;
    for (int64_t i = begin; i < next_position.prefix; ++i)
      IngestConstraint(next_position.source->constraints_[i]);
  } else {
    // Reset ALL peers before ordered replay. A binding may query its peers,
    // which must agree on the valid prefix rather than retain removed facts.
    const_int_bound.Reset();
    modular_set.Reset();
    rewrite_simplify.Reset();
    canonical_simplify.Reset();
    int_set.Reset();
    known_comparisons.Reset();
    z3_prover.Reset();
    next->ForEach([&](const Constraint& fact) { IngestConstraint(fact); });
  }
  last_snapshot_ = std::move(next);
  synchronizing_ = false;
  // An exception ends this analyzer's lifetime; partially updated state is not reusable.
}

void AnalyzerObj::IngestBinding(const Var& var, const PrimExpr& expr) {
  PrimExpr value = canonical_simplify(expr);
  value = rewrite_simplify(value);
  const_int_bound.IngestBinding(var, value);
  modular_set.IngestBinding(var, value);
  rewrite_simplify.IngestBinding(var, value);
  canonical_simplify.IngestBinding(var, value);
  int_set.IngestBinding(var, value);
  known_comparisons.IngestBinding(var, expr);
  z3_prover.IngestBinding(var, expr);
}

void AnalyzerObj::IngestBinding(const Var& var, const Range& range) {
  if (prim::IsOne(range->extent)) {
    IngestBinding(var, range->begin);
  } else {
    const_int_bound.IngestBinding(var, range);
    int_set.IngestBinding(var, range);
    known_comparisons.IngestBinding(var, range);
    z3_prover.IngestBinding(var, range);
  }
}

namespace {
// Collectors retain syntax, including branch-likelihood annotations. Their truth
// is the wrapped predicate's truth; remove only the annotation in the analyzer's
// derived input, without changing the captured premise or the transformed IR.
PrimExpr PredicateTruth(const PrimExpr& expression) {
  if (const auto* call = expression.as<CallObj>();
      call && call->op.same_as(prim::likely_op()) && call->args.size() == 1) {
    if (auto argument = call->args[0].as<PrimExpr>()) return PredicateTruth(*argument);
  }
  if (const auto* op = expression.as<prim::NotObj>()) return prim::Not(PredicateTruth(op->a));
  if (const auto* op = expression.as<prim::AndObj>())
    return PredicateTruth(op->a) && PredicateTruth(op->b);
  if (const auto* op = expression.as<prim::OrObj>())
    return PredicateTruth(op->a) || PredicateTruth(op->b);
  if (!expression.ty().MatchesCode(DLDataTypeCode::kDLBool))
    return expression != IntImm(expression.ty(), 0);
  return expression;
}
}  // namespace

void AnalyzerObj::IngestConstraint(const Constraint& fact) {
  if (const auto* binding = fact.as<BindingConstraintObj>()) {
    // A captured mutable load is data, not an enduring substitution to a future load.
    if (SideEffect(binding->value) <= CallEffectKind::kPure)
      IngestBinding(binding->var, binding->value);
  } else if (const auto* range = fact.as<RangeConstraintObj>()) {
    if (SideEffect(range->range->begin) <= CallEffectKind::kPure &&
        SideEffect(range->range->extent) <= CallEffectKind::kPure)
      IngestBinding(range->var, range->range);
  } else if (const auto* predicate = fact.as<PredicateConstraintObj>()) {
    const auto facts = PreparePredicate(predicate->condition);
    for (const PrimExpr& condition : facts) {
      const_int_bound.IngestPredicate(condition);
      modular_set.IngestPredicate(condition);
      rewrite_simplify.IngestPredicate(condition);
      int_set.IngestPredicate(condition);
      known_comparisons.IngestPredicate(condition);
      z3_prover.IngestPredicate(condition);
    }
  }
}

std::vector<PrimExpr> AnalyzerObj::PreparePredicate(const PrimExpr& predicate) {
  // assume_always promises truth, not stable values of mutable reads. Until
  // translation models that distinction, effectful facts are conservatively ignored.
  if (SideEffect(predicate) > CallEffectKind::kPure) return {};
  PrimType type = predicate.ty();
  if (!type.IsScalar() ||
      !(type.MatchesCode(DLDataTypeCode::kDLBool) || type.MatchesCode(DLDataTypeCode::kDLInt) ||
        type.MatchesCode(DLDataTypeCode::kDLUInt)))
    return {};
  // Normalize against the previous valid prefix before installing this fact
  // in any backend. In particular, !(i != 0) must reach bounds as i == 0;
  // simplifying after installation could prove the premise itself true and
  // discard the useful literal. Scalar integer guards use nonzero truth.
  PrimExpr original = PredicateTruth(predicate);
  PrimExpr condition = rewrite_simplify(original);
  // Bounds and literal matching also use exact expression structure. Keep the
  // original truth alongside its normalized equivalent so canonical and
  // rewrite queries can both recognize the premise. Compute both before any
  // backend ingests either; otherwise normalization can erase the premise.
  std::vector<PrimExpr> facts{original};
  detail::CollectDerivedConstraintFacts(original, &facts);
  if (!condition.same_as(original)) {
    facts.push_back(condition);
    detail::CollectDerivedConstraintFacts(condition, &facts);
  }
  return facts;
}

bool AnalyzerObj::CanProveGreaterEqual(const PrimExpr& expr, int64_t lower_bound) {
  if (const auto* ptr = expr.as<IntImmObj>()) {
    return ptr->value >= lower_bound;
  }
  auto bd = this->const_int_bound(this->rewrite_simplify(expr));
  if (bd->min_value >= lower_bound) return true;
  return false;
}

bool AnalyzerObj::CanProveLess(const PrimExpr& expr, int64_t upper_bound) {
  if (const auto* ptr = expr.as<IntImmObj>()) {
    return ptr->value < upper_bound;
  }
  auto bd = this->const_int_bound(this->rewrite_simplify(expr));
  if (bd->max_value < upper_bound) return true;
  return false;
}

bool AnalyzerObj::CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs) {
  const auto* clhs = lhs.as<IntImmObj>();
  const auto* crhs = rhs.as<IntImmObj>();
  if (clhs && crhs) return clhs->value == crhs->value;
  if (prim::is_pos_inf(lhs) || prim::is_neg_inf(lhs) || prim::is_pos_inf(rhs) ||
      prim::is_neg_inf(rhs)) {
    return lhs.same_as(rhs);
  }
  PrimExpr difference = lhs - rhs;
  return CanProve(difference == prim::MakeConst(difference.ty(), 0));
}

bool AnalyzerObj::CanProveLessEqualThanSymbolicShapeValue(const PrimExpr& lhs,
                                                          const PrimExpr& shape) {
  if (this->CanProve(lhs <= shape, ProofStrength::kSymbolicBound)) return true;
  // no need to do further attempt if shape is already a constant.
  if (prim::IsConstInt(shape)) return false;
  // collect constant scale and ignore symbolic part
  // so 32 * n => cscale = 32
  ffi::BigInt cscale = 1;
  auto fcollect = [&](const PrimExpr& expr) {
    if (auto* ptr = expr.as<IntImmObj>()) {
      cscale *= ptr->value;
    }
  };
  UnpackReduction<prim::MulObj>(shape, fcollect);
  cscale = cscale < 0 ? -cscale : cscale;
  int value_bits = shape.ty().bits() - shape.ty().MatchesCode(DLDataTypeCode::kDLInt);
  if (cscale >= (ffi::BigInt(1) << value_bits)) return false;
  PrimExpr const_shape_bound = IntImm(shape.ty(), cscale);
  if (this->CanProve(lhs <= const_shape_bound, ProofStrength::kSymbolicBound)) return true;
  return false;
}

bool AnalyzerObj::CanProve(const PrimExpr& expr, ProofStrength strength) {
  // Avoid potentially expensive simplification unless required.
  if (const auto* ptr = expr.as<IntImmObj>()) {
    return ptr->value != 0;
  }
  PrimExpr simplified = Simplify(expr);
  const auto* as_int = simplified.as<IntImmObj>();
  if (as_int && as_int->value != 0) return true;
  if (strength >= ProofStrength::kSymbolicBound) {
    // NOTE: we intentionally only pattern match common bound predicate i < bound
    // and put this implementation at the top-level.
    // This is to avoid repeatitive calling of this function
    // that causes speed issues.
    // This strategy can only be called from top-level and not from sub-analyzers.
    ffi::Optional<PrimExpr> pos_diff;
    int lower_bound = 0;
    if (const auto* ptr_lt = expr.as<prim::LTObj>()) {
      pos_diff = ptr_lt->b - ptr_lt->a;
      lower_bound = 1;
    }
    if (const auto* ptr_le = expr.as<prim::LEObj>()) {
      pos_diff = ptr_le->b - ptr_le->a;
      lower_bound = 0;
    }
    if (const auto* ptr_gt = expr.as<prim::GTObj>()) {
      pos_diff = ptr_gt->a - ptr_gt->b;
      lower_bound = 1;
    }
    if (const auto* ptr_ge = expr.as<prim::GEObj>()) {
      pos_diff = ptr_ge->a - ptr_ge->b;
      lower_bound = 0;
    }
    if (pos_diff) {
      IntSet iset = this->int_set(this->Simplify(pos_diff.value()));
      if (iset.HasLowerBound()) {
        ConstIntBound relaxed_lower_bound = this->const_int_bound(this->Simplify(iset.min()));
        if (relaxed_lower_bound->min_value >= lower_bound) return true;
      }
    }
  }

  // Z3 is an expensive best-effort fallback. Gate it behind the higher
  // kSymbolicBound strength so the common kDefault path (including deeply
  // recursive internal CanProve calls) never pays the prover cost.
  if (strength >= ProofStrength::kSymbolicBound && z3_prover.CanProve(simplified)) {
    return true;
  }
  return false;
}

PrimExpr AnalyzerObj::Simplify(const PrimExpr& expr, int steps) {
  PrimExpr res = expr;

  // Always starts with a canonical simplification, as some structural property
  // of an expression might be destroyed by rewrite simplification.
  res = this->canonical_simplify(res);

  for (int i = 0; i < steps; ++i) {
    if (prim::IsConstInt(res)) {
      return res;
    }
    if (i % 2 == 0) {
      res = this->rewrite_simplify(res);
    } else {
      res = this->canonical_simplify(res);
    }
  }

  return res;
}

Analyzer AnalyzerObj::Clone() const {
  Analyzer cloned;
  cloned->const_int_bound.CopyFrom(this->const_int_bound);
  cloned->modular_set.CopyFrom(this->modular_set);
  cloned->rewrite_simplify.CopyFrom(this->rewrite_simplify);
  cloned->canonical_simplify.CopyFrom(this->canonical_simplify);
  cloned->int_set.CopyFrom(this->int_set);
  cloned->known_comparisons.CopyFrom(this->known_comparisons);
  cloned->z3_prover.CopyFrom(this->z3_prover);
  cloned->last_snapshot_ = last_snapshot_;
  return cloned;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AnalyzerObj>();
  refl::GlobalDef()
      .def("sym.Analyzer", []() { return Analyzer(); })
      .def("sym.AnalyzerClone", [](Analyzer analyzer) { return analyzer->Clone(); })
      .def_method("sym.AnalyzerSyncConstraintContext", &AnalyzerObj::SyncConstraintContext)
      .def_method("sym.AnalyzerGetConstraintContext", &AnalyzerObj::GetConstraintContext)
      .def("sym.AnalyzerConstIntBound",
           [](Analyzer analyzer, const PrimExpr& expr) { return analyzer->const_int_bound(expr); })

      .def("sym.AnalyzerConstIntBoundIsBound",
           [](Analyzer analyzer, const Var& var) { return analyzer->const_int_bound.IsBound(var); })

      .def("sym.AnalyzerModularSet",
           [](Analyzer analyzer, const PrimExpr& expr) { return analyzer->modular_set(expr); })
      .def("sym.AnalyzerSimplify", [](Analyzer analyzer, const PrimExpr& expr,
                                      int steps) { return analyzer->Simplify(expr, steps); })
      .def("sym.AnalyzerRewriteSimplify",
           [](Analyzer analyzer, const PrimExpr& expr) { return analyzer->rewrite_simplify(expr); })
      .def("sym.AnalyzerGetRewriteSimplifyStats",
           [](Analyzer analyzer) { return analyzer->rewrite_simplify.GetStatsCounters(); })
      .def("sym.AnalyzerResetRewriteSimplifyStats",
           [](Analyzer analyzer) { analyzer->rewrite_simplify.ResetStatsCounters(); })
      .def("sym.AnalyzerCanonicalSimplify",
           [](Analyzer analyzer, const PrimExpr& expr) {
             return analyzer->canonical_simplify(expr);
           })
      .def("sym.AnalyzerIntSet",
           [](Analyzer analyzer, const PrimExpr& expr,
              ffi::Optional<ffi::Map<Var, IntSet>> opt_dom_map) {
             if (opt_dom_map.has_value()) {
               return analyzer->int_set(expr, opt_dom_map.value());
             }
             return analyzer->int_set(expr);
           })

      .def("sym.AnalyzerCanProve",
           [](Analyzer analyzer, const PrimExpr& expr, int strength) {
             return analyzer->CanProve(expr, static_cast<ProofStrength>(strength));
           })
      .def("sym.AnalyzerEnterAllowUintAsIndex",
           []() {
             // With has a noexcept(false) destructor, so it cannot use make_shared.
             auto ctx = std::shared_ptr<With<AllowUIntAsIndex>>(new With<AllowUIntAsIndex>());
             return ffi::Function::FromPacked(
                 [ctx](ffi::PackedArgs, ffi::Any*) mutable { ctx.reset(); });
           })
      .def("sym.AnalyzerSetMaximumRewriteSteps",
           [](Analyzer analyzer, int64_t maximum) {
             analyzer->rewrite_simplify.SetMaximumRewriteSteps(maximum);
           })

      .def_method("sym.AnalyzerCanProveEqual", &AnalyzerObj::CanProveEqual)
      .def("sym.AnalyzerTryCompare",
           [](Analyzer analyzer, const PrimExpr& lhs, const PrimExpr& rhs) {
             return static_cast<int64_t>(analyzer->known_comparisons.TryCompare(lhs, rhs));
           })
      .def("sym.AnalyzerIsZ3Enabled",
           [](Analyzer analyzer) { return analyzer->z3_prover.IsEnabled(); })
      .def("sym.AnalyzerGetSMTLIB2",
           [](Analyzer analyzer, ffi::Optional<PrimExpr> expr) {
             return analyzer->z3_prover.GetSMTLIB2(expr);
           })
      .def("sym.AnalyzerSetZ3TimeoutMs",
           [](Analyzer analyzer, int64_t timeout_ms) {
             analyzer->z3_prover.SetTimeoutMs(static_cast<unsigned>(timeout_ms));
           })
      .def("sym.AnalyzerSetZ3RLimit",
           [](Analyzer analyzer, int64_t rlimit) {
             analyzer->z3_prover.SetRLimit(static_cast<unsigned>(rlimit));
           })
      .def("sym.AnalyzerGetZ3Stats",
           [](Analyzer analyzer) { return analyzer->z3_prover.GetStats(); });
}

}  // namespace sym
}  // namespace tvm
