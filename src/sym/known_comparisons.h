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
 * \file sym/known_comparisons.h
 * \brief Private directly known comparison state for the rewrite simplifier.
 */
#ifndef TVM_SYM_KNOWN_COMPARISONS_H_
#define TVM_SYM_KNOWN_COMPARISONS_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/pattern_match.h>
#include <tvm/sym/analyzer.h>

#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "constraint_extract.h"

namespace tvm {
namespace sym {

// Retain normalized direct comparisons, without transitive inference or calls
// back into the analyzer. The rewriter owns ingestion, reset and query scopes.
class KnownComparisons {
 public:
  CompareResult TryCompare(const PrimExpr& lhs, const PrimExpr& rhs) const;
  void IngestBinding(const Var& var, const PrimExpr& expr);
  void IngestBinding(const Var& var, const Range& range);
  void IngestPredicate(const PrimExpr& expr) { AddKnown(expr, &scoped_knowns_); }

 private:
  // Structural keys avoid comparing expression trees for every stored fact.
  enum class Key : size_t {};
  Key ExprToKey(const PrimExpr& expr);
  std::optional<Key> ExprToPreviousKey(const PrimExpr& expr) const;
  std::unordered_map<PrimExpr, Key, ffi::StructuralHash, ffi::StructuralEqual> expr_to_key_;

  // lhs OP rhs + offset. Strict integer comparisons normalize to <= or >=.
  struct Comparison {
    Comparison(Key lhs, Key rhs, int64_t offset, CompareResult result);
    std::optional<Comparison> WithLHS(Key new_lhs) const;
    Key lhs_;
    Key rhs_;
    int64_t offset_{0};
    CompareResult result_{CompareResult::kInconsistent};
  };

  static CompareResult Reverse(CompareResult result);
  static std::tuple<PrimExpr, PrimExpr, int64_t> ExtractOffsets(const PrimExpr& lhs,
                                                                const PrimExpr& rhs);
  std::optional<Comparison> FromExpr(const PrimExpr& expr);
  void AddKnown(const PrimExpr& expr, std::vector<Comparison>* vec);
  std::vector<Comparison> CollectDirectComparisons(Key lhs_key, Key rhs_key) const;
  CompareResult MergeComparisons(const std::vector<Comparison>& lhs_to_rhs, int64_t offset) const;

  // Keep the defining Var separate from expression keys: type promotion may
  // wrap it in a Cast, but a temporary binding must still replace its facts.
  std::unordered_map<Var, std::vector<Comparison>> bindings_;
  std::vector<Comparison> scoped_knowns_;
};

// Internal utility, return the CompareResult resulting from swapping
// the left-hand side with the right-hand side.
inline CompareResult KnownComparisons::Reverse(CompareResult res) {
  switch (res) {
    case CompareResult::kInconsistent:
      return CompareResult::kInconsistent;
    case CompareResult::kEQ:
      return CompareResult::kEQ;
    case CompareResult::kLT:
      return CompareResult::kGT;
    case CompareResult::kLE:
      return CompareResult::kGE;
    case CompareResult::kGT:
      return CompareResult::kLT;
    case CompareResult::kGE:
      return CompareResult::kLE;
    case CompareResult::kNE:
      return CompareResult::kNE;
    case CompareResult::kUnknown:
      return CompareResult::kUnknown;
    default:
      TVM_FFI_THROW(InternalError) << "Invalid CompareResult: " << static_cast<int>(res);
  }
}

// Internal utility, extract constant offsets out of the two sides of
// a comparison.  Given lhs and rhs, return a tuple of three elements
// (lhs_inner, rhs_inner, offset), such that (lhs OP rhs) and
// (lhs_inner OP rhs_inner + offset) are equivalent.
inline std::tuple<PrimExpr, PrimExpr, int64_t> KnownComparisons::ExtractOffsets(
    const PrimExpr& lhs, const PrimExpr& rhs) {
  auto extract_offset = [](const PrimExpr& expr) -> std::pair<PrimExpr, ffi::BigInt> {
    PVar<PrimExpr> x;
    PVar<IntImm> c;
    if ((x + c).Match(expr)) {
      return {x.Eval(), c.Eval()->value};
    } else if ((x - c).Match(expr)) {
      return {x.Eval(), -c.Eval()->value};
    } else if (c.Match(expr)) {
      return {0, c.Eval()->value};
    } else {
      return {expr, 0};
    }
  };

  auto lhs_split = extract_offset(lhs);
  auto rhs_split = extract_offset(rhs);
  if (auto offset = (rhs_split.second - lhs_split.second).as<int64_t>(); offset.has_value()) {
    return {lhs_split.first, rhs_split.first, *offset};
  }
  return {lhs, rhs, 0};
}

inline std::optional<KnownComparisons::Comparison> KnownComparisons::FromExpr(
    const PrimExpr& expr) {
  CompareResult res;
  PVar<PrimExpr> x, y;
  if ((x <= y).Match(expr)) {
    res = CompareResult::kLE;
  } else if ((x >= y).Match(expr)) {
    res = CompareResult::kGE;
  } else if ((x < y).Match(expr)) {
    res = CompareResult::kLT;
  } else if ((x > y).Match(expr)) {
    res = CompareResult::kGT;
  } else if ((x == y).Match(expr)) {
    res = CompareResult::kEQ;
  } else if ((x != y).Match(expr)) {
    res = CompareResult::kNE;
  } else {
    return std::nullopt;
  }

  PrimExpr lhs_expr = x.Eval();
  PrimExpr rhs_expr = y.Eval();

  if (!lhs_expr.ty().MatchesCode(DLDataTypeCode::kDLInt) ||
      !rhs_expr.ty().MatchesCode(DLDataTypeCode::kDLInt)) {
    return std::nullopt;
  }
  if (lhs_expr.as<IntImmObj>() && rhs_expr.as<IntImmObj>()) {
    return std::nullopt;
  }

  auto [lhs, rhs, offset] = ExtractOffsets(lhs_expr, rhs_expr);
  ffi::BigInt normalized_offset = offset;
  if (res == CompareResult::kLT) --normalized_offset;
  if (res == CompareResult::kGT) ++normalized_offset;
  if (!normalized_offset.as<int64_t>().has_value()) return std::nullopt;
  Key lhs_key = ExprToKey(lhs);
  Key rhs_key = ExprToKey(rhs);

  return Comparison(lhs_key, rhs_key, offset, res);
}

inline KnownComparisons::Comparison::Comparison(Key lhs, Key rhs, int64_t offset,
                                                CompareResult result)
    : lhs_(lhs), rhs_(rhs), offset_(offset), result_(result) {
  // Integer strict comparisons become non-strict comparisons with a unit offset.
  if (result_ == CompareResult::kLT) {
    result_ = CompareResult::kLE;
    offset_ -= 1;
  }
  if (result_ == CompareResult::kGT) {
    result_ = CompareResult::kGE;
    offset_ += 1;
  }
}

inline std::optional<KnownComparisons::Key> KnownComparisons::ExprToPreviousKey(
    const PrimExpr& expr) const {
  auto it = expr_to_key_.find(expr);
  if (it != expr_to_key_.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

inline KnownComparisons::Key KnownComparisons::ExprToKey(const PrimExpr& expr) {
  if (auto prev = ExprToPreviousKey(expr)) {
    return prev.value();
  } else {
    Key new_key = Key(expr_to_key_.size());
    expr_to_key_[expr] = new_key;
    return new_key;
  }
}

inline std::optional<KnownComparisons::Comparison> KnownComparisons::Comparison::WithLHS(
    Key new_lhs) const {
  if (new_lhs == lhs_) {
    return *this;
  } else if (new_lhs == rhs_) {
    auto reversed = (-ffi::BigInt(offset_)).as<int64_t>();
    if (!reversed.has_value()) return std::nullopt;
    return Comparison(rhs_, lhs_, *reversed, Reverse(result_));
  } else {
    return std::nullopt;
  }
}

inline void KnownComparisons::AddKnown(const PrimExpr& expr, std::vector<Comparison>* vec) {
  // Only stable facts may enter the expression-key table. Queries then need no
  // side-effect traversal before the integer fast path or key lookup.
  if (SideEffect(expr) > CallEffectKind::kPure) return;
  for (const auto& subexpr : ExtractConstraints(expr, false)) {
    if (auto cmp = FromExpr(subexpr)) vec->push_back(*cmp);
  }
}

inline void KnownComparisons::IngestBinding(const Var& var, const Range& range) {
  auto& knowns = bindings_[var];
  knowns.clear();
  PrimExpr prim_var = var.as_or_throw<PrimExpr>();
  if (prim::IsConstInt(range->extent, 1)) {
    AddKnown(prim_var == range->begin, &knowns);
  } else {
    AddKnown(prim_var >= range->begin, &knowns);
    AddKnown(prim_var < range->begin + range->extent, &knowns);
  }
}

inline void KnownComparisons::IngestBinding(const Var& var, const PrimExpr& expr) {
  IngestBinding(var, Range::FromMinExtent(expr, 1));
}

inline CompareResult KnownComparisons::TryCompare(const PrimExpr& lhs_expr,
                                                  const PrimExpr& rhs_expr) const {
  // Currently only supports integer checks
  if (!lhs_expr.ty().MatchesCode(DLDataTypeCode::kDLInt) ||
      !rhs_expr.ty().MatchesCode(DLDataTypeCode::kDLInt)) {
    return CompareResult::kUnknown;
  }

  // Bail out early if possible.  This int check should have been
  // constant-folded earlier, so this check shouldn't occur.
  auto* x_int = lhs_expr.as<IntImmObj>();
  auto* y_int = rhs_expr.as<IntImmObj>();
  if (x_int && y_int) {
    if (x_int->value < y_int->value) {
      return CompareResult::kLT;
    } else if (x_int->value > y_int->value) {
      return CompareResult::kGT;
    } else {
      return CompareResult::kEQ;
    }
  }

  auto [lhs, rhs, offset] = ExtractOffsets(lhs_expr, rhs_expr);
  auto lhs_key = ExprToPreviousKey(lhs);
  auto rhs_key = ExprToPreviousKey(rhs);

  if (!lhs_key.has_value() || !rhs_key.has_value()) {
    return CompareResult::kUnknown;
  }

  auto lhs_to_rhs = CollectDirectComparisons(lhs_key.value(), rhs_key.value());
  return MergeComparisons(lhs_to_rhs, offset);
}

inline std::vector<KnownComparisons::Comparison> KnownComparisons::CollectDirectComparisons(
    Key lhs_key, Key rhs_key) const {
  std::vector<Comparison> output;

  auto append_known = [&](Comparison cmp) {
    if (auto normalized = cmp.WithLHS(lhs_key)) {
      if (normalized.value().rhs_ == rhs_key) {
        output.push_back(normalized.value());
      }
    }
  };

  for (const auto& [var, knowns] : bindings_) {
    for (const auto& known : knowns) append_known(known);
  }
  for (const auto& known : scoped_knowns_) {
    append_known(known);
  }

  return output;
}

inline CompareResult KnownComparisons::MergeComparisons(const std::vector<Comparison>& lhs_to_rhs,
                                                        int64_t offset) const {
  // Just because we found a comparison involving LHS and RHS doesn't
  // mean that it's useful.  e.g. Knowing that `x < y` doesn't let us
  // prove whether `x + 5 < y`.
  CompareResult result = CompareResult::kUnknown;
  for (const auto& cmp : lhs_to_rhs) {
    switch (cmp.result_) {
      case CompareResult::kInconsistent:
        result = CompareResult::kInconsistent;
        break;

      case CompareResult::kEQ:
        if (offset == cmp.offset_) {
          result = result & CompareResult::kEQ;
        } else {
          result = result & CompareResult::kNE;
        }
        break;

      case CompareResult::kLE:
        if (cmp.offset_ < offset) {
          result = result & CompareResult::kLT;
        } else if (cmp.offset_ <= offset) {
          result = result & CompareResult::kLE;
        }
        break;

      case CompareResult::kGE:
        if (cmp.offset_ > offset) {
          result = result & CompareResult::kGT;
        } else if (cmp.offset_ >= offset) {
          result = result & CompareResult::kGE;
        }
        break;

      case CompareResult::kNE:
        if (offset == cmp.offset_) {
          result = result & CompareResult::kNE;
        }
        break;

      case CompareResult::kUnknown:
        break;

      case CompareResult::kGT:
      case CompareResult::kLT:
        TVM_FFI_THROW(InternalError)
            << "Internal error, normalized comparisons should only include <= and >=";

      default:
        TVM_FFI_THROW(InternalError) << "Invalid CompareResult: " << static_cast<int>(cmp.result_);
    }
  }

  return result;
}

}  // namespace sym
}  // namespace tvm

#endif  // TVM_SYM_KNOWN_COMPARISONS_H_
