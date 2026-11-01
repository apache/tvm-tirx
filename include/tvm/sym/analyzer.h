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
 * \file tvm/sym/analyzer.h
 * \brief Algebra expression simplifications.
 */
#ifndef TVM_SYM_ANALYZER_H_
#define TVM_SYM_ANALYZER_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/with_context.h>
#include <tvm/sym/constraint.h>
#include <tvm/sym/int_set.h>

#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
/*! \brief namespace of symbolic analysis. */
namespace sym {
//-------------------------------------------------------
// Base integer analysis API.
//
// We have multiple type of analyzers to do relaxed
// integer set analysis(bound analysis, modulo) and
// equivalence checking and simplification.
//
// Importantly, each analyzer may need result from
// another analyzer.
//-------------------------------------------------------

// Forward declare the analyzer object and its reference handle.
class AnalyzerObj;
class Analyzer;
enum class CompareResult : int;
namespace detail {
class TempConstraintScope;
}  // namespace detail

/*! \brief Integer division semantics used by symbolic analysis. */
enum DivMode {
  /*! \brief Truncated division. */
  kTruncDiv,
  /*! \brief Floor division. */
  kFloorDiv
};

/*!
 * \brief The strength used for top-level condition proofs.
 * \note The higher, the more time consuming it can be.
 *
 * Do not use level beyond kDefault in internal recursive rewriting in sym
 * analysis and only use it at top-level simplification to avoid speed issues.
 */
enum class ProofStrength : int {
  /*! \brief Default analysis followed by the optional Z3 fallback. */
  kDefault = 0,
  /*!
   * \brief Prove using symbolic bound analysis
   */
  kSymbolicBound = 1,
};

/*!
 * \brief Inclusive constant integer bounds, representing [min_value, max_value].
 *
 * The infinity sentinels represent unbounded endpoints, not finite integer values.
 */
class ConstIntBoundObj : public ffi::Object {
 public:
  /*! \brief Inclusive lower endpoint, or kNegInf when unbounded. */
  int64_t min_value;
  /*! \brief Inclusive upper endpoint, or kPosInf when unbounded. */
  int64_t max_value;

  /*! \brief Register the fields exposed through FFI reflection. */
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConstIntBoundObj>()
        .def_ro("min_value", &ConstIntBoundObj::min_value)
        .def_ro("max_value", &ConstIntBoundObj::max_value);
  }

  /*! \brief Number to represent +inf */
  static const constexpr int64_t kPosInf = std::numeric_limits<int64_t>::max();
  /*!
   * \brief Number to represent -inf
   * \note The sentinels satisfy -kPosInf == kNegInf.
   */
  static const constexpr int64_t kNegInf = -kPosInf;

  /*! \brief Compare and hash these value objects structurally. */
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.ConstIntBound", ConstIntBoundObj, ffi::Object);
};

/*!
 * \brief Managed reference to ConstIntBoundObj.
 * \sa ConstIntBoundObj
 */
class ConstIntBound : public ffi::ObjectRef {
 public:
  /*! \brief Construct inclusive bounds; endpoints may use the infinity sentinels. */
  TVM_TIRX_DLL ConstIntBound(int64_t min_value, int64_t max_value);

  /*! \brief Sentinel for an unbounded upper endpoint. */
  static const constexpr int64_t kPosInf = ConstIntBoundObj::kPosInf;
  /*! \brief Sentinel for an unbounded lower endpoint. */
  static const constexpr int64_t kNegInf = ConstIntBoundObj::kNegInf;
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConstIntBound, ffi::ObjectRef, ConstIntBoundObj);
};

/*!
 * \brief Analyzer to get constant integer bound over expression.
 */
class ConstIntBoundAnalyzer {
 public:
  /*! \brief Query results keyed by expression object identity. */
  using BoundMapType =
      std::unordered_map<PrimExpr, ConstIntBound, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  /*!
   * \brief Compute conservative constant bounds under the current premises.
   * \param expr The integer expression to analyze.
   * \return Inclusive bounds; unproved endpoints use the infinity sentinels.
   */
  TVM_TIRX_DLL ConstIntBound operator()(const PrimExpr& expr) const;

  /*!
   * \brief Compute bounds and optionally collect intermediate query results.
   * \param expr The integer expression to analyze.
   * \param bound Optional output map, cleared on entry and borrowed only for this call.
   * Its results describe the current premises and are not updated by later synchronization.
   * \return Inclusive conservative bounds for expr.
   */
  TVM_TIRX_DLL ConstIntBound operator()(const PrimExpr& expr, BoundMapType* bound);

  /*!
   * \brief Check whether a binding or recognized predicate supplies bounds for a variable.
   * \param var The variable.
   * \return True when such information is recorded, even if its bounds are unbounded.
   */
  TVM_TIRX_DLL bool IsBound(const Var& var) const;

 private:
  friend class AnalyzerObj;
  // Clear context-derived facts before replay; retain query settings and diagnostics.
  void Reset();
  // Record a value binding after AnalyzerObj validates and orders the premises.
  void IngestBinding(const Var& var, const PrimExpr& expr);
  // Record a half-open variable range from the accepted premise prefix.
  void IngestBinding(const Var& var, const Range& range);
  // Consume a predicate prepared against the preceding premise prefix.
  void IngestPredicate(const PrimExpr& constraint);
  // Borrow the owning analyzer for cross-backend queries.
  explicit ConstIntBoundAnalyzer(AnalyzerObj* parent);
  // Exported for destruction through the inline Analyzer object allocator.
  TVM_TIRX_DLL ~ConstIntBoundAnalyzer();
  // Copy durable facts/settings into this backend without sharing mutable query state.
  void CopyFrom(const ConstIntBoundAnalyzer& other);
  struct Entry;
  class Impl;
  // Owned backend state; its definition stays inside the compiler library.
  std::unique_ptr<Impl> impl_;
};

/*!
 * \brief Integer congruence class {coeff * x + base | x in Z}.
 *
 * A zero coefficient represents the singleton {base}; a coefficient of one
 * represents all integers. This information can establish divisibility, for
 * example every element of {4 * x | x in Z} is divisible by four.
 */
class ModularSetObj : public ffi::Object {
 public:
  /*! \brief Coefficient of the arbitrary integer multiplier. */
  int64_t coeff;
  /*! \brief Constant offset of the congruence class. */
  int64_t base;

  /*! \brief Register the fields exposed through FFI reflection. */
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModularSetObj>()
        .def_ro("coeff", &ModularSetObj::coeff)
        .def_ro("base", &ModularSetObj::base);
  }

  /*! \brief Compare and hash these value objects structurally. */
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.ModularSet", ModularSetObj, ffi::Object);
};

/*!
 * \brief Managed reference to ModularSetObj.
 * \sa ModularSetObj
 */
class ModularSet : public ffi::ObjectRef {
 public:
  /*! \brief Store coeff and base as supplied, without normalization. */
  TVM_TIRX_DLL ModularSet(int64_t coeff, int64_t base);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ModularSet, ffi::ObjectRef, ModularSetObj);
};

/*!
 * \brief Analyzer to get modular information over expression.
 */
class ModularSetAnalyzer {
 public:
  /*!
   * \brief Compute conservative congruence information under the current premises.
   * \param expr The integer expression to analyze.
   * \return A congruence class containing every possible value of expr.
   */
  TVM_TIRX_DLL ModularSet operator()(const PrimExpr& expr);

 private:
  friend class AnalyzerObj;
  // Clear context-derived facts before replay; retain query settings and diagnostics.
  void Reset();
  // Record a value binding after AnalyzerObj validates and orders the premises.
  void IngestBinding(const Var& var, const PrimExpr& expr);
  // Consume a predicate prepared against the preceding premise prefix.
  void IngestPredicate(const PrimExpr& constraint);
  // Borrow the owning analyzer for cross-backend queries.
  explicit ModularSetAnalyzer(AnalyzerObj* parent);
  // Exported for destruction through the inline Analyzer object allocator.
  TVM_TIRX_DLL ~ModularSetAnalyzer();
  // Copy durable facts/settings into this backend without sharing mutable query state.
  void CopyFrom(const ModularSetAnalyzer& other);
  struct Entry;
  class Impl;
  // Owned backend state; its definition stays inside the compiler library.
  std::unique_ptr<Impl> impl_;
};

/*!
 * \brief Rewrite-rule based simplifier.
 */
class RewriteSimplifier {
 public:
  /*!
   * \brief Simplify an expression under the current premises.
   * \param expr The expression to simplify.
   * \return An equivalent expression under those premises.
   */
  TVM_TIRX_DLL PrimExpr operator()(const PrimExpr& expr);

  /*!
   * \brief Compare signed integer expressions using active premises and constant offsets.
   * \param lhs The left-hand side of the comparison.
   * \param rhs The right-hand side of the comparison.
   * \return The directly known ordering, or kUnknown when it cannot be established.
   */
  TVM_TIRX_DLL CompareResult TryCompare(const PrimExpr& lhs, const PrimExpr& rhs);

  /*!
   * \brief Read cumulative rewrite statistics.
   * \return A snapshot of the current counters.
   */
  TVM_TIRX_DLL ffi::ObjectRef GetStatsCounters() const;

  /*! \brief Clear cumulative rewrite counters without changing facts or the rewrite limit. */
  TVM_TIRX_DLL void ResetStatsCounters();

  /*!
   * \brief Set the maximum allowed cumulative number of successful rewrites.
   *
   * By default, the simplifier may perform as many steps as are
   * required.  If a positive limit is set, then the simplifier will
   * throw an exception when exceeding that number of rewrite steps.
   * This allows tests to guard against performance regressions.
   *
   * Note: To maintain accurate usage counters, `Analyzer` instances
   * should be re-used wherever possible.  For example, TIR
   * transformations should declare a single `Analyzer` that is used
   * throughout the pass. Internal helpers may borrow its `AnalyzerObj*`.
   *
   * The limit applies to the cumulative counter, not separately to each query.
   * ResetStatsCounters starts a new accounting interval.
   * \param maximum Positive rewrite limit, or a nonpositive value for no limit.
   */
  TVM_TIRX_DLL void SetMaximumRewriteSteps(int64_t maximum);

 private:
  friend class AnalyzerObj;
  // Clear context-derived facts before replay; retain query settings and diagnostics.
  void Reset();
  // Keep the original relation alongside the simplified substitution.
  void IngestBinding(const Var& var, const PrimExpr& value, const PrimExpr& original);
  // Record a range relation after AnalyzerObj validates and orders the premises.
  void IngestBinding(const Var& var, const Range& range);
  // Consume a predicate prepared against the preceding premise prefix.
  void IngestPredicate(const PrimExpr& constraint);
  friend class CanonicalSimplifier;
  // Borrow the owning analyzer for cross-backend queries.
  explicit RewriteSimplifier(AnalyzerObj* parent);
  // Exported for destruction through the inline Analyzer object allocator.
  TVM_TIRX_DLL ~RewriteSimplifier();
  // Copy durable facts/settings into this backend without sharing mutable query state.
  void CopyFrom(const RewriteSimplifier& other);
  class Impl;
  // Owned backend state; its definition stays inside the compiler library.
  ffi::ObjectPtr<Impl> impl_;
};

/*!
 * \brief Canonical-form based simplifier.
 */
class CanonicalSimplifier {
 public:
  /*!
   * \brief Simplify an expression under the current premises.
   * \param expr The expression to simplify.
   * \return An equivalent expression under those premises.
   */
  TVM_TIRX_DLL PrimExpr operator()(const PrimExpr& expr);

 private:
  friend class AnalyzerObj;
  // Clear context-derived facts before replay; retain query settings and diagnostics.
  void Reset();
  // Record a value binding after AnalyzerObj validates and orders the premises.
  void IngestBinding(const Var& var, const PrimExpr& expr);
  // Borrow the owning analyzer for cross-backend queries.
  explicit CanonicalSimplifier(AnalyzerObj* parent);
  // Exported for destruction through the inline Analyzer object allocator.
  TVM_TIRX_DLL ~CanonicalSimplifier();
  // Copy durable facts/settings into this backend without sharing mutable query state.
  void CopyFrom(const CanonicalSimplifier& other);
  class Impl;
  // Owned backend state; its definition stays inside the compiler library.
  ffi::ObjectPtr<Impl> impl_;
};

/*!
 * \brief Possible orderings of two expressions, represented as a bit set.
 *
 * EQ, LT, and GT are independent bits. Intersection combines facts that all
 * hold; union combines alternative possibilities. Zero indicates inconsistency.
 */
enum class CompareResult : int {
  /*! \brief No ordering is possible under the known facts. */
  kInconsistent = 0,
  /*! \brief Equal. */
  kEQ = 1,
  /*! \brief Strictly less. */
  kLT = 2,
  /*! \brief Less than or equal. */
  kLE = 3,
  /*! \brief Strictly greater. */
  kGT = 4,
  /*! \brief Greater than or equal. */
  kGE = 5,
  /*! \brief Unequal. */
  kNE = 6,
  /*! \brief Any ordering is possible. */
  kUnknown = 7
};

/*!
 * \brief Intersect the possible orderings.
 * \param lhs First set of orderings.
 * \param rhs Second set of orderings.
 * \return The combined set of orderings.
 */
inline constexpr CompareResult operator&(CompareResult lhs, CompareResult rhs) {
  return CompareResult(static_cast<int>(lhs) & static_cast<int>(rhs));
}
/*!
 * \brief Unite the possible orderings.
 * \param lhs First set of orderings.
 * \param rhs Second set of orderings.
 * \return The combined set of orderings.
 */
inline constexpr CompareResult operator|(CompareResult lhs, CompareResult rhs) {
  return CompareResult(static_cast<int>(lhs) | static_cast<int>(rhs));
}

/*!
 * \brief Integer set analyzer.
 */
class IntSetAnalyzer {
 public:
  /*!
   * \brief Find a symbolic integer set that contains all possible values of
   *        expr given the supplied variable domains.
   *
   * \param expr The expression of interest.
   * \param dom_map The variable domains to relax; variables absent from the map remain symbolic.
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL IntSet operator()(const PrimExpr& expr, const ffi::Map<Var, IntSet>& dom_map);

  /*!
   * \brief Find a symbolic integer set that contains all possible
   *        values of expr using current bindings, ranges, and recognized predicates.
   *
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL IntSet operator()(const PrimExpr& expr);

 private:
  friend class AnalyzerObj;
  // Clear context-derived facts before replay; retain query settings and diagnostics.
  void Reset();
  // Record a value binding after AnalyzerObj validates and orders the premises.
  void IngestBinding(const Var& var, const PrimExpr& expr);
  // Record a half-open variable range from the accepted premise prefix.
  void IngestBinding(const Var& var, const Range& range);
  // Consume a predicate prepared against the preceding premise prefix.
  void IngestPredicate(const PrimExpr& constraint);
  // Borrow the owning analyzer for cross-backend queries.
  explicit IntSetAnalyzer(AnalyzerObj* parent);
  // Exported for destruction through the inline Analyzer object allocator.
  TVM_TIRX_DLL ~IntSetAnalyzer();
  // Copy durable facts/settings into this backend without sharing mutable query state.
  void CopyFrom(const IntSetAnalyzer& other);
  class Impl;
  // Owned backend state; its definition stays inside the compiler library.
  Impl* impl_;
};

/*!
 * \brief Optional Z3-backed prover over the analyzer's accepted premises.
 *
 * Diagnostic and enumeration queries may propagate translation or solver errors.
 * Their temporary assertions do not change the accepted premise environment.
 */
class Z3Prover {
 public:
  /*!
   * \brief Whether the Z3 backend is compiled into this build (TVM_TIRX_USE_Z3=ON).
   *
   * \return true if the real Z3 prover is available, false for the stub.
   */
  TVM_TIRX_DLL bool IsEnabled() const;

  /*!
   * \brief Attempt to prove that expr holds under the current premises.
   * \param expr Boolean expression to prove.
   * \return True only when proved; false for unsupported expressions, an unavailable
   * backend, an inconclusive solver result, or a caught Z3 error. False is not a disproof.
   */
  TVM_TIRX_DLL bool CanProve(const PrimExpr& expr);

  /*!
   * \brief Get the SMTLIB2 representation of the current context.
   *
   * \param expr Optional boolean expression whose negation is added as a proof query.
   * \return The SMTLIB2 string, or a comment indicating that Z3 is disabled.
   */
  TVM_TIRX_DLL ffi::String GetSMTLIB2(const ffi::Optional<PrimExpr> expr);

  /*!
   * \brief Get statistics about Z3 prover.
   *
   * \return The statistics string, or a comment indicating that Z3 is disabled.
   */
  TVM_TIRX_DLL ffi::String GetStats();

  /*!
   * \brief Set timeout in milliseconds for Z3 prover.
   *
   * \param timeout_ms The timeout in milliseconds; zero disables the timeout. Ignored by the stub.
   */
  TVM_TIRX_DLL void SetTimeoutMs(unsigned timeout_ms);

  /*!
   * \brief Set resource limitation for Z3 prover.
   *
   * \param rlimit Z3 resource limit; zero disables it. Ignored by the stub.
   */
  TVM_TIRX_DLL void SetRLimit(unsigned rlimit);

  /*!
   * \brief Get a counterexample model for a boolean expression.
   *
   * \param expr Boolean expression whose negation is checked under the current premises.
   * \return A model satisfying the negation, or an empty string if no model is found
   * (including an unknown result). The stub returns a comment indicating Z3 is disabled.
   */
  TVM_TIRX_DLL ffi::String GetModel(const PrimExpr& expr);

  /*!
   * \brief Count the number of integer values that satisfy the current constraints.
   *
   * Enumerates distinct values until the limit, unsatisfiability, or an inconclusive
   * result. The count may be partial; it is not a proof of the domain cardinality.
   *
   * \param var The variable to count satisfying values for.
   * \param max_count Maximum number of values to enumerate; nonpositive values enumerate none.
   * \param min_consecutive Minimum size of each consecutive run in the enumerated values;
   * nonpositive values disable this check. The check does not establish complete coverage.
   * \return The enumerated count, -1 for an unsupported variable or disabled Z3, or -2
   * if a nonempty enumerated run is shorter than min_consecutive. Errors may propagate.
   */
  TVM_TIRX_DLL int64_t CountSatisfyingValues(const Var& var, int64_t max_count = 2048,
                                             int64_t min_consecutive = 1);

 private:
  friend class AnalyzerObj;
  // Clear facts, translations, and solver state; retain resource limits.
  void Reset();
  // Record a value binding after AnalyzerObj validates and orders the premises.
  void IngestBinding(const Var& var, const PrimExpr& expr);
  // Record a half-open variable range from the accepted premise prefix.
  void IngestBinding(const Var& var, const Range& range);
  // Consume a predicate prepared against the preceding premise prefix.
  void IngestPredicate(const PrimExpr& constraint);
  friend class Analyzer;
  // Borrow the owning analyzer for cross-backend queries.
  explicit Z3Prover(AnalyzerObj* parent);
  // Exported for destruction through the inline Analyzer object allocator.
  TVM_TIRX_DLL ~Z3Prover();
  // Copy durable facts/settings into this backend without sharing mutable query state.
  void CopyFrom(const Z3Prover& other);
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/*!
 * \brief Coordinated symbolic analyses over an explicitly synchronized premise environment.
 *
 * Queries may update caches, solver state, and statistics, but do not append
 * persistent premises. Proof methods are conservative: false means not proved.
 * Sub-analyzers borrow their owner and must not outlive it. This mutable object
 * must not be queried or synchronized concurrently from multiple threads.
 */
class TVM_TIRX_DLL AnalyzerObj : public ffi::Object {
 public:
  /*! \brief sub-analyzer: const integer bound */
  ConstIntBoundAnalyzer const_int_bound;
  /*! \brief sub-analyzer: modular set */
  ModularSetAnalyzer modular_set;
  /*! \brief sub-analyzer rewrite simplify */
  RewriteSimplifier rewrite_simplify;
  /*! \brief sub-analyzer canonical simplify */
  CanonicalSimplifier canonical_simplify;
  /*! \brief sub-analyzer: int set */
  IntSetAnalyzer int_set;
  /*! \brief sub-analyzer using Z3 */
  Z3Prover z3_prover;
  /*! \brief Construct an analyzer with no accepted premises and default query settings. */
  AnalyzerObj();
  /*!
   * \brief Set whether DEFAULT-strength queries try Z3 after native analysis fails.
   *
   * Disabled by default. This setting is local to the analyzer and preserved by Clone.
   * Stronger proof queries and direct Z3 APIs are unaffected. Disabling the fallback
   * does not clear accepted premises or existing solver state.
   * \param enabled Whether to allow the DEFAULT-strength Z3 fallback.
   */
  void SetZ3DefaultFallback(bool enabled);
  /*!
   * \brief Synchronize all derived facts to the supplied premise environment.
   *
   * Unchanged positions need no replay; recognized additive prefixes update
   * incrementally, and other transitions clear and replay all backends. Query
   * settings and cumulative rewrite counters survive synchronization.
   *
   * The context must bind each Var identity at most once across value and range
   * bindings. Unsupported or effectful premises may be ignored conservatively.
   * Captured constraint and expression nodes must not be mutated through aliases.
   * Appending to the caller's context after this call takes effect only after a
   * later synchronization. Synchronization during a query or another synchronization
   * is rejected. If ingestion throws, discard this analyzer: partially applied
   * backend state is not rolled back.
   *
   * \param context Ordered premises to snapshot and accept.
   * \throws ffi::Error If bindings repeat a Var identity, synchronization is
   * reentrant, or an analyzer invariant fails. Backend ingestion errors may also propagate.
   */
  void SyncConstraintContext(const ConstraintContext& context);
  /*!
   * \brief Snapshot the accepted premise environment.
   * \return A context that can be extended independently. Constraint and expression
   * nodes are shared and must not be mutated through aliases. Appending to this
   * context does not update the analyzer until SyncConstraintContext is called.
   */
  ConstraintContext GetConstraintContext() const;
  /*!
   * \brief Attempt to prove expr >= lower_bound.
   *
   * Bounds help distinguish truncating and floor division/remainder semantics.
   *
   * \param expr The expression.
   * \param lower_bound The lower bound.
   * \return True if proved; false if unknown or false.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveGreaterEqual(const PrimExpr& expr, int64_t lower_bound);
  /*!
   * \brief Attempt to prove expr < upper_bound.
   *
   * Bounds help distinguish truncating and floor division/remainder semantics.
   *
   * \param expr The expression.
   * \param upper_bound The upper bound.
   * \return True if proved; false if unknown or false.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveLess(const PrimExpr& expr, int64_t upper_bound);
  /*!
   * \brief Attempt to prove lhs == rhs.
   *
   * \param lhs The input lhs.
   * \param rhs The input rhs.
   * \return True if equality is proved; false does not establish inequality.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs);
  /*!
   * \brief Attempt to prove lhs <= a possibly symbolic shape.
   *
   * By calling this function, the caller gives an extra hint that shape > 0,
   * because it appeared in buffer shape.
   *
   * This is useful to prove condition such as 32 <= 32 * n where the 32 * n
   * is known to be a shape. Use this routine to reduce the symbolic comparisons
   * in buffer compaction.
   *
   * The underlying analyzer will use the kSymbolicBound proof.
   *
   * \param lhs The input lhs.
   * \param shape The symbolic shape.
   * \return True if proved using the caller's positive-shape guarantee; false otherwise.
   */
  bool CanProveLessEqualThanSymbolicShapeValue(const PrimExpr& lhs, const PrimExpr& shape);
  /*!
   * \brief Attempt to prove a condition under the current premises.
   *
   * \param cond The expression to be proved.
   * \param strength Proof effort; kSymbolicBound additionally enables symbolic bound analysis.
   *
   * \return True if proved; false includes unsupported or inconclusive queries.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   * Do not use strength beyond default in sub-analyzers and
   * only use it in top-level predicate analysis.
   * When built with Z3, explicit stronger queries try it after native analysis fails.
   * DEFAULT-strength queries only try Z3 when SetZ3DefaultFallback(true) enables it.
   * Z3's existing resource limit and timeout apply; unknown means cannot prove.
   */
  bool CanProve(const PrimExpr& cond, ProofStrength strength = ProofStrength::kDefault);

  /*!
   * \brief Simplify expr.
   *
   * \param expr The expression to be simplified.
   * \param steps Number of alternating rewrite/canonical passes after an initial
   * canonical pass. The default of two runs canonical, rewrite, then canonical.
   * Nonpositive values run only the initial canonical pass. Processing may stop
   * early at a constant expression.
   * \return An equivalent expression under the current premises.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  PrimExpr Simplify(const PrimExpr& expr, int steps = 2);

  /*!
   * \brief Copy accepted facts and query settings into an independent analyzer.
   *
   * Backend state is independent: subsequent synchronization, simplification,
   * and query settings on either analyzer do not affect the other. Premise and
   * expression nodes are shared and must not be mutated through aliases.
   * Query scratch, solver instances, and accumulated statistics are not copied.
   * Copying an Analyzer handle alone instead shares the same mutable state.
   *
   * \return A new analyzer with independent backend state and the same accepted premises.
   */
  Analyzer Clone() const;

  /*!
   * \brief Analyzer methods update facts, constraints, caches, and stats.
   *
   * Marking the object mutable makes the `Analyzer` ObjectRef expose a
   * non-const `operator->`, so APIs can take `const Analyzer&` while still
   * allowing calls such as `analyzer->SyncConstraintContext(...)`.
   * `const Analyzer&` keeps the handle itself from being rebound; it does
   * not make the underlying AnalyzerObj immutable.
   */
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.Analyzer", AnalyzerObj, ffi::Object);

 private:
  friend class detail::TempConstraintScope;
  // Policy for DEFAULT-strength queries; explicit stronger queries can still use Z3.
  bool z3_default_fallback_{false};
  // Record a value binding after AnalyzerObj validates and orders the premises.
  void IngestBinding(const Var& var, const PrimExpr& expr);
  // Record a half-open variable range from the accepted premise prefix.
  void IngestBinding(const Var& var, const Range& range);
  // Dispatch a supported pure premise to the participating backends.
  void IngestConstraint(const Constraint& constraint);
  // Normalize and derive predicate facts using only the already valid prefix.
  std::vector<PrimExpr> PreparePredicate(const PrimExpr& condition);
  // Accepted premise prefix, advanced after each fact during synchronization.
  ConstraintContext last_snapshot_;
  // Query-local premises, managed in LIFO order by TempConstraintScope.
  std::vector<Constraint> temp_constraints_;
  // Prevent reentry and reuse after an incomplete synchronization.
  bool synchronizing_{false};
  // Nested query count; synchronization requires zero active queries.
  int active_queries_{0};
};

/*!
 * \brief Managed reference to AnalyzerObj.
 *
 * Analyzer is a lightweight, reference-counted handle around a heap-allocated
 * AnalyzerObj. Because it is now a first-class FFI object, an Analyzer can be
 * passed across the tvm-ffi boundary (e.g. handed from Python into a C++ pass)
 * and shared, so that accumulated bindings/constraints persist across calls.
 * Copying an Analyzer copies the handle, and both handles share the same
 * mutable AnalyzerObj state.
 * This is not a deep copy of analyzer facts or caches.
 *
 * \sa AnalyzerObj
 */
class Analyzer : public ffi::ObjectRef {
 public:
  /*! \brief Construct a fresh analyzer. */
  Analyzer() : Analyzer(ffi::make_object<AnalyzerObj>()) {}
  /*! \brief Adopt a reference to a non-null analyzer object. */
  explicit Analyzer(ffi::ObjectPtr<AnalyzerObj> n) : ffi::ObjectRef(std::move(n)) {
    TVM_FFI_ICHECK(this->get() != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Analyzer, ffi::ObjectRef, AnalyzerObj);
};

/*!
 * \brief Scoped no-overflow assumption for unsigned index arithmetic.
 *
 * Unsigned arithmetic normally wraps modulo 2^bits. Use
 * With<AllowUIntAsIndex> only when unsigned values are guaranteed not to
 * overflow, enabling the analyzer's index rewrites within that scope.
 * The assumption is off by default, thread-local, and supports nested scopes.
 * Constructing this context alone does not enable the assumption.
 */
class AllowUIntAsIndex {
 public:
  /*!
   * \brief Check the current thread's unsigned-index assumption.
   * \return True inside an active With<AllowUIntAsIndex> scope.
   */
  TVM_TIRX_DLL static bool Enabled();

 private:
  friend class With<AllowUIntAsIndex>;
  // Called across the DLL boundary by the inline With<> constructor.
  TVM_TIRX_DLL void EnterWithScope();
  // Called across the DLL boundary by the inline With<> destructor.
  TVM_TIRX_DLL void ExitWithScope();
  // Number of active scopes in this thread.
  static thread_local int depth_;
};

}  // namespace sym
}  // namespace tvm
#endif  // TVM_SYM_ANALYZER_H_
