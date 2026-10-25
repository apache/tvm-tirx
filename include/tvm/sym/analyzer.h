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
namespace detail {
class TempConstraintScope;
}  // namespace detail

enum DivMode {
  /*! \brief Truncated division. */
  kTruncDiv,
  /*! \brief Floor division. */
  kFloorDiv
};

/*!
 * \brief The strength used in top-level condition proves
 * \note The higher, the more time consuming it can be.
 *
 * Do not use level beyond kDefault in internal recursive rewriting in sym
 * analysis and only use it at top-level simplification to avoid speed issues.
 */
enum class ProofStrength : int {
  /*! \brief default strength, can be used in. */
  kDefault = 0,
  /*!
   * \brief Prove using symbolic bound analysis
   */
  kSymbolicBound = 1,
};

/*!
 * \brief Constant integer up and lower bound(inclusive).
 *  Useful for value bound analysis.
 *
 *  set = [min_value, max_value]
 */
class ConstIntBoundObj : public ffi::Object {
 public:
  int64_t min_value;
  int64_t max_value;

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
   * \note We can make use the of fact that -kPosInf == kNegInf in the project.
   */
  static const constexpr int64_t kNegInf = -kPosInf;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.ConstIntBound", ConstIntBoundObj, ffi::Object);
};

/*!
 * \brief reference class to ConstIntBoundObj
 * \sa ConstIntBoundObj
 */
class ConstIntBound : public ffi::ObjectRef {
 public:
  /*!
   * \brief constructor by fields.
   * \param min_value The mininum value.
   * \param max_value The maximum value.
   */
  TVM_TIRX_DLL ConstIntBound(int64_t min_value, int64_t max_value);

  static const constexpr int64_t kPosInf = ConstIntBoundObj::kPosInf;
  static const constexpr int64_t kNegInf = ConstIntBoundObj::kNegInf;
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConstIntBound, ffi::ObjectRef, ConstIntBoundObj);
};

/*!
 * \brief Analyzer to get constant integer bound over expression.
 */
class ConstIntBoundAnalyzer {
 public:
  using BoundMapType =
      std::unordered_map<PrimExpr, ConstIntBound, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL ConstIntBound operator()(const PrimExpr& expr) const;

  /*!
   * \brief analyze the expr with the intermediate memorized to avoid redundant computation
   * \param expr The expression of interest.
   * \param bound Output map, cleared on entry so results cannot cross context epochs
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL ConstIntBound operator()(const PrimExpr& expr, BoundMapType* bound);

  /*!
   * \brief Check if a variable is bound to a range.
   * \param var The variable.
   * \return Whether the variable is bound to a range.
   */
  TVM_TIRX_DLL bool IsBound(const Var& var) const;

 private:
  friend class AnalyzerObj;
  TVM_TIRX_DLL void Reset();
  TVM_TIRX_DLL void IngestBinding(const Var& var, const PrimExpr& expr);
  TVM_TIRX_DLL void IngestBinding(const Var& var, const Range& range);
  TVM_TIRX_DLL void IngestPredicate(const PrimExpr& constraint);
  explicit ConstIntBoundAnalyzer(AnalyzerObj* parent);
  TVM_TIRX_DLL ~ConstIntBoundAnalyzer();
  void CopyFrom(const ConstIntBoundAnalyzer& other);
  struct Entry;
  class Impl;
  /*! \brief Internal impl */
  std::unique_ptr<Impl> impl_;
};

/*!
 * \brief Range of a linear integer function.
 *  Use to do specify the possible index values.
 *
 *  set = { coeff * x + base | x in Z }
 *
 *  When coeff != 0, it can also be written as
 *  set = { n | n % coeff == base }
 *
 *  This is useful to decide if the index is dividable by certain value.
 *  For example, if index = 0 + 4 x, then we know it can be divided by 4.
 */
class ModularSetObj : public ffi::Object {
 public:
  /*! \brief linear co-efficient */
  int64_t coeff;
  /*! \brief The base */
  int64_t base;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModularSetObj>()
        .def_ro("coeff", &ModularSetObj::coeff)
        .def_ro("base", &ModularSetObj::base);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.ModularSet", ModularSetObj, ffi::Object);
};

/*!
 * \brief reference of ModularSetObj
 * \sa ModularSetObj
 */
class ModularSet : public ffi::ObjectRef {
 public:
  TVM_TIRX_DLL ModularSet(int64_t coeff, int64_t base);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ModularSet, ffi::ObjectRef, ModularSetObj);
};

/*!
 * \brief Analyzer to get modular information over expression.
 */
class ModularSetAnalyzer {
 public:
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL ModularSet operator()(const PrimExpr& expr);

 private:
  friend class AnalyzerObj;
  TVM_TIRX_DLL void Reset();
  TVM_TIRX_DLL void IngestBinding(const Var& var, const PrimExpr& expr);
  TVM_TIRX_DLL void IngestPredicate(const PrimExpr& constraint);
  explicit ModularSetAnalyzer(AnalyzerObj* parent);
  TVM_TIRX_DLL ~ModularSetAnalyzer();
  void CopyFrom(const ModularSetAnalyzer& other);
  struct Entry;
  class Impl;
  /*! \brief Internal impl */
  std::unique_ptr<Impl> impl_;
};

/*!
 * \brief Rewrite-rule based simplifier.
 */
class RewriteSimplifier {
 public:
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL PrimExpr operator()(const PrimExpr& expr);

  /*! \brief Return the statistics counters */
  TVM_TIRX_DLL ffi::ObjectRef GetStatsCounters() const;

  /*! \brief Reset the statistics counters */
  TVM_TIRX_DLL void ResetStatsCounters();

  /*! \brief Set the maximum allowed number of rewrite steps
   *
   * By default, the simplifier may perform as many steps as are
   * required.  If a positive limit is set, then the simplifier will
   * throw an exception when exceeding that number of rewrite steps.
   * This allows tests to guard against performance regressions.
   *
   * Note: To maintain accurate usage counters, `Analyzer` instances
   * should be re-used wherever possible.  For example, TIR
   * transformations should declare a single `Analyzer` that is used
   * throughout the pass.  Internal helper functions that only borrow
   * the analyzer temporarily may receive the underlying `AnalyzerObj*`
   * from their calling scope.
   */
  TVM_TIRX_DLL void SetMaximumRewriteSteps(int64_t maximum);

 private:
  friend class AnalyzerObj;
  TVM_TIRX_DLL void Reset();
  TVM_TIRX_DLL void IngestBinding(const Var& var, const PrimExpr& expr);
  TVM_TIRX_DLL void IngestPredicate(const PrimExpr& constraint);
  friend class CanonicalSimplifier;
  explicit RewriteSimplifier(AnalyzerObj* parent);
  TVM_TIRX_DLL ~RewriteSimplifier();
  void CopyFrom(const RewriteSimplifier& other);
  class Impl;
  /*! \brief Internal impl */
  ffi::ObjectPtr<Impl> impl_;
};

/*!
 * \brief Canonical-form based simplifier.
 */
class CanonicalSimplifier {
 public:
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL PrimExpr operator()(const PrimExpr& expr);

 private:
  friend class AnalyzerObj;
  TVM_TIRX_DLL void Reset();
  TVM_TIRX_DLL void IngestBinding(const Var& var, const PrimExpr& expr);
  explicit CanonicalSimplifier(AnalyzerObj* parent);
  TVM_TIRX_DLL ~CanonicalSimplifier();
  void CopyFrom(const CanonicalSimplifier& other);
  class Impl;
  /*! \brief Internal impl */
  ffi::ObjectPtr<Impl> impl_;
};

/*! \brief Structure for representing result of known
 *
 * Values are assigned to allow these flags to be used in bitwise
 * operations.
 */
enum class CompareResult : int {
  kInconsistent = 0,
  kEQ = 1,
  kLT = 2,
  kLE = 3,
  kGT = 4,
  kGE = 5,
  kNE = 6,
  kUnknown = 7
};

inline constexpr CompareResult operator&(CompareResult lhs, CompareResult rhs) {
  return CompareResult(static_cast<int>(lhs) & static_cast<int>(rhs));
}
inline constexpr CompareResult operator|(CompareResult lhs, CompareResult rhs) {
  return CompareResult(static_cast<int>(lhs) | static_cast<int>(rhs));
}

/*!
 * \brief Using previously specified knowns, compare the expressions provided
 *
 * Compare directly related expressions, allowing constant offsets.
 */
class KnownComparisonAnalyzer {
 public:
  /* \brief Using previously specified knowns, compare the expressions provided
   *
   * \param lhs The left-hand side of the comparison
   *
   * \param rhs The right-hand side of the comparison
   *
   * Directly known comparisons also allow constant offsets on either side.
   *
   * \return The most specific result that can be proven about the
   * comparison.  If nothing can be proven, returns kUnknown.
   */
  TVM_TIRX_DLL CompareResult TryCompare(const PrimExpr& lhs, const PrimExpr& rhs);

 private:
  friend class AnalyzerObj;
  TVM_TIRX_DLL void Reset();
  TVM_TIRX_DLL void IngestBinding(const Var& var, const PrimExpr& expr);
  TVM_TIRX_DLL void IngestBinding(const Var& var, const Range& range);
  TVM_TIRX_DLL void IngestPredicate(const PrimExpr& constraint);
  explicit KnownComparisonAnalyzer(AnalyzerObj* parent);
  TVM_TIRX_DLL ~KnownComparisonAnalyzer();
  void CopyFrom(const KnownComparisonAnalyzer& other);
  class Impl;
  /*! \brief Internal impl */
  std::unique_ptr<Impl> impl_;
};

/*!
 * \brief Integer set analyzer.
 */
class IntSetAnalyzer {
 public:
  /*!
   * \brief Find a symbolic integer set that contains all possible values of
   *        expr given the domain of each variables.
   *
   * \param expr The expression of interest.
   * \param dom_map The domain map to indicate which variable to relax.
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL IntSet operator()(const PrimExpr& expr, const ffi::Map<Var, IntSet>& dom_map);

  /*!
   * \brief Find a symbolic integer set that contains all possible
   *        values of expr given the domain of each variables, using
   *        the domain map defined by bound variables.
   *
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_TIRX_DLL IntSet operator()(const PrimExpr& expr);

 private:
  friend class AnalyzerObj;
  TVM_TIRX_DLL void Reset();
  TVM_TIRX_DLL void IngestBinding(const Var& var, const PrimExpr& expr);
  TVM_TIRX_DLL void IngestBinding(const Var& var, const Range& range);
  TVM_TIRX_DLL void IngestPredicate(const PrimExpr& constraint);
  explicit IntSetAnalyzer(AnalyzerObj* parent);
  TVM_TIRX_DLL ~IntSetAnalyzer();
  void CopyFrom(const IntSetAnalyzer& other);
  class Impl;
  /*! \brief Internal impl */
  Impl* impl_;
};

class Z3Prover {
 public:
  /*!
   * \brief Whether the Z3 backend is compiled into this build (USE_Z3=ON).
   *
   * \return true if the real Z3 prover is available, false for the stub.
   */
  TVM_TIRX_DLL bool IsEnabled() const;

  /*!
   * \brief Whether can we prove expr is always true.
   *
   * \param expr The expression.
   * \return Whether we can prove it.
   */
  TVM_TIRX_DLL bool CanProve(const PrimExpr& expr);

  /*!
   * \brief Get the SMTLIB2 representation of the current context.
   *
   * \param expr The optional expression to check.
   * \return The SMTLIB2 string.
   */
  TVM_TIRX_DLL ffi::String GetSMTLIB2(const ffi::Optional<PrimExpr> expr);

  /*!
   * \brief Get statistics about Z3 prover.
   *
   * \return The statistics string.
   */
  TVM_TIRX_DLL ffi::String GetStats();

  /*!
   * \brief Set timeout in milliseconds for Z3 prover.
   *
   * \param timeout_ms The timeout in milliseconds.
   */
  TVM_TIRX_DLL void SetTimeoutMs(unsigned timeout_ms);

  /*!
   * \brief Set resource limitation for Z3 prover.
   *
   * \param rlimit the resource limitation.
   */
  TVM_TIRX_DLL void SetRLimit(unsigned rlimit);

  /*!
   * \brief Get the Z3 model for the given expression if satisfiable.
   *
   * \param expr The expression to get the model for.
   * \return The model as a string.
   */
  TVM_TIRX_DLL ffi::String GetModel(const PrimExpr& expr);

  /*!
   * \brief Count the number of integer values that satisfy the current constraints.
   *
   * This method uses Z3's model enumeration to count how many distinct values of
   * the given variable satisfy all current constraints.
   *
   * \param var The variable to count satisfying values for.
   * \param max_count Maximum number of solutions to enumerate.
   * \param min_consecutive Minimum consecutive count requirement.
   * \return The number of distinct values that satisfy the constraints, or a negative error code.
   */
  TVM_TIRX_DLL int64_t CountSatisfyingValues(const Var& var, int64_t max_count = 2048,
                                             int64_t min_consecutive = 1);

 private:
  friend class AnalyzerObj;
  TVM_TIRX_DLL void Reset();
  TVM_TIRX_DLL void IngestBinding(const Var& var, const PrimExpr& expr);
  TVM_TIRX_DLL void IngestBinding(const Var& var, const Range& range);
  TVM_TIRX_DLL void IngestPredicate(const PrimExpr& constraint);
  friend class Analyzer;
  explicit Z3Prover(AnalyzerObj* parent);
  TVM_TIRX_DLL ~Z3Prover();
  void CopyFrom(const Z3Prover& other);
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/*!
 * \brief Analyzer that contains bunch of sub-analyzers.
 *
 * Each sub-analyzer can make use of another sub-analyzer
 * by weak reference of this.
 *
 * NOTE for sub-analyzer developers:
 * If the analyzer uses memoization, we need to clear the internal
 * cache when information about a Var has been overridden.
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
  /*! \brief sub-analyzer known comparisons */
  KnownComparisonAnalyzer known_comparisons;
  /*! \brief sub-analyzer using Z3 */
  Z3Prover z3_prover;
  /*! \brief constructor */
  AnalyzerObj();
  /*! \brief Synchronize all derived facts to the supplied premise environment.
   * Unchanged positions are free; additive prefixes update incrementally and
   * other transitions clear and replay. Do not mutate captured premises through aliases.
   */
  void SyncConstraintContext(const ConstraintContext& context);
  /*! \brief Return an independently extensible snapshot of the accepted premises. */
  ConstraintContext GetConstraintContext() const;
  /*!
   * \brief Whether can we prove expr >= val.

   *  Non-negative proof is very useful in integer analysis
   *  to lower divisions and mods given difference in trunc and ceil mode.
   *
   * \param expr The expression.
   * \param lower_bound The lower bound.
   * \return Whether we can prove it.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveGreaterEqual(const PrimExpr& expr, int64_t lower_bound);
  /*!
   * \brief Whether can we prove expr < val.

   *  Non-negative proof is very useful in integer analysis
   *  to lower divisions and mods given difference in trunc and ceil mode.
   *
   * \param expr The expression.
   * \param upper_bound The upper bound.
   * \return Whether we can prove it.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveLess(const PrimExpr& expr, int64_t upper_bound);
  /*!
   * \brief Whether can we prove lhs == rhs.
   *
   * \param lhs The input lhs.
   * \param rhs The input rhs.
   * \return Whether we can prove lhs == rhs.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs);
  /*!
   * \brief Whether we can prove lhs is smaller than possibly symbolic shape.
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
   * \return Whether we can prove lhs <= shape.
   */
  bool CanProveLessEqualThanSymbolicShapeValue(const PrimExpr& lhs, const PrimExpr& shape);
  /*!
   * \brief Whether can we prove condition.
   *
   * \param cond The expression to be proved.
   * \param strength the strength of the prove.
   *
   * \return The result.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   * Do not use strength beyond default in sub-analyzers and
   * only use it in top-level predicate analysis.
   */
  bool CanProve(const PrimExpr& cond, ProofStrength strength = ProofStrength::kDefault);

  /*!
   * \brief Simplify expr.
   *
   * \param expr The expression to be simplified.
   * \param steps The simplification runs in the order of
   *        rewrite_simplify (step 1) -> canonical_simplify (step 2) ->
   *        rewrite_simplify (step 3) -> canonical_simplify (step 4) -> ...
   *        param steps controls how many steps to run.
   *        Default is 2, i.e., rewrite_simplify + canonical_simplify.
   * \return The result.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  PrimExpr Simplify(const PrimExpr& expr, int steps = 2);

  /*!
   * \brief Deep-copy this analyzer into a new, independent Analyzer.
   *
   * The returned analyzer carries the same accumulated facts (variable
   * bounds, modular sets, rewrite/canonical bindings, integer-set domains,
   * literal constraints and known comparisons) as this one, but owns
   * its own state: binding or simplifying on either analyzer afterwards does
   * not affect the other. This is the deep copy that handle-copying an
   * Analyzer does not provide.
   *
   * \return A new Analyzer holding an independent copy of the facts.
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
  void IngestBinding(const Var& var, const PrimExpr& expr);
  void IngestBinding(const Var& var, const Range& range);
  void IngestConstraint(const Constraint& constraint);
  std::vector<PrimExpr> PreparePredicate(const PrimExpr& condition);
  ConstraintContext last_snapshot_;
  std::vector<Constraint> temp_constraints_;
  bool synchronizing_{false};
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
  /*! \brief Default-construct a fresh analyzer (allocates an AnalyzerObj). */
  Analyzer() : Analyzer(ffi::make_object<AnalyzerObj>()) {}
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
  /*! \brief Whether unsigned index analysis is enabled in the current thread. */
  TVM_TIRX_DLL static bool Enabled();

 private:
  friend class With<AllowUIntAsIndex>;
  TVM_TIRX_DLL void EnterWithScope();
  TVM_TIRX_DLL void ExitWithScope();
  static thread_local int depth_;
};

}  // namespace sym
}  // namespace tvm
#endif  // TVM_SYM_ANALYZER_H_
