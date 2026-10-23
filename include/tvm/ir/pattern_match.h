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
 * \file tvm/ir/pattern_match.h
 * \brief Shared expression-template pattern and capture protocol.
 *
 * \rst
 * .. code-block:: cpp
 *
 *   #include <tvm/ir/prim/pattern_match.h>
 *
 *   void MatchMultiplyAdd(const tvm::PrimExpr& expr) {
 *     tvm::PVar<tvm::PrimExpr> a, b, c;
 *     if ((a * b + c).Match(expr)) {
 *       tvm::PrimExpr lhs = a.Eval();
 *       tvm::PrimExpr rhs = b.Eval();
 *       tvm::PrimExpr addend = c.Eval();
 *     }
 *   }
 * \endrst
 */
#ifndef TVM_IR_PATTERN_MATCH_H_
#define TVM_IR_PATTERN_MATCH_H_

#include <tvm/ffi/cast.h>

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tvm {
/*!
 * \brief Base class of all the patterns.
 *
 * There are two major member functions supported by each pattern.
 * - Match: checks if value matches the pattern.
 * - Eval: construct a new value based on matched values in PVar.
 *
 * We use curiously recurring template pattern to construct
 * expression templates.
 *
 * \tparam Derived The type of the derived class.
 */
template <typename Derived>
class Pattern {
 public:
  /*!
   * \brief Nested storage type in the expression.
   *
   *  Depending on the Derived class,
   *  Nested can be Derived (nest by value) or
   *  const Derived& (nest by reference).
   *
   *  The trick of Nested typedef originates from Eigen.
   *
   * \note We use nest by value for intermediate expressions,
   *       and nest by reference for PVars.
   */
  using Nested = Derived;

  /*!
   * \brief Check if value matches the current pattern.
   *
   * This call also populates the PVars with matched value.
   * The values in PVars are valid until the next call to Match.
   *
   * \param value The value to be matched against
   *
   * \return whether value matches the pattern.
   */
  template <typename NodeType>
  inline bool Match(const NodeType& value) const {
    return Match(value, []() { return true; });
  }

  /*!
   * \brief Check if value matches the current pattern.
   *
   * This call also populates the PVars with matched value.
   * The values in PVars are valid until the next call to Match.
   *
   * \param value The value to be matched against
   *
   * \param cond A callable that performs additional validation,
   * returning true if the match passes.  This will typically be a
   * lambda function written in terms of the filled PVars.
   *
   * \return whether value matches the pattern.
   */
  template <typename NodeType, typename Condition>
  bool Match(const NodeType& value, Condition cond) const {
    derived().InitMatch_();
    return derived().Match_(value) && cond();
  }
  /*! \return Derived instance of current class. */
  const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

/*!
 * \brief Default deep equality checker
 * \tparam T the comparison point.
 */
template <typename T>
class PEqualChecker {
 public:
  bool operator()(const T& lhs, const T& rhs) const { return lhs == rhs; }
};

/*!
 * \brief Pattern variable container.
 *
 * PVar is used as a "hole" in the pattern that can be matched.
 *
 * \tparam T the type of the hole.
 *
 * \note PVar is not thread safe.
 *       Do not use the same PVar in multiple threads.
 */
template <typename T>
class PVar : public Pattern<PVar<T>> {
 public:
  // Store PVars by reference in the expression.
  using Nested = const PVar<T>&;

  void InitMatch_() const { value_.reset(); }

  bool Match_(const T& value) const {
    if (!value_) {
      value_ = value;
      return true;
    } else {
      return PEqualChecker<T>()(*value_, value);
    }
  }

  template <typename NodeRefType, typename = typename std::enable_if<
                                      !std::is_same<NodeRefType, T>::value &&
                                      std::is_base_of<ffi::ObjectRef, NodeRefType>::value &&
                                      std::is_base_of<ffi::ObjectRef, T>::value>::type>
  bool Match_(const NodeRefType& value) const {
    if (auto typed_value = value.template as<T>()) {
      return Match_(*typed_value);
    } else {
      return false;
    }
  }

  T Eval() const {
    TVM_FFI_ICHECK(value_.has_value());
    return *value_;
  }

  T EvalOr(const T& default_value) const { return value_.value_or(default_value); }

 protected:
  /*! \brief The matched value */
  mutable std::optional<T> value_;
};

/*!
 * \brief Wrapper for pattern variable container with extra match logic.
 *
 * \tparam Derived the type of derived class.
 * \tparam T the type of the hole.
 */
template <typename Derived, typename T>
class PVarWithCheck : public Pattern<PVarWithCheck<Derived, T>> {
 public:
  // Store by reference in the expression.
  using Nested = const PVarWithCheck<Derived, T>&;

  void InitMatch_() const { pvar_.InitMatch_(); }

  bool Match_(const T& value) const {
    if (!static_cast<const Derived*>(this)->Match_(value)) return false;
    return pvar_.Match_(value);
  }

  template <typename NodeRefType,
            typename = typename std::enable_if<std::is_base_of<NodeRefType, T>::value>::type>
  bool Match_(const NodeRefType& value) const {
    if (auto typed_value = value.template as<T>()) {
      return Match_(*typed_value);
    } else {
      return false;
    }
  }

  T Eval() const { return pvar_.Eval(); }

 protected:
  PVar<T> pvar_;
};

/*!
 * \brief Constant Pattern variable container.
 *
 * \tparam T the type of the hole.
 */
template <typename T>
class PConst : public Pattern<PConst<T>> {
 public:
  PConst(T value)  // NOLINT(*)
      : value_(value) {}

  void InitMatch_() const {}

  bool Match_(const T& value) const { return PEqualChecker<T>()(value_, value); }

  T Eval() const { return value_; }

 private:
  const T value_;
};

template <typename... TPattern>
class PMatchesOneOf {
 public:
  explicit PMatchesOneOf(const TPattern&... patterns) : patterns_{patterns...} {}

  /*! \brief Check if value matches one of the patterns.
   *
   * This call also populates the PVars with matched value based on
   * the first successful match.  The values in PVars are valid until
   * the next call to Match.
   *
   * \param value The value to be matched against.
   *
   * \return Whether value matches the pattern.
   */
  template <typename NodeType>
  inline bool Match(const NodeType& value) const {
    return Match(value, []() { return true; });
  }

  /*! \brief Check if value matches one of the patterns.
   *
   * This call also populates the PVars with matched value based on
   * the first successful match.  The values in PVars are valid until
   * the next call to Match.
   *
   * \param value The value to be matched against.
   *
   * \param cond A callable that performs additional validation,
   * returning true if the match passes.  This will typically be a
   * lambda function written in terms of the filled PVars.  This will
   * be called once for each successful pattern match.  If `cond()`
   * returns false, the next match will be attempted.
   *
   * \return Whether value matches the pattern.
   */
  template <typename NodeType, typename Condition>
  inline bool Match(const NodeType& value, Condition cond) const {
    return MatchImpl(value, cond, std::make_index_sequence<sizeof...(TPattern)>());
  }

 private:
  template <typename NodeType, typename Condition>
  inline bool MatchImpl(const NodeType& value, Condition cond, std::index_sequence<>) const {
    return false;
  }

  template <typename NodeType, typename Condition, size_t FirstIndex, size_t... RemainingIndices>
  inline bool MatchImpl(const NodeType& value, Condition cond,
                        std::index_sequence<FirstIndex, RemainingIndices...>) const {
    return std::get<FirstIndex>(patterns_).Match(value, cond) ||
           MatchImpl(value, cond, std::index_sequence<RemainingIndices...>());
  }

  // Hold the patterns by const&.  This follows the same usage as both
  // the `PVar`, which occurs as `const PVar<T>&` when it appears
  // inside other patterns.  Because the `PVar<T>::value_` field is
  // mutable, it can still be updated through these const references.
  // So long as the call to `Match()` occurs within the same
  // expression as created the patterns, this avoids accidental copies
  // without creating dangling references.  This may be improved in
  // the future by use of `constexpr` constructors/operators, allowing
  // more typical value semantics.
  std::tuple<const TPattern&...> patterns_;
};

/* \brief Return a proxy object that returns true after the first match
 *
 * In the RewriteSimplifier, there are often several expressions that
 * simplify to the same resulting expression.  This utility allows
 * them to be specified as a single rule, reducing duplication of the
 * result/condition of a rewrite.
 */
template <typename... TPattern>
inline std::enable_if_t<(std::is_base_of_v<Pattern<TPattern>, TPattern> && ... && true),
                        PMatchesOneOf<TPattern...>>
MatchesOneOf(const TPattern&... patterns) {
  return PMatchesOneOf<TPattern...>(patterns...);
}
}  // namespace tvm
#endif  // TVM_IR_PATTERN_MATCH_H_
