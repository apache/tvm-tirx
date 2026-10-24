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
 * \file tvm/sym/constraint.h
 * \brief Data-only symbolic premises collected during traversal.
 */
#ifndef TVM_SYM_CONSTRAINT_H_
#define TVM_SYM_CONSTRAINT_H_

#include <tvm/ffi/container/list.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/expr.h>

#include <utility>

namespace tvm {
namespace sym {

/*!
 * \brief Base node for a symbolic premise; it owns no analyzer state.
 */
class ConstraintObj : public ffi::Object {
 public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("sym.Constraint", ConstraintObj, ffi::Object);
};

/*!
 * \brief A binding, range, or predicate, without inference or normalization.
 */
class Constraint : public ffi::ObjectRef {
 public:
  /*!
   * \brief Adopt an existing premise node.
   */
  explicit Constraint(ffi::ObjectPtr<ConstraintObj> node) : ffi::ObjectRef(std::move(node)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Constraint, ffi::ObjectRef, ConstraintObj);
};

/*!
 * \brief The value captured when a variable is bound. A mutable-read RHS is retained as data;
 * it must not be re-evaluated or treated as an enduring equality by a consumer.
 */
class BindingConstraintObj : public ConstraintObj {
 public:
  BindingConstraintObj(Var var, PrimExpr value) : var(std::move(var)), value(std::move(value)) {}
  explicit BindingConstraintObj(ffi::UnsafeInit tag) : var(tag), value(tag) {}
  /*!
   * \brief The variable whose bound value is recorded.
   */
  Var var;
  /*!
   * \brief The RHS expression at the binding site, including any mutable reads.
   */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<BindingConstraintObj>()
        .def_ro("var", &BindingConstraintObj::var)
        .def_ro("value", &BindingConstraintObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.BindingConstraint", BindingConstraintObj, ConstraintObj);
};

/*!
 * \brief Managed reference to BindingConstraintObj.
 */
class BindingConstraint : public Constraint {
 public:
  /*!
   * \brief Construct a premise from its fields, without inference.
   */
  BindingConstraint(Var var, PrimExpr value)
      : Constraint(ffi::make_object<BindingConstraintObj>(std::move(var), std::move(value))) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BindingConstraint, Constraint,
                                                BindingConstraintObj);
};

/*!
 * \brief A variable lies in a half-open range. The range need not enumerate every attainable
 * value.
 */
class RangeConstraintObj : public ConstraintObj {
 public:
  RangeConstraintObj(Var var, Range range) : var(std::move(var)), range(std::move(range)) {}
  explicit RangeConstraintObj(ffi::UnsafeInit tag) : var(tag), range(tag) {}
  /*!
   * \brief The variable whose possible values lie in the range.
   */
  Var var;
  /*!
   * \brief A half-open superset of the variable's possible values.
   */
  Range range;

  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<RangeConstraintObj>()
        .def_ro("var", &RangeConstraintObj::var)
        .def_ro("range", &RangeConstraintObj::range);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.RangeConstraint", RangeConstraintObj, ConstraintObj);
};

/*!
 * \brief Managed reference to RangeConstraintObj.
 */
class RangeConstraint : public Constraint {
 public:
  /*!
   * \brief Construct a premise from its fields, without inference.
   */
  RangeConstraint(Var var, Range range)
      : Constraint(ffi::make_object<RangeConstraintObj>(std::move(var), std::move(range))) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RangeConstraint, Constraint, RangeConstraintObj);
};

/*!
 * \brief A boolean premise. Its validity and lifetime are supplied by the collector.
 */
class PredicateConstraintObj : public ConstraintObj {
 public:
  explicit PredicateConstraintObj(PrimExpr condition, bool assume_always = false)
      : condition(std::move(condition)), assume_always(assume_always) {}
  explicit PredicateConstraintObj(ffi::UnsafeInit tag) : condition(tag) {}
  /*!
   * \brief The boolean predicate supplied by the collector.
   */
  PrimExpr condition;
  /*!
   * \brief Whether to assume the predicate remains true throughout the active
   * constraint scope, including after mutations.
   * Defaults to false. Setting true supplies the caller's guarantee of predicate
   * truth; it does not require referenced values to remain unchanged.
   */
  bool assume_always{false};

  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<PredicateConstraintObj>()
        .def_ro("condition", &PredicateConstraintObj::condition)
        .def_ro("assume_always", &PredicateConstraintObj::assume_always, refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.PredicateConstraint", PredicateConstraintObj,
                                    ConstraintObj);
};

/*!
 * \brief Managed reference to PredicateConstraintObj.
 */
class PredicateConstraint : public Constraint {
 public:
  /*!
   * \brief Construct a premise from its fields, without inference.
   */
  explicit PredicateConstraint(PrimExpr condition, bool assume_always = false)
      : Constraint(ffi::make_object<PredicateConstraintObj>(std::move(condition), assume_always)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PredicateConstraint, Constraint,
                                                PredicateConstraintObj);
};

/*!
 * \brief Tracks the constraints relevant to each point in a traversal.
 *
 * Holds variable bindings, value ranges, and predicates collected during traversal.
 * Callbacks can inspect these constraints when processing the current expression or statement.
 */
class ConstraintContextObj : public ffi::Object {
 public:
  ConstraintContextObj() = default;
  ConstraintContextObj(ffi::ObjectPtr<ConstraintContextObj> parent, int64_t parent_local_prefix)
      : parent_(std::move(parent)), parent_prefix_size_(parent_local_prefix) {
    TVM_FFI_CHECK(
        parent_ && 0 <= parent_local_prefix && parent_local_prefix <= parent_->LocalSize(),
        ValueError)
        << "Constraint context parent local prefix is out of bounds";
  }
  ConstraintContextObj(const ConstraintContextObj&) = delete;
  ConstraintContextObj& operator=(const ConstraintContextObj&) = delete;

  /*!
   * \brief Append a premise locally, preserving every previously captured prefix.
   */
  void Append(const Constraint& constraint) { constraints_.push_back(constraint); }

  /*!
   * \brief Number of local entries; excludes captured ancestors.
   */
  int64_t LocalSize() const { return constraints_.size(); }

  /*!
   * \brief Read constraints in inherited/local order.
   * The callback receives a const Constraint&. Later appends do not enter this iteration.
   */
  template <typename F>
  void ForEach(F&& f) const {
    // Keep this object alive if a callback replaces the caller's last handle.
    auto context = ffi::GetObjectPtr<ConstraintContextObj>(const_cast<ConstraintContextObj*>(this));
    context->ForEach(context->LocalSize(), std::forward<F>(f));
  }

  /*!
   * \brief Returns an independent array of the constraints in this context.
   * Includes inherited constraints in order. Later additions to the context do not
   * change the returned array. Constraint objects remain shared.
   */
  ffi::Array<Constraint> ToArray() const {
    ffi::Array<Constraint> result;
    ForEach([&](const Constraint& constraint) { result.push_back(constraint); });
    return result;
  }

  static constexpr bool _type_mutable = true;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.ConstraintContext", ConstraintContextObj, ffi::Object);

 private:
  template <typename F>
  void ForEach(int64_t local_prefix, F&& f) const {
    if (parent_) parent_->ForEach(parent_prefix_size_, f);
    for (int64_t i = 0; i < local_prefix; ++i) {
      // Read by value: a callback may append and reallocate the local list.
      const Constraint constraint = constraints_[i];
      f(constraint);
    }
  }

  // Fixed parent context; null for a root. Parent links are acyclic.
  const ffi::ObjectPtr<ConstraintContextObj> parent_;
  // Fixed prefix length in the parent's LOCAL sequence, not its effective count.
  const int64_t parent_prefix_size_{0};
  /*!
   * \brief Append-only sequence of constraints local to this context.
   *
   * New entries may be appended at the end. Existing entries must not be
   * removed, replaced, reordered, or modified through aliases. Appending
   * preserves the contents and meaning of every previously captured prefix.
   */
  ffi::List<Constraint> constraints_;
};

/*!
 * \brief Managed reference to ConstraintContextObj. Copies share the same context.
 */
class ConstraintContext : public ffi::ObjectRef {
 public:
  /*!
   * \brief Construct an empty root.
   */
  ConstraintContext() : ffi::ObjectRef(ffi::make_object<ConstraintContextObj>()) {}
  /*!
   * \brief Construct a child capturing [0, parent_local_prefix) of its parent's local entries.
   * Later parent appends do not enter this child's effective constraints.
   * \param parent Context supplying inherited constraints.
   * \param parent_local_prefix Number of parent-local entries to capture, between zero
   * and parent->LocalSize(), inclusive.
   */
  ConstraintContext(ConstraintContext parent, int64_t parent_local_prefix)
      : ffi::ObjectRef(ffi::make_object<ConstraintContextObj>(
            ffi::GetObjectPtr<ConstraintContextObj>(parent.get()), parent_local_prefix)) {}
  /*!
   * \brief Adopt an existing context object.
   */
  explicit ConstraintContext(ffi::ObjectPtr<ConstraintContextObj> node)
      : ffi::ObjectRef(std::move(node)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ConstraintContext, ffi::ObjectRef,
                                                ConstraintContextObj);
};

}  // namespace sym
}  // namespace tvm
#endif  // TVM_SYM_CONSTRAINT_H_
