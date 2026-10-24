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
 * \file tvm/tirx/structural_visit_with_constraint_context.h
 * \brief Structural traversal with ordered lexical symbolic premises.
 */
#ifndef TVM_TIRX_STRUCTURAL_VISIT_WITH_CONSTRAINT_CONTEXT_H_
#define TVM_TIRX_STRUCTURAL_VISIT_WITH_CONSTRAINT_CONTEXT_H_

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/sym/constraint.h>

namespace tvm {
namespace tirx {

/*!
 * \brief Visits values using their structural visit behavior while maintaining
 * the constraint context relevant to each callback.
 * \sa ffi::StructuralVisitor, sym::ConstraintContext
 */
class StructuralVisitorWithConstraintContextObj : public ffi::StructuralVisitorObj {
 public:
  using VisitorObjType = StructuralVisitorWithConstraintContextObj;
  using StateTupleType = std::tuple<const sym::ConstraintContext&>;
  /*!
   * \brief Construct a visitor using constraint-aware default descent.
   */
  StructuralVisitorWithConstraintContextObj();
  /*!
   * \brief Active effective premises. Consumers must not append during traversal.
   * Retain this handle and constraints->LocalSize() for a stable position, or use
   * constraints->ToArray() for an independent sequence of handles. The context itself remains live.
   */
  sym::ConstraintContext constraints;

  /*!
   * \brief Add a fact to the current scope, creating its child context on the first append.
   * \param constraint Binding, range, or predicate valid in the current scope.
   */
  void AppendConstraint(const sym::Constraint& constraint) {
    if (needs_child_context_) {
      constraints = sym::ConstraintContext(constraints, constraints->LocalSize());
      needs_child_context_ = false;
    }
    constraints->Append(constraint);
  }

  /*!
   * \brief Run an Expected-returning callback in a constraint scope.
   * Use AppendConstraint to add facts. Scopes without facts keep the incoming context.
   * Success and visit interrupts restore the incoming position. Terminal errors
   * propagate without restoration; discard the traversal context after failure.
   */
  template <typename F>
  auto WithConstraintScope(F&& f) -> decltype(f()) {
    sym::ConstraintContext saved_context = constraints;
    int64_t saved_prefix = constraints->LocalSize();
    bool saved_needs_child = needs_child_context_;
    needs_child_context_ = true;
    auto result = std::forward<F>(f)();
    if (!result.is_err()) {
      constraints = saved_context->LocalSize() == saved_prefix
                        ? saved_context
                        : sym::ConstraintContext(saved_context, saved_prefix);
      needs_child_context_ = saved_needs_child;
    }
    return result;
  }

  /*!
   * \brief Descend with the sparse specialized hook, or the ordinary fallback.
   * Call this from a Visit callback to descend its matched node exactly once.
   * Calling it from that node's registered hook re-enters the same hook.
   *
   * - Bind and Assert add facts for later statements in the same sequence.
   * - If, While, and lazy if_then_else add scalar boolean guards inside branches/bodies.
   * - For and thread launch add ranges in their bodies; Let adds a binding in its body.
   * - Binding values and loop bounds are visited before their facts are added.
   * - Predicates retain their expressions (including likely). Only tirx.assume_always sets
   *   assume_always=true, guaranteeing truth throughout the containing scope, including mutations.
   * - Branch predicates, binding values, range bounds, and explicit assume_always predicates
   *   are collected without effect filtering. Ordinary assertions require a pure condition.
   *   Only known positive loop steps yield ranges.
   * - Binding values retain their original reads; eager Select adds no branch guards.
   * - Call and RegionStmt use tirx::op_attr hooks with the complete containing node.
   *   Unregistered Ops use ordinary Call descent or isolated RegionStmt body scopes.
   *
   * Semantic hooks assume well-formed IR. No analyzer is created. These expressions do not
   * snapshot mutable reads: consumers need stable captured values or lifetime guarantees,
   * invalidation, or skipped inference. An unchanged context does not imply unchanged memory.
   * Do not use one visitor for overlapping traversals.
   */
  ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> DefaultVisitExpected(
      ffi::AnyView value) noexcept;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.StructuralVisitorWithConstraintContext",
                              StructuralVisitorWithConstraintContextObj, ffi::StructuralVisitorObj);

 protected:
  explicit StructuralVisitorWithConstraintContextObj(const ffi::StructuralVisitorVTable* vtable)
      : ffi::StructuralVisitorObj(vtable) {}

  StateTupleType StateTuple() const noexcept { return {constraints}; }

 private:
  bool needs_child_context_{false};
};

/*!
 * \brief Owning reference to the active specialized visitor.
 */
class StructuralVisitorWithConstraintContext : public ffi::StructuralVisitor {
 public:
  /*!
   * \brief Adopt an active traversal object.
   */
  explicit StructuralVisitorWithConstraintContext(
      ffi::ObjectPtr<StructuralVisitorWithConstraintContextObj> node)
      : ffi::StructuralVisitor(std::move(node)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralVisitorWithConstraintContext,
                                                ffi::StructuralVisitor,
                                                StructuralVisitorWithConstraintContextObj);
};

/*!
 * \brief Expected form of StructuralWalkWithConstraintContext.
 */
template <ffi::WalkOrder order, typename... Callbacks>
ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> StructuralWalkWithConstraintContextExpected(
    ffi::AnyView root, Callbacks&&... callbacks) noexcept {
  static_assert(sizeof...(Callbacks) != 0, "At least one callback is required");
  using Engine = ffi::StructuralWalkEngine<StructuralVisitorWithConstraintContextObj, order,
                                           std::decay_t<Callbacks>...>;
  auto visitor = ffi::make_object<Engine>(std::forward<Callbacks>(callbacks)...);
  return visitor->VisitExpected(root);
}

/*!
 * \brief Walk every occurrence with (value, const sym::ConstraintContext&) callbacks.
 * A trailing TVMFFIDefRegionKind is optional, as in ffi::StructuralWalk.
 * Return Expected<WalkResult>; pre-order Skip prunes descent and Interrupt stops
 * the walk. Post callbacks see a scoped node after its internal scope is restored.
 * Sequence traversal visits its statements directly in order; lazy calls visit
 * their arguments directly so each branch has its own scope. These two hooks
 * do not dispatch a callback for their intermediate Array containers.
 * \code
 * StructuralWalkWithConstraintContext<ffi::WalkOrder::kPreOrder>(stmt,
 *   [](const Var& var, const sym::ConstraintContext& facts)
 *       -> ffi::Expected<ffi::WalkResult> {
 *     auto snapshot = facts->ToArray();
 *     return ffi::WalkResult::Advance();
 *   });
 * \endcode
 */
template <ffi::WalkOrder order, typename... Callbacks>
ffi::Optional<ffi::VisitInterrupt> StructuralWalkWithConstraintContext(ffi::AnyView root,
                                                                       Callbacks&&... callbacks) {
  return StructuralWalkWithConstraintContextExpected<order>(root,
                                                            std::forward<Callbacks>(callbacks)...)
      .value();
}

/*!
 * \brief Expected form of StructuralVisitWithConstraintContext.
 */
template <typename... Callbacks>
ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> StructuralVisitWithConstraintContextExpected(
    ffi::AnyView root, Callbacks&&... callbacks) noexcept {
  static_assert(sizeof...(Callbacks) != 0, "At least one callback is required");
  using Engine = ffi::StructuralVisitEngine<StructuralVisitorWithConstraintContextObj,
                                            std::decay_t<Callbacks>...>;
  auto visitor = ffi::make_object<Engine>(std::forward<Callbacks>(callbacks)...);
  return visitor->VisitExpected(root);
}

/*!
 * \brief Visit with callbacks (value, StructuralVisitorWithConstraintContextObj*).
 * The first matching callback owns descent. Return completion, interrupt, or
 * error as in ffi::StructuralVisit. Recurse with visitor->VisitExpected(child)
 * or visitor->DefaultVisitExpected(value); constraints is the active context handle.
 */
template <typename... Callbacks>
ffi::Optional<ffi::VisitInterrupt> StructuralVisitWithConstraintContext(ffi::AnyView root,
                                                                        Callbacks&&... callbacks) {
  return StructuralVisitWithConstraintContextExpected(root, std::forward<Callbacks>(callbacks)...)
      .value();
}

/*!
 * \brief Sparse hook with the same result/error contract as FStructuralVisit.
 */
using FStructuralVisitWithConstraintContext =
    ffi::reflection::NativeFunctionView<ffi::Optional<ffi::VisitInterrupt>(
        StructuralVisitorWithConstraintContextObj*, ffi::AnyView)>;
namespace type_attr {
/*!
 * \brief Optional constraint-aware descent; missing hooks use ordinary structural descent.
 */
inline constexpr ffi::reflection::AttrKey<FStructuralVisitWithConstraintContext>
    kStructuralVisitWithConstraintContext{"__s_visit_with_constraint_context__"};
}  // namespace type_attr

inline StructuralVisitorWithConstraintContextObj::StructuralVisitorWithConstraintContextObj()
    : StructuralVisitorWithConstraintContextObj([] {
        static const ffi::StructuralVisitorVTable table{[](ffi::StructuralVisitorObj* self,
                                                           ffi::AnyView value) noexcept {
          return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
              static_cast<StructuralVisitorWithConstraintContextObj*>(self)->DefaultVisitExpected(
                  value));
        }};
        return &table;
      }()) {}

inline ffi::Expected<ffi::Optional<ffi::VisitInterrupt>>
StructuralVisitorWithConstraintContextObj::DefaultVisitExpected(ffi::AnyView value) noexcept {
  static ffi::reflection::TypeAttrColumn column(type_attr::kStructuralVisitWithConstraintContext);
  auto attr = column[value.type_index()];
  if (attr == nullptr) return ffi::StructuralVisitorObj::DefaultVisitExpected(value);
  auto hook = attr.try_cast<FStructuralVisitWithConstraintContext>();
  if (!hook) {
    return ffi::Unexpected(ffi::Error(
        "TypeError",
        "__s_visit_with_constraint_context__ must be an opaque function pointer or ffi.Function",
        ""));
  }
  return hook->CallExpected(this, value);
}

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_STRUCTURAL_VISIT_WITH_CONSTRAINT_CONTEXT_H_
