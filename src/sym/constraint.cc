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

#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/sym/constraint.h>

namespace tvm {
namespace sym {
void ConstraintContextObj::RegisterReflection() {
  ffi::reflection::ObjectDef<ConstraintContextObj>();
  // Python iteration borrows contexts and reads individual entries, never the backing storage.
  ffi::reflection::GlobalDef()
      .def("sym.ConstraintContextGetParent",
           [](ConstraintContext context) {
             const auto* node = context.get();
             return ffi::Tuple<ffi::Optional<ConstraintContext>, int64_t>(
                 node->parent_ ? ffi::Optional<ConstraintContext>(ConstraintContext(node->parent_))
                               : std::nullopt,
                 node->parent_prefix_size_);
           })
      .def("sym.ConstraintContextGetLocal", [](ConstraintContext context, int64_t index) {
        TVM_FFI_CHECK(0 <= index && index < context->LocalSize(), IndexError)
            << "Constraint context local index is out of bounds";
        return context->constraints_[index];
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::ObjectDef<ConstraintObj>();
  ConstraintContextObj::RegisterReflection();
  BindingConstraintObj::RegisterReflection();
  RangeConstraintObj::RegisterReflection();
  PredicateConstraintObj::RegisterReflection();
  ffi::reflection::GlobalDef()
      .def("sym.ConstraintContext",
           [](ffi::Optional<ConstraintContext> parent, int64_t prefix) {
             if (parent.has_value()) return ConstraintContext(parent.value(), prefix);
             TVM_FFI_CHECK_EQ(prefix, 0, ValueError)
                 << "A root context requires a zero parent prefix";
             return ConstraintContext();
           })
      .def_method("sym.ConstraintContextAppend", &ConstraintContextObj::Append)
      .def_method("sym.ConstraintContextLocalSize", &ConstraintContextObj::LocalSize)
      .def_method("sym.ConstraintContextSnapshot", &ConstraintContextObj::Snapshot)
      .def("sym.BindingConstraint",
           [](Var var, PrimExpr value) { return BindingConstraint(var, value); })
      .def("sym.RangeConstraint", [](Var var, Range range) { return RangeConstraint(var, range); })
      .def_packed("sym.PredicateConstraint", [](ffi::PackedArgs args, ffi::Any* result) {
        TVM_FFI_CHECK(args.size() == 1 || args.size() == 2, TypeError)
            << "PredicateConstraint expects a condition and an optional assume_always flag";
        *result =
            PredicateConstraint(args[0].cast<PrimExpr>(), args.size() == 2 && args[1].cast<bool>());
      });
}
}  // namespace sym
}  // namespace tvm
