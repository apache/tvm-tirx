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
 * \file working_with_ir.cc
 * \brief C++ fragments that build IR, and that traverse and rewrite add_one.
 *
 * Each function is registered as the global function
 * tirx_ir_guide_examples.cpp.<name>; working_with_ir_in_cpp.py calls the object
 * system functions without arguments and the others on add_one.
 */
// [includes.begin]
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
// [includes.end]

#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace tvm;

/*! \brief Print an object the way TVMScript writes it. */
std::string Print(const ffi::ObjectRef& node) {
  std::ostringstream os;
  os << node;
  return os.str();
}

// [construct.begin]
PrimExpr XTimes2Plus1() {
  PrimVar x("x", PrimType::Int(32));
  return x * 2 + 1;
}

ffi::String Construct() { return Print(XTimes2Plus1()); }
// [construct.end]

// [objects.begin]
ffi::Array<ffi::String> Objects() {
  prim::Add add = XTimes2Plus1().as_or_throw<prim::Add>();
  prim::Add alias = add;
  return {"a = " + Print(add->a), "b = " + Print(add->b),
          std::string("same object: ") + (alias.same_as(add) ? "true" : "false")};
}
// [objects.end]

// [cow.begin]
ffi::Array<ffi::String> CopyOnWrite() {
  prim::Add add = XTimes2Plus1().as_or_throw<prim::Add>();
  prim::Add changed = add;
  changed.CopyOnWrite()->b = 3;
  return {"add = " + Print(add), "changed = " + Print(changed)};
}
// [cow.end]

// [casts.begin]
ffi::Array<ffi::String> Casts() {
  Expr value = XTimes2Plus1();
  const prim::AddObj* node = value.as<prim::AddObj>();
  std::optional<prim::Add> add = value.as<prim::Add>();
  std::optional<prim::Mul> mul = value.as<prim::Mul>();
  ffi::Array<ffi::String> lines = {
      std::string("as<AddObj>: ") + (node != nullptr ? "pointer" : "nullptr"),
      std::string("as<Add>: ") + (add.has_value() ? "Add" : "nullopt"),
      std::string("as<Mul>: ") + (mul.has_value() ? "Mul" : "nullopt")};
  try {
    value.as_or_throw<prim::Mul>();
  } catch (const ffi::Error& err) {
    lines.push_back("as_or_throw<Mul>: " + std::string(err.message()));
  }
  return lines;
}
// [casts.end]

// [views.begin]
ffi::Array<ffi::String> TypedViews() {
  Var i("i", PrimType::Int(32));
  Var f("f", PrimType::Float(32));
  ffi::Array<ffi::String> lines = {
      std::string("i as PrimExpr: ") + (i.as<PrimExpr>().has_value() ? "yes" : "no"),
      std::string("i as IntExpr: ") + (i.as<IntExpr>().has_value() ? "yes" : "no"),
      std::string("f as IntExpr: ") + (f.as<IntExpr>().has_value() ? "yes" : "no")};
  try {
    f.as_or_throw<IntExpr>();
  } catch (const ffi::Error& err) {
    lines.push_back("as_or_throw<IntExpr>: " + std::string(err.message()));
  }
  return lines;
}
// [views.end]

// [describe.begin]
class Describe : public ExprFunctor<std::string(const Expr&)> {
 public:
  using ExprFunctor::Dispatch_;

  std::string Dispatch_(const prim::AddObj* add) override {
    return "(" + Dispatch(add->a) + " + " + Dispatch(add->b) + ")";
  }
  std::string Dispatch_(const prim::MulObj* mul) override {
    return "(" + Dispatch(mul->a) + " * " + Dispatch(mul->b) + ")";
  }
  std::string Dispatch_(const VarObj* var) override { return var->name; }
  std::string Dispatch_(const IntImmObj* imm) override { return Print(ffi::GetRef<IntImm>(imm)); }
  std::string DispatchDefault_(const ffi::Object*) override { return "<other>"; }
};
// [describe.end]

// [dispatch_use.begin]
ffi::Array<ffi::String> Dispatching() {
  PrimVar x("x", PrimType::Int(32));
  Describe describe;
  return {describe(XTimes2Plus1()), describe(x - 1)};
}
// [dispatch_use.end]

// [walk.begin]
ffi::Array<ffi::String> TensorsWritten(const tirx::Function& func) {
  ffi::Array<ffi::String> writes;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      func, [&](const TensorStore& store) -> ffi::Expected<ffi::WalkResult> {
        writes.push_back(store->dest.as<VarObj>()->name);
        return ffi::WalkResult::Advance();
      });
  return writes;
}
// [walk.end]

// [dispatch.begin]
ffi::Array<ffi::String> TraceStatements(const tirx::Function& func) {
  ffi::Array<ffi::String> trace;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      func,
      [&](const For& loop) -> ffi::Expected<ffi::WalkResult> {
        trace.push_back("loop " + loop->var.as<VarObj>()->name);
        return ffi::WalkResult::Advance();
      },
      [&](const TensorStore& store) -> ffi::Expected<ffi::WalkResult> {
        trace.push_back("store " + store->dest.as<VarObj>()->name);
        return ffi::WalkResult::Advance();
      });
  return trace;
}
// [dispatch.end]

// [stop.begin]
ffi::Optional<TensorStore> FindWrite(const tirx::Function& func) {
  ffi::Optional<ffi::VisitInterrupt> found = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      func, [](const TensorStore& store) -> ffi::Expected<ffi::WalkResult> {
        return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(store));
      });
  if (!found.has_value()) return std::nullopt;
  return found.value()->value.cast<TensorStore>();
}
// [stop.end]

// [definitions.begin]
ffi::Array<ffi::String> Definitions(const tirx::Function& func) {
  ffi::Array<ffi::String> sites;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      func, [&](const Var& var, TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
        if (kind == kTVMFFIDefRegionKindSimple) {
          sites.push_back(var->name + " simple");
        } else if (kind == kTVMFFIDefRegionKindPattern) {
          sites.push_back(var->name + " pattern");
        }
        return ffi::WalkResult::Advance();
      });
  return sites;
}
// [definitions.end]

// [specialize.begin]
tirx::Function Specialize(const tirx::Function& func, const Var& n) {
  ffi::Any result =
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(func, [&](const Var& var) -> ffi::Any {
        if (var.same_as(n)) {
          return IntImm(var->ty.as_or_throw<PrimType>(), 128);
        }
        return ffi::Unchanged();
      });
  return result.cast<tirx::Function>();
}
// [specialize.end]

// [equal.begin]
ffi::Array<bool> Compare(const tirx::Function& first, const tirx::Function& second) {
  return {
      first.same_as(second),
      ffi::StructuralEqual::Equal(first, second),
      ffi::StructuralHash::Hash(first) == ffi::StructuralHash::Hash(second),
  };
}
// [equal.end]

// [visit.begin]
class LoopNestVisitor : public tirx::StmtExprVisitor {
 public:
  using tirx::StmtExprVisitor::Visit_;

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForObj* loop) override {
    loops_.push_back(loop->var.as<VarObj>()->name);
    ffi::Optional<ffi::VisitInterrupt> result = tirx::StmtExprVisitor::Visit_(loop);
    loops_.pop_back();
    return result;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorStoreObj* store) override {
    std::string entry = std::string(store->dest.as<VarObj>()->name) + " in";
    for (const std::string& loop : loops_) entry += " " + loop;
    writes.push_back(entry);
    return tirx::StmtExprVisitor::Visit_(store);
  }

  ffi::Array<ffi::String> writes;

 private:
  std::vector<std::string> loops_;
};

ffi::Array<ffi::String> LoopsAroundWrites(const tirx::Function& func) {
  auto visitor = ffi::make_object<LoopNestVisitor>();
  visitor->Visit(func);
  return visitor->writes;
}
// [visit.end]

// [mutate.begin]
class ReverseLoopMutator : public tirx::StmtExprMutator {
 public:
  using tirx::StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode mode) override {
    reversed_.emplace(loop->var.as<VarObj>(), loop->end - 1 - loop->var);
    return tirx::StmtExprMutator::Mutate_(loop, mode);
  }

  UnchangedOr<Expr> Mutate_(const VarObj* var, InplaceMode mode) override {
    auto it = reversed_.find(var);
    if (it != reversed_.end()) return it->second;
    return tirx::StmtExprMutator::Mutate_(var, mode);
  }

 private:
  std::unordered_map<const VarObj*, PrimExpr> reversed_;
};

tirx::Function ReverseLoop(const tirx::Function& func) {
  auto mutator = ffi::make_object<ReverseLoopMutator>();
  return mutator->Mutate(func).ValueOrUnchanged(func).as_or_throw<tirx::Function>();
}
// [mutate.end]

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx_ir_guide_examples.cpp.construct", Construct)
      .def("tirx_ir_guide_examples.cpp.objects", Objects)
      .def("tirx_ir_guide_examples.cpp.copy_on_write", CopyOnWrite)
      .def("tirx_ir_guide_examples.cpp.casts", Casts)
      .def("tirx_ir_guide_examples.cpp.typed_views", TypedViews)
      .def("tirx_ir_guide_examples.cpp.dispatching", Dispatching)
      .def("tirx_ir_guide_examples.cpp.tensors_written", TensorsWritten)
      .def("tirx_ir_guide_examples.cpp.trace_statements", TraceStatements)
      .def("tirx_ir_guide_examples.cpp.find_write", FindWrite)
      .def("tirx_ir_guide_examples.cpp.definitions", Definitions)
      .def("tirx_ir_guide_examples.cpp.specialize", Specialize)
      .def("tirx_ir_guide_examples.cpp.compare", Compare)
      .def("tirx_ir_guide_examples.cpp.loops_around_writes", LoopsAroundWrites)
      .def("tirx_ir_guide_examples.cpp.reverse_loop", ReverseLoop);
}

}  // namespace
