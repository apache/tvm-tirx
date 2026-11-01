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
 * \brief C++ fragments that inspect, traverse, and rewrite add_one.
 *
 * Each function receives IR objects from Python and is registered as the
 * global function tirx_ir_guide_examples.cpp.<name>; working_with_ir_in_cpp.py
 * calls each function on add_one.
 */
// [includes.begin]
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
// [includes.end]

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace tvm;

// [inspect.begin]
ffi::String Inspect(const tirx::Function& func) {
  std::ostringstream os;
  for (const Var& param : func->params) {
    os << "param " << param->name << "\n";
  }
  SeqStmt body = func->body;
  if (const ForObj* loop = body[0].as<ForObj>()) {
    os << "loop over " << loop->var.as<VarObj>()->name << " up to " << loop->end << "\n";
    if (const TensorStoreObj* store = loop->body[0].as<TensorStoreObj>()) {
      os << "  store to " << store->dest.as<VarObj>()->name << ": " << store->value;
    }
  }
  return os.str();
}
// [inspect.end]

// [create.begin]
TensorStore DoubleStore(const tirx::Function& func) {
  For loop = func->body[0].as_or_throw<For>();
  TensorStore store = loop->body[0].as_or_throw<TensorStore>();
  PrimExpr doubled = store->value * 2;
  return TensorStore(store->dest, store->indices, doubled);
}
// [create.end]

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
      .def("tirx_ir_guide_examples.cpp.inspect", Inspect)
      .def("tirx_ir_guide_examples.cpp.double_store", DoubleStore)
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
