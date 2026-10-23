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
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/ir/stmt.h>
#include <tvm/script/printer/printer.h>
#include <tvm/script/printer/support.h>

#include <utility>

#include "utils.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  using script::printer::RegisterScriptRepr;
  refl::GlobalDef()
      .def("node.TVMScriptPrinterScript", tvm::Script)
      .def("script.printer.Script", tvm::Script);

  RegisterScriptRepr<DataTypeImmObj>();
  RegisterScriptRepr<GenericConstObj>();
  RegisterScriptRepr<AnyTypeObj>();
  RegisterScriptRepr<NoneTypeObj>();
  RegisterScriptRepr<CallObj>();
  RegisterScriptRepr<DictAttrsObj>();
  RegisterScriptRepr<FloatImmObj>();
  RegisterScriptRepr<FuncTypeObj>();
  RegisterScriptRepr<GlobalVarObj>();
  RegisterScriptRepr<IRModuleObj>();
  RegisterScriptRepr<IntImmObj>();
  RegisterScriptRepr<MissingTypeObj>();
  RegisterScriptRepr<PtrTypeObj>();
  RegisterScriptRepr<PrimTypeObj>();
  RegisterScriptRepr<RangeObj>();
  RegisterScriptRepr<StringImmObj>();
  RegisterScriptRepr<TensorLoadObj>();
  RegisterScriptRepr<TensorRegionTypeObj>();
  RegisterScriptRepr<TupleObj>();
  RegisterScriptRepr<TupleGetItemObj>();
  RegisterScriptRepr<TupleTypeObj>();
  RegisterScriptRepr<VarObj>();
  RegisterScriptRepr<prim::AddObj>();
  RegisterScriptRepr<prim::AndObj>();
  RegisterScriptRepr<prim::BitwiseAndObj>();
  RegisterScriptRepr<prim::BitwiseNotObj>();
  RegisterScriptRepr<prim::BitwiseOrObj>();
  RegisterScriptRepr<prim::BitwiseXorObj>();
  RegisterScriptRepr<prim::BroadcastObj>();
  RegisterScriptRepr<prim::CastObj>();
  RegisterScriptRepr<prim::DivObj>();
  RegisterScriptRepr<prim::EQObj>();
  RegisterScriptRepr<prim::FloorDivObj>();
  RegisterScriptRepr<prim::FloorModObj>();
  RegisterScriptRepr<prim::GEObj>();
  RegisterScriptRepr<prim::GTObj>();
  RegisterScriptRepr<prim::LEObj>();
  RegisterScriptRepr<prim::LShiftObj>();
  RegisterScriptRepr<prim::LTObj>();
  RegisterScriptRepr<prim::LetObj>();
  RegisterScriptRepr<prim::MaxObj>();
  RegisterScriptRepr<prim::MinObj>();
  RegisterScriptRepr<prim::ModObj>();
  RegisterScriptRepr<prim::MulObj>();
  RegisterScriptRepr<prim::NEObj>();
  RegisterScriptRepr<prim::NotObj>();
  RegisterScriptRepr<prim::OrObj>();
  RegisterScriptRepr<prim::RShiftObj>();
  RegisterScriptRepr<prim::RampObj>();
  RegisterScriptRepr<prim::SelectObj>();
  RegisterScriptRepr<prim::ShuffleObj>();
  RegisterScriptRepr<prim::SubObj>();
  RegisterScriptRepr<AssertStmtObj>();
  RegisterScriptRepr<RegionStmtObj>();
  RegisterScriptRepr<BindObj>();
  RegisterScriptRepr<BreakObj>();
  RegisterScriptRepr<TensorStoreObj>();
  RegisterScriptRepr<ContinueObj>();
  RegisterScriptRepr<EvaluateObj>();
  RegisterScriptRepr<ForObj>();
  RegisterScriptRepr<IfObj>();
  RegisterScriptRepr<LambdaExprObj>();
  RegisterScriptRepr<ReturnObj>();
  RegisterScriptRepr<SeqStmtObj>();
  RegisterScriptRepr<WhileObj>();
  RegisterScriptRepr<TensorRegionObj>();
}

}  // namespace tvm
