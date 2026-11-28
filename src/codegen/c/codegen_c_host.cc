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
 * \file codegen_c_host.cc
 */
#include "codegen_c_host.h"

#include <tvm/codegen/codegen.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/abi.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../min_max_utils.h"

namespace tvm {
namespace codegen {

CodeGenCHost::CodeGenCHost() {
  module_name_ = name_supply_->FreshName(ffi::symbol::tvm_ffi_library_ctx);
}

void CodeGenCHost::Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
                        std::string target_str) {
  emit_asserts_ = emit_asserts;
  emit_fwd_func_decl_ = emit_fwd_func_decl;
  packed_globals_.clear();
  decl_stream << "// tvm target: " << target_str << "\n";
  decl_stream << "#include \"tvm/ffi/extra/c_env_api.h\"\n";
  decl_stream << "#include \"tvm/runtime/base.h\"\n";
  decl_stream << "#include \"tvm/runtime/c_backend_api.h\"\n";
  decl_stream << "#include \"tvm/ffi/c_api.h\"\n";
  decl_stream << "#define TVM_DLL TVM_FFI_DLL_EXPORT\n";
  decl_stream << "#include <math.h>\n";
  decl_stream << "#include <stdbool.h>\n";
  CodeGenCHost::InitGlobalContext();
  CodeGenC::Init(output_ssa);
}

void CodeGenCHost::InitFuncState(const Function& func) {
  CodeGenC::InitFuncState(func);
  object_frame_ = any_frame_ = "NULL";
  object_frame_count_ = any_frame_count_ = 0;
  // Keep parameters and locals from shadowing a handle or callback emitted for a prior function.
  for (const auto& [_, names] : packed_globals_) {
    name_supply_->ReserveName(names.handle);
    name_supply_->ReserveName(names.initializer);
  }
}

void CodeGenCHost::InitGlobalContext() {
  decl_stream << "void* " << ffi::symbol::tvm_ffi_library_ctx << " = NULL;\n";
}

void CodeGenCHost::DefineModuleName() { decl_stream << "void* " << module_name_ << " = NULL;\n"; }

void CodeGenCHost::AddFunction(const GlobalVar& gvar, const Function& func) {
  return AddFunction(gvar, func, /*emit_fwd_func_decl=*/false);
}

void CodeGenCHost::AddFunction(const GlobalVar& gvar, const Function& func,
                               bool emit_fwd_func_decl) {
  if (func->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
    DeclareFunction(gvar, func);
    return;
  }
  auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
  if (global_symbol) {
    function_names_.push_back(global_symbol.value());
  }

  emit_fwd_func_decl_ = emit_fwd_func_decl;
  CodeGenC::AddFunction(gvar, func);
  if (func->HasNonzeroAttr(tvm::tirx::attr::kIsEntryFunc.name) && !has_tvm_ffi_main_func_) {
    TVM_FFI_ICHECK(global_symbol.has_value())
        << "CodeGenCHost: The entry func must have the global_symbol attribute, "
        << "but function " << gvar << " only has attributes " << func->attrs;

    function_names_.push_back(ffi::symbol::tvm_ffi_main);
    stream << "// CodegenC: NOTE: Auto-generated entry function\n";
    PrintFuncPrefix(stream);
    PrintType(func->ret_type, stream);
    stream << " " << ffi::symbol::tvm_ffi_main
           << "(void* self, void* args,int num_args, void* result) {\n";
    stream << "  return " << global_symbol.value() << "(self, args, num_args, result);\n";
    stream << "}\n";
  }
}

void CodeGenCHost::GenerateForwardFunctionDeclarations(ffi::String global_symbol,
                                                       const ffi::Array<Type>& arg_types,
                                                       const Type& ret_type) {
  // Runtime declarations come from the included ABI headers. Reconstructing
  // them from IR would lose const qualifiers on their borrowed pointer inputs.
  if (!emit_fwd_func_decl_ || global_symbol.starts_with("TVMFFI")) return;
  for (auto& func_already_defined : GetFunctionNames()) {
    if (global_symbol == func_already_defined) {
      return;
    }
  }
  this->PrintFuncPrefix(fwd_decl_stream);
  this->PrintType(ret_type, fwd_decl_stream);
  fwd_decl_stream << " " << global_symbol << "(";
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (i > 0) {
      fwd_decl_stream << ", ";
    }
    CodeGenSourceBase::PrintType(arg_types[i], fwd_decl_stream);
  }
  fwd_decl_stream << ");\n";
}

void CodeGenCHost::PrintFuncPrefix(std::ostream& os) {  // NOLINT(*)
  os << "#ifdef __cplusplus\n"
     << "extern \"C\"\n"
     << "#endif\n"
     << "TVM_DLL ";
}

void CodeGenCHost::PrintType(const PrimType& type, std::ostream& os) {  // NOLINT(*)
  int lanes = type.lanes();
  if (type.MatchesCode(DLDataTypeCode::kDLOpaqueHandle) && !type.IsVoid()) {
    TVM_FFI_ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (type.IsVoid()) {
    os << "void";
    return;
  }
  if (type.MatchesCode(DLDataTypeCode::kDLBool) && lanes == 1) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (type.MatchesCode(DLDataTypeCode::kDLFloat)) {
    switch (type.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (type.MatchesCode(DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLInt)) {
    if (type.MatchesCode(DLDataTypeCode::kDLUInt)) {
      os << 'u';
    }
    switch (type.bits()) {
      case 8:
        os << "int8_t";
        break;
      case 16:
        os << "int16_t";
        break;
      case 32:
        os << "int32_t";
        break;
      case 64:
        os << "int64_t";
        break;
      case 1:
        os << "int32_t";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  TVM_FFI_THROW(InternalError) << "Cannot convert type " << type->dtype << " to C type";
}

void CodeGenCHost::Dispatch_(const prim::BroadcastObj* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExprToSSA(op->value);
  int lanes = op->ty.as_or_throw<PrimType>().lanes();
  os << "((";
  PrintType(op->ty.as_or_throw<PrimType>(), os);
  os << ")(";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
}

// Initialized ownership frames and full-capacity cleanup on error. Normal-path
// retains, transfers and drops are already explicit in the input IR.
void CodeGenCHost::PrintType(const Type& type, std::ostream& os) {
  if (type.as<ObjectTypeObj>())
    os << "TVMFFIObjectHandle";
  else if (type.as<AnyTypeObj>())
    os << "TVMFFIAny";
  else
    CodeGenC::PrintType(type, os);
}

void CodeGenCHost::DeclareFrameCleanup() {
  if (managed_cleanup_declared_) return;
  managed_cleanup_declared_ = true;
  decl_stream << R"(
// Match the runtime's cold-code convention with C-compatible syntax.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((cold))
#endif
static void __tvm_managed_cleanup_on_error(TVMFFIObjectHandle* objects, int64_t nobj,
                                          TVMFFIAny* anys, int64_t nany) {
  TVMFFIObjectHandle primary = NULL;
  TVMFFIErrorMoveFromRaised(&primary);
  for (int64_t i = 0; i < nobj; ++i) {
    TVMFFIObjectHandle value = objects[i];
    objects[i] = NULL;
    if (value) TVMFFIObjectDecRef(value);
  }
  for (int64_t i = 0; i < nany; ++i) {
    TVMFFIAny value = anys[i];
    anys[i].type_index = kTVMFFINone;
    anys[i].zero_padding = 0;
    anys[i].v_int64 = 0;
    if (value.type_index >= kTVMFFIStaticObjectBegin) TVMFFIObjectDecRef(value.v_obj);
  }
  TVMFFIErrorSetRaised(primary);
  if (primary) TVMFFIObjectDecRef(primary);
}
)";
}

void CodeGenCHost::EmitFrameCleanup(const std::string& scratch_result) {
  bool has_frames = object_frame_count_ || any_frame_count_;
  if (!has_frames && scratch_result.empty()) return;
  DeclareFrameCleanup();
  if (!scratch_result.empty()) {
    // A failing packed callee can leave an owned value in the ordinary result cell.
    PrintIndent();
    stream << "__tvm_managed_cleanup_on_error(NULL, 0, " << scratch_result << ", 1);\n";
  }
  if (has_frames) {
    PrintIndent();
    stream << "__tvm_managed_cleanup_on_error(" << object_frame_ << ", " << object_frame_count_
           << ", " << any_frame_ << ", " << any_frame_count_ << ");\n";
  }
  PrintIndent();
}

void CodeGenCHost::CheckRuntimeStatus(const std::string& call, const std::string& scratch_result) {
  auto status = name_supply_->FreshName("status");
  PrintIndent();
  stream << "int " << status << " = " << call << ";\n";
  PrintIndent();
  stream << "if (" << status << " != 0) {\n";
  auto scope = BeginScope();
  EmitFrameCleanup(scratch_result);
  stream << "return " << status << ";\n";
  EndScope(scope);
  PrintIndent();
  stream << "}\n";
}

// Ordinary packed calls share argument/result storage and error handling.
std::string CodeGenCHost::PrintGetFuncFromBackend(const std::string& func_name) {
  auto [it, inserted] = packed_globals_.try_emplace(func_name);
  PackedGlobalNames& names = it->second;
  if (inserted) {
    auto fresh_global = [&](const std::string& hint) {
      std::string name;
      do {
        name = name_supply_->FreshName(hint);
      } while (func_name_supply_->ContainsName(name));
      func_name_supply_->ReserveName(name);
      return name;
    };
    names.handle = fresh_global(func_name + "_packed");
    names.initializer = fresh_global(names.handle + "_init");
    decl_stream << "static void* " << names.handle << " = NULL;\n";
    decl_stream << "static int " << names.initializer << "(void** result) {\n"
                << "  return TVMFFIEnvModLookupFromImports(" << module_name_ << ", ";
    PrintEscapedCString(func_name, decl_stream);
    decl_stream << ", result);\n}\n";
  }
  CheckRuntimeStatus("TVMFFIHandleInitOnce(&" + names.handle + ", " + names.initializer + ")");
  return names.handle;
}

void CodeGenCHost::PrintCallPacked(const CallObj* op, std::ostream& os) {
  auto name = op->args[0].as_or_throw<StringImm>()->value;
  int64_t begin = op->args[2].as_or_throw<IntImm>()->value.as<int64_t>().value();
  int64_t end = op->args[3].as_or_throw<IntImm>()->value.as<int64_t>().value();
  bool lookup = op->op.same_as(tirx::call_packed_lowered_op());
  std::string callee;
  if (lookup) {
    std::string handle = PrintGetFuncFromBackend(name);
    callee = "TVMFFIFunctionCall(" + handle;
  } else {
    auto symbol = ffi::symbol::tvm_ffi_symbol_prefix + name;
    if (!func_name_supply_->ContainsName(symbol)) {
      decl_stream << "#ifdef __cplusplus\nextern \"C\"\n#endif\nint " << symbol
                  << "(void*, TVMFFIAny*, int32_t, TVMFFIAny*);\n";
    }
    callee = symbol + "(NULL";
  }
  auto stack = PrintExprToSSA(op->args[1]);
  auto result_stack = PrintExprToSSA(op->args[4]);
  auto result_index = PrintExpr(op->args[5]);
  auto result = name_supply_->FreshName("packed_result_slot");
  PrintIndent();
  stream << "TVMFFIAny* " << result << " = (TVMFFIAny*)" << result_stack << " + (" << result_index
         << ");\n";
  CheckRuntimeStatus(callee + ", ((TVMFFIAny*)" + stack + " + " + std::to_string(begin) + "), " +
                         std::to_string(end - begin) + ", " + result + ")",
                     op->args[1].same_as(op->args[4]) ? result : "");
  if (!IsVoidType(op->ty) && !(op->ty.as<PrimType>() && op->ty.as_or_throw<PrimType>().IsVoid())) {
    PrintType(op->ty, os << "((");
    auto type = op->ty.as<PrimType>();
    os << ")" << result << "->"
       << (op->ty.as<PtrTypeObj>()                      ? "v_ptr"
           : type && type.value().MatchesCode(kDLFloat) ? "v_float64"
                                                        : "v_int64")
       << ")";
  }
}

std::string CodeGenCHost::EmitFrameArray(const char* name, const char* type, const char* empty,
                                         int64_t count) {
  // Emit a stack array and initialize every slot before a fallible operation can use it.
  if (!count) return "NULL";
  auto frame = name_supply_->FreshName(name);
  PrintIndent();
  stream << type << " " << frame << "[" << count << "];\n";
  PrintIndent();
  stream << "for (int64_t i = 0; i < " << count << "; ++i) " << frame << "[i] = " << empty << ";\n";
  return frame;
}

void CodeGenCHost::EmitAllocObjectFrame(const CallObj* op, std::ostream& os) {
  // Emit a frame of null object handles and record its bounds for failure cleanup.
  object_frame_count_ = op->args[0].as_or_throw<IntImm>()->value.as<int64_t>().value();
  object_frame_ = EmitFrameArray("object_frame", "TVMFFIObjectHandle", "NULL", object_frame_count_);
  os << object_frame_;
}

void CodeGenCHost::EmitAllocAnyFrame(const CallObj* op, std::ostream& os) {
  // Emit a frame of empty tagged values so failure cleanup can scan every slot.
  any_frame_count_ = op->args[0].as_or_throw<IntImm>()->value.as<int64_t>().value();
  any_frame_ = EmitFrameArray("any_frame", "TVMFFIAny", "(TVMFFIAny){kTVMFFINone, 0, {0}}",
                              any_frame_count_);
  os << any_frame_;
}

void CodeGenCHost::EmitRetainAnySlot(const CallObj* op, std::ostream& os) {
  // Convert the borrowed value or string literal to an owned Any in the destination slot.
  auto ptr = PrintExpr(op->args[0]);
  auto index = PrintExpr(op->args[1]);
  auto borrowed = name_supply_->FreshName("borrowed_any");
  std::string convert;
  if (auto* literal = op->args[2].as<StringImmObj>()) {
    PrintIndent();
    stream << "static const TVMFFIByteArray " << borrowed << " = {";
    PrintEscapedCString(literal->value, stream);
    stream << ", " << literal->value.size() << "};\n";
    convert = "TVMFFIStringFromByteArray";
  } else {
    auto value = PrintExpr(op->args[2]);
    PrintIndent();
    stream << "TVMFFIAny " << borrowed << " = " << value << ";\n";
    convert = "TVMFFIAnyViewToOwnedAny";
  }
  CheckRuntimeStatus(convert + "(&" + borrowed + ", ((TVMFFIAny*)" + ptr + ") + " + index + ")");
  os << "0";
}

std::string CodeGenCHost::EmitClearSlot(const CallObj* op, const char* type, const char* empty) {
  // Save the value and clear its slot before a release can trigger frame cleanup.
  auto ptr = PrintExprToSSA(op->args[0]);
  auto index = PrintExpr(op->args[1]);
  auto slot = name_supply_->FreshName("drop_slot");
  auto value = name_supply_->FreshName("drop");
  PrintIndent();
  stream << type << "* " << slot << " = (" << type << "*)" << ptr << " + (" << index << ");\n";
  PrintIndent();
  stream << type << " " << value << " = *" << slot << ";\n";
  PrintIndent();
  stream << "*" << slot << " = " << empty << ";\n";
  return value;
}

void CodeGenCHost::EmitConditionalDecRef(const std::string& object, const std::string& condition) {
  PrintIndent();
  stream << "if (" << condition << ") {\n";
  auto scope = BeginScope();
  CheckRuntimeStatus("TVMFFIObjectDecRef(" + object + ")");
  EndScope(scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenCHost::EmitDropObjectSlot(const CallObj* op, std::ostream& os) {
  // A non-null handle owns one object reference.
  auto object = EmitClearSlot(op, "TVMFFIObjectHandle", "NULL");
  EmitConditionalDecRef(object, object);
  os << "0";
}

void CodeGenCHost::EmitDropAnySlot(const CallObj* op, std::ostream& os) {
  // Only an object-tagged Any owns an object reference.
  auto value = EmitClearSlot(op, "TVMFFIAny", "(TVMFFIAny){kTVMFFINone, 0, {0}}");
  EmitConditionalDecRef(value + ".v_obj", value + ".type_index >= kTVMFFIStaticObjectBegin");
  os << "0";
}

void CodeGenCHost::Dispatch_(const CallObj* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(tirx::alloc_object_frame_op())) return EmitAllocObjectFrame(op, os);
  if (op->op.same_as(tirx::alloc_any_frame_op())) return EmitAllocAnyFrame(op, os);
  if (op->op.same_as(tirx::retain_any_slot_op())) return EmitRetainAnySlot(op, os);
  if (op->op.same_as(tirx::drop_object_slot_op())) return EmitDropObjectSlot(op, os);
  if (op->op.same_as(tirx::drop_any_slot_op())) return EmitDropAnySlot(op, os);

  if (op->op.same_as(tirx::stack_alloca_op())) {
    std::string stack_name = name_supply_->FreshName("stack");
    const std::string& type = op->args[0].as<StringImmObj>()->value;
    const IntImmObj* num = op->args[1].as<IntImmObj>();
    TVM_FFI_ICHECK(num != nullptr);
    static_assert(alignof(TVMFFIAny) % alignof(DLTensor) == 0, "invariant");
    size_t count = num->value.as<size_t>().value();
    size_t unit = sizeof(TVMFFIAny);
    size_t size = 0;
    if (type == "shape") {
      size = (count * sizeof(ffi::Shape::index_type) + unit - 1) / unit;
    } else if (type == "tvm_ffi_any") {
      size = (count * sizeof(TVMFFIAny) + unit - 1) / unit;
    } else if (type == "array") {
      size = (count * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      TVM_FFI_THROW(InternalError) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
    this->stream << "TVMFFIAny " << stack_name << "[" << size << "];\n";
    os << "((";
    PrintType(op->ty, os);
    os << ")" << stack_name << ")";
  } else if (op->op.same_as(tirx::call_packed_lowered_op())) {
    this->PrintCallPacked(op, os);
  } else if (op->op.same_as(tirx::call_cpacked_lowered_op())) {
    this->PrintCallPacked(op, os);
  } else if (op->op.same_as(tirx::throw_last_error_op())) {
    this->PrintIndent();
    EmitFrameCleanup();
    this->stream << "return -1;\n";
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

void CodeGenCHost::Dispatch_(const AssertStmtObj* op) {  // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    PrintIndent();
    stream << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    int num_parts = static_cast<int>(op->message_parts.size());
    PrintIndent();
    stream << "const char* __tvm_assert_parts[" << num_parts << "] = {";
    for (int i = 0; i < num_parts; ++i) {
      if (i > 0) stream << ", ";
      PrintEscapedCString(op->message_parts[i]->value, stream);
    }
    stream << "};\n";
    PrintIndent();
    stream << "TVMFFIErrorSetRaisedFromCStrParts(";
    PrintEscapedCString(op->error_kind->value, stream);
    stream << ", __tvm_assert_parts, " << num_parts << ");\n";
    PrintIndent();
    EmitFrameCleanup();
    stream << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
    stream << "}\n";
  }
}

void CodeGenCHost::Dispatch_(const FloatImmObj* op, std::ostream& os) {  // NOLINT(*)
  if (std::isfinite(op->value)) {
    CodeGenC::Dispatch_(op, os);
    return;
  }

  const char* literal;
  if (std::isnan(op->value)) {
    literal = "NAN";
  } else if (op->value < 0) {
    literal = "-INFINITY";
  } else {
    literal = "INFINITY";
  }
  os << "((";
  PrintType(op->ty.as_or_throw<PrimType>(), os);
  os << ")" << literal << ")";
}

void CodeGenCHost::Dispatch_(const prim::MinObj* op, std::ostream& os) {  // NOLINT(*)
  PrintMinMax(op->a, op->b, /*is_min=*/true, os);
}

void CodeGenCHost::Dispatch_(const prim::MaxObj* op, std::ostream& os) {  // NOLINT(*)
  PrintMinMax(op->a, op->b, /*is_min=*/false, os);
}

void CodeGenCHost::PrintMinMax(const PrimExpr& lhs, const PrimExpr& rhs, bool is_min,
                               std::ostream& os) {  // NOLINT(*)
  std::string lhs_value = "(" + PrintExprToSSA(lhs) + ")";
  std::string rhs_value = "(" + PrintExprToSSA(rhs) + ")";
  PrimType dtype = lhs.ty().as_or_throw<PrimType>();
  FloatNaNStatus lhs_nan_status = GetFloatNaNStatus(lhs);
  FloatNaNStatus rhs_nan_status = GetFloatNaNStatus(rhs);
  const char* compare = is_min ? "<" : ">";
  const char* take_rhs_compare = is_min ? ">=" : "<=";

  std::string condition;
  std::string true_value = lhs_value;
  std::string false_value = rhs_value;
  if (!dtype.MatchesCode(DLDataTypeCode::kDLFloat) ||
      lhs_nan_status == FloatNaNStatus::kKnownNotNaN) {
    // Integers and a known non-NaN lhs need only the strict comparison.
    condition = lhs_value + " " + compare + " " + rhs_value;
  } else if (lhs_nan_status == FloatNaNStatus::kKnownNaN) {
    os << lhs_value;
    return;
  } else if (rhs_nan_status == FloatNaNStatus::kKnownNaN) {
    // isnan(lhs) ? lhs : rhs: select lhs only if it is also NaN.
    condition = lhs_value + " != " + lhs_value;
  } else if (rhs_nan_status == FloatNaNStatus::kKnownNotNaN) {
    // For min: lhs >= rhs ? rhs : lhs. A NaN lhs makes >= false and is kept.
    condition = lhs_value + " " + take_rhs_compare + " " + rhs_value;
    true_value = rhs_value;
    false_value = lhs_value;
  } else {
    // For min: (lhs < rhs || isnan(lhs)) ? lhs : rhs. Max uses > instead.
    condition = "(" + lhs_value + " " + compare + " " + rhs_value + ") || (" + lhs_value +
                " != " + lhs_value + ")";
  }
  os << "((" << condition << ") ? " << true_value << " : " << false_value << ")";
}

ffi::Module BuildCHost(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = true;
  bool emit_fwd_func_decl = true;

  CodeGenCHost cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str());
  cg.SetConstantsByteAlignment(target->GetAttr<int64_t>("constants-byte-alignment").value_or(16));

  std::vector<std::pair<GlobalVar, Function>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    TVM_FFI_ICHECK(base_func->IsInstance<FunctionObj>()) << "CodegenCHost: Can only take Function";
    auto function = base_func.as_or_throw<Function>();
    funcs.push_back({gvar, function});
  }

  // Sort functions by name for deterministic output.
  std::sort(funcs.begin(), funcs.end(), [](const auto& kv_a, const auto& kv_b) {
    return kv_a.first->name_hint < kv_b.first->name_hint;
  });

  for (const auto& [gvar, function] : funcs) {
    cg.DeclareFunction(gvar, function);
  }

  // Codegen all functions.  Passing emit_fwd_func_decl=true adds a
  // forward declaration for any `tirx::call_extern_op`, based on the
  // arguments provided to it.
  for (const auto& [gvar, function] : funcs) {
    cg.AddFunction(gvar, function, emit_fwd_func_decl);
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.c", BuildCHost);
}
}  // namespace codegen
}  // namespace tvm
