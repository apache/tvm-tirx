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
 * \file lower_ffi_abi.cc
 * \brief Lower packed calls and managed values to explicit host FFI ABI operations.
 */
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir_utils.h"

namespace tvm {
namespace tirx {
namespace {

// -----------------------------------------------------------------------------
// Overall strategy
// -----------------------------------------------------------------------------
// Example code before and after the transformation:
//
//   @I.ir_module
//   class Before:
//       @T.function
//       def main(out: T.handle) -> T.int32:
//           x: T.let[T.Any] = T.call_packed("f", ty=T.Any)
//           y: T.let[T.Any] = T.call_packed("g", x, ty=T.Any)
//           return T.store_return(out, y)
//
//   @I.ir_module
//   class After:
//       @T.function
//       def main(out: T.handle) -> T.int32:
//           frame: T.let[T.handle] = T.alloc_any_frame(T.int64(2))
//           args: T.let[T.handle] = T.stack_alloca("tvm_ffi_any", 1)
//
//           T.call_packed_lowered("f", args, 0, 0, frame, 0, ty="void")
//           T.abi_field_set(args, 0, 18, T.abi_field_get(frame, 0, 18, ty=T.Any))
//           T.call_packed_lowered("g", args, 0, 1, frame, 1, ty="void")
//           T.drop_any_slot(frame, T.int64(0))
//
//           # Move y to the caller and clear its former slot.
//           T.abi_field_set(out, 0, 18, T.abi_field_get(frame, 1, 18, ty=T.Any))
//           T.abi_field_set(frame, 1, 13, 0)
//           T.abi_field_set(frame, 1, 14, 0)
//           T.abi_field_set(frame, 1, 15, T.int64(0))
//           return 0
//
// Strategy: Allocate explicit frames for owned Any and Object locals. Insert a
// drop after each local's last use, or transfer its reference to the caller if
// the local is returned.
//
// - ManagedLocalLastUse: Step 0. Map each original statement to the locals
//   whose last use occurs there.
// - ABIRewriter: Step 1. Rewrite calls, returns, and scope exits using that plan.
//   - ManagedFrame: Used by ABIRewriter to reserve and reuse ownership slots
//     and emit the reads, retains, drops, and clears for those slots.
//   - PackedCallBuilder: Used by ABIRewriter to pack arguments into scratch
//     and emit each call with its chosen result slot.
// - Host codegen contract after this pass: On error, release all references
//   still owned by the frames, including partially written call results,
//   and propagate the original error.

// -----------------------------------------------------------------------------
// Input and lifetime analysis
// -----------------------------------------------------------------------------

/*!
 * \brief Compute last-use statements in the lexical scope of each managed local.
 * \note A use inside a branch or loop is charged to its enclosing statement in the defining scope.
 *   Borrowed parameters have no scheduled drop.
 */
class ManagedLocalLastUse : public StmtExprVisitor {
 public:
  /*!
   * \brief Collect lexical last uses before rewriting any statement.
   * \param func Host definition with unique statement occurrences.
   * \return The ending locals for each original statement, in definition order.
   */
  ffi::Map<Stmt, ffi::List<Var>> Run(const Function& func) {
    CheckAndRecordManagedType(func->ret_type);
    for (const Var& param : func->params) CheckAndRecordManagedType(param->ty, param);
    Visit(func->body);
    ffi::Map<Stmt, ffi::List<Var>> drops_after;
    for (const auto& [var, stmt] : last_use_) {
      if (std::optional<ffi::List<Var>> vars = drops_after.Get(stmt)) {
        vars->push_back(var);
      } else {
        drops_after.Set(stmt, {var});
      }
    }
    return drops_after;
  }

 private:
  /*! \brief Bindings in one lexical sequence and the statement to which nested uses are charged. */
  struct Scope {
    ffi::Optional<Stmt> body_owner;
    // Enclosing statement to charge when a nested expression reads this scope's local.
    ffi::Optional<Stmt> current_stmt;
    // Managed locals defined here, or borrowed parameters in the root scope.
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> managed_local_vars;
  };

  bool CheckAndRecordManagedType(const Type& type, ffi::Optional<Var> definition = std::nullopt) {
    if (auto object = type.as<ObjectTypeObj>()) {
      TVM_FFI_CHECK(object->type_key == "ffi.Object", TypeError)
          << "Managed packed ABI supports only declared ffi.Object, not " << object->type_key;
    }
    bool managed = type.as<AnyTypeObj>() || type.as<ObjectTypeObj>();
    if (managed && definition) {
      scopes_.Current().managed_local_vars.insert(definition.value());
    }
    return managed;
  }

  const Scope* FindDefScope(const Var& var) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (it->managed_local_vars.count(var)) {
        return &*it;
      }
    }
    return nullptr;
  }

  // Frame accesses and returns cross all enclosing boundaries. A loop exit crosses
  // only the boundaries up to and including its target loop.
  void CheckBoundaries(bool stop_at_loop = false) const {
    static auto sequential = Op::GetAttrMap(tvm::op_attr::kRegionIsSequentialScope);
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (!it->body_owner) continue;
      const Stmt& owner = it->body_owner.value();
      if (auto region = owner.as<RegionStmtObj>()) {
        TVM_FFI_CHECK(sequential.get(region->op, false), ValueError)
            << "Unsupported managed host region or control-flow exit across it: " << region->op;
      }
      if (auto loop = owner.as<ForObj>()) {
        TVM_FFI_CHECK(loop->kind == ForKind::kDefault, ValueError)
            << "Managed owners require sequential loops; an exit cannot cross a nonsequential loop";
        if (stop_at_loop) break;
      } else if (stop_at_loop && owner.as<WhileObj>()) {
        break;
      }
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj* seq) final {
    auto owner = scopes_.Current().current_stmt;
    scopes_.WithNewScope([&] {
      Scope& scope = scopes_.Current();
      scope.body_owner = owner;
      for (const Stmt& stmt : seq->seq) {
        scope.current_stmt = stmt;
        Visit(stmt);
      }
    });
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ReturnObj* ret) final {
    CheckBoundaries();
    const auto* call = ret->value.as<CallObj>();
    if (call && call->op.same_as(store_return_op())) {
      call = call->args[1].as<CallObj>();
    }
    if (call && (call->op.same_as(call_packed_op()) || call->op.same_as(call_cpacked_op()))) {
      TVM_FFI_THROW(ValueError)
          << "A packed call cannot be returned directly. Bind its result to a variable first.";
    }
    return Visit(ret->value);
  }

  ffi::Optional<VisitInterrupt> Visit_(const BreakObj*) final {
    CheckBoundaries(/*stop_at_loop=*/true);
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ContinueObj*) final {
    CheckBoundaries(/*stop_at_loop=*/true);
    return std::nullopt;
  }

  // An unused binding ends at its own statement. Read its initializer before recording it.
  ffi::Optional<VisitInterrupt> Visit_(const BindObj* bind) final {
    Visit(bind->value);
    if (CheckAndRecordManagedType(bind->var->ty, bind->var)) {
      CheckBoundaries();
      last_use_.Set(bind->var, ffi::GetRef<Stmt>(bind));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* var) final {
    if (var->ty.as<AnyTypeObj>() || var->ty.as<ObjectTypeObj>()) {
      Var binding = ffi::GetRef<Var>(var);
      const Scope* scope = FindDefScope(binding);
      if (scope && scope->current_stmt) {
        CheckBoundaries();
        last_use_.Set(binding, scope->current_stmt.value());
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* call) final {
    if (call->op.same_as(store_return_op()) &&
        (call->args[1]->ty.as<AnyTypeObj>() || call->args[1]->ty.as<ObjectTypeObj>())) {
      // Returning even a borrowed parameter uses a temporary ownership slot.
      CheckBoundaries();
    }
    if (call->op.same_as(call_packed_op()) || call->op.same_as(call_cpacked_op())) {
      TVM_FFI_CHECK(!inside_call_argument_, ValueError)
          << "A packed call cannot be used inside a call argument. Bind its result to a variable "
             "first.";
    }
    // The lazy conditional checks its arms separately. The packed wrapper's
    // store_return represents the source return, not an eager call in that source.
    bool outer = inside_call_argument_;
    inside_call_argument_ |=
        !call->op.same_as(prim::if_then_else_op()) && !call->op.same_as(store_return_op());
    CheckAndRecordManagedType(call->ty);
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (i == 4 && call->op.same_as(stack_make_dltensor_op())) {
        continue;  // Only the static type of the dtype witness enters the descriptor.
      }
      const Expr& arg = call->args[i];
      if (CheckAndRecordManagedType(arg->ty)) {
        TVM_FFI_CHECK(!call->op.same_as(reinterpret_op()), ValueError)
            << "LowerFFIABI input cannot reinterpret a managed value";
      }
      Visit(arg);
    }
    inside_call_argument_ = outer;
    return std::nullopt;
  }

  bool inside_call_argument_ = false;
  // Accumulate the last statement for each local, retaining definition order for ties.
  ffi::Map<Var, Stmt> last_use_;
  // Active lexical blocks and their statement cursors, with borrowed parameters at the
  // root.
  ScopeStack<Scope> scopes_;
};

// -----------------------------------------------------------------------------
// Packed-call construction
// -----------------------------------------------------------------------------

// PackedCallBuilder reserves argument, shape, and DLTensor scratch. It encodes
// evaluated arguments and emits the low-level call into the chosen result cell.
class PackedCallBuilder {
 public:
  /*!
   * \brief Element offsets saved on entry to a packed call and restored after argument
   * preparation.
   */
  struct CallStackOffsets {
    /*! \brief First TVMFFIAny cell available to this call. */
    int64_t args_offset;
    /*! \brief First DLTensor descriptor available to this call. */
    int64_t dl_tensor_offset;
    /*! \brief First int64 dimension available to this call. */
    int64_t shape_offset;
  };

  /*!
   * \brief Check whether a declared ABI type represents void or None.
   * \param type The declared type of the operand or result.
   * \return True for primitive void, the empty tuple used as void, or None.
   * \note A producer of one of these types may still have side effects.
   */
  static bool IsVoidOrNoneType(const Type& type) {
    auto prim = type.as<PrimType>();
    return type.as<NoneTypeObj>() || IsVoidType(type) || (prim && prim.value().IsVoid());
  }

  /*!
   * \brief Check whether an enclosing packed call has a live reservation.
   * \return True between ReserveCall and its matching EmitCall, even for a zero-argument call.
   */
  bool HasActivePackedCall() const { return args_stack_.offset != 0; }

  /*!
   * \brief Emit the scratch allocations for one function or parallel body.
   * \param prefix Statement list to which the allocation bindings are appended.
   * \note Call after rewriting the body so each allocation uses the peak number of elements.
   */
  void PrependAllocations(ffi::Array<Stmt>* prefix) const {
    args_stack_.Allocate(prefix, "tvm_ffi_any");
    dltensor_stack_.Allocate(prefix, "array");
    shape_stack_.Allocate(prefix, "shape");
  }

  /*!
   * \brief Reserve the argument cells for a packed call before visiting its operands.
   * \param cells Number of TVMFFIAny cells, including a scratch result cell when needed.
   * \return The previous offsets in TVMFFIAny cells, DLTensor descriptors and int64 dimensions.
   * \note Pass these offsets to EmitCall to restore the enclosing call's reservations.
   */
  CallStackOffsets ReserveCall(size_t cells) {
    return {args_stack_.Reserve(std::max<size_t>(1, cells)), dltensor_stack_.offset,
            shape_stack_.offset};
  }

  /*!
   * \brief Reserve contiguous shape or stride storage for the active call.
   * \param count Number of int64 dimensions.
   * \return A pointer to the first reserved dimension. An empty shape still reserves one element.
   */
  Expr ReserveShape(size_t count) {
    TVM_FFI_CHECK(HasActivePackedCall(), ValueError)
        << "stack_make_shape must occur in a packed call";
    int64_t begin = shape_stack_.Reserve(std::max<size_t>(1, count));
    Expr data = shape_stack_.data;
    if (begin) {
      data = Call(data->ty, ptr_byte_offset_op(), {data, IntImm::Int64(begin * sizeof(int64_t))});
    }
    return data;
  }

  /*!
   * \brief Reserve one DLTensor descriptor for the active call.
   * \return The descriptor stack and the element index used by ABI field loads and stores.
   * \note The descriptor stays reserved until EmitCall restores the saved offsets.
   */
  std::pair<Var, int64_t> ReserveDLTensor() {
    TVM_FFI_CHECK(HasActivePackedCall(), ValueError)
        << "stack_make_dltensor must occur in a packed call";
    return {dltensor_stack_.data, dltensor_stack_.Reserve(1)};
  }

  /*!
   * \brief Emit stores that encode a value as a TVMFFIAny.
   * \param body Statement list to which evaluation and stores are appended.
   * \param any_frame Base pointer of the destination TVMFFIAny array.
   * \param index Absolute element index in the destination array.
   * \param value Value to encode. Its evaluation precedes changes to the destination.
   * \param tensor_argument Whether the value is a prepared DLTensor address needing its ABI tag.
   * \note The caller manages ownership when storing an object or copying an Any.
   */
  static void StorePackedValue(ffi::Array<Stmt>* body, const Var& any_frame, int64_t index,
                               Expr value, bool tensor_argument = false) {
    auto store_field = [&](int field, Expr payload) {
      body->push_back(
          Evaluate(Call(PrimType::Int(32), abi_field_set_op(),
                        {any_frame, IntImm::Int32(index), IntImm::Int32(field), payload})));
    };
    if (value->ty.as<AnyTypeObj>()) {
      store_field(kTVMFFIAnyValue, value);
      return;
    }
    bool immediate = value.as<VarObj>() || value.as<IntImmObj>() || value.as<FloatImmObj>() ||
                     value.as<ConstantObj>() || value.as<StringImmObj>();
    if (!immediate && !tensor_argument && !IsVoidOrNoneType(value->ty)) {
      Var saved("packed_value", value->ty);
      body->push_back(Bind(saved, value));
      value = saved;
    }
    Expr tag = IntImm::Int32(kTVMFFINone);
    if (IsVoidOrNoneType(value->ty)) {
      if (!immediate) {
        body->push_back(Evaluate(value));
      }
      value = IntImm::Int64(0);
    } else if (value->ty.as<ObjectTypeObj>()) {
      Var pointer("object", PtrType::VoidPtrType());
      body->push_back(Bind(pointer, Call(pointer->ty, reinterpret_op(), {value})));
      value = pointer;
      tag = Call(PrimType::Int(32), abi_field_get_op(),
                 {pointer, IntImm::Int32(0), IntImm::Int32(kTVMFFIObjectTypeIndex)});
    } else if (value.as<StringImmObj>()) {
      tag = IntImm::Int32(kTVMFFIRawStr);
      value = tirx::reinterpret(PtrType::VoidPtrType(), value);
    } else if (tensor_argument) {
      tag = IntImm::Int32(kTVMFFIDLTensorPtr);
    } else if (value->ty.as<PtrTypeObj>()) {
      tag = Call(PrimType::Int(32), prim::if_then_else_op(),
                 {Call(PrimType::Bool(), isnullptr_op(), {value}), IntImm::Int32(kTVMFFINone),
                  IntImm::Int32(kTVMFFIOpaquePtr)});
    } else {
      auto scalar = value.as_or_throw<PrimExpr>();
      PrimType abi_type = APIType(scalar.ty());
      tag = IntImm::Int32(scalar.ty().MatchesCode(kDLBool) ? kTVMFFIBool
                          : abi_type.MatchesCode(kDLFloat) ? kTVMFFIFloat
                                                           : kTVMFFIInt);
      if (scalar.ty() != abi_type) {
        value = prim::Cast(abi_type, scalar);
      }
    }
    store_field(kTVMFFIAnyTypeIndex, tag);
    store_field(kTVMFFIAnyZeroPadding, IntImm::Int32(0));
    store_field(kTVMFFIAnyUnionValue, value);
  }

  /*!
   * \brief Restore the enclosing reservations and construct the lowered packed call.
   * \param body Statement list for initializing a scratch result to None before invocation.
   * \param call Logical call whose first operand identifies the callee.
   * \param type Result type of the emitted call expression.
   * \param offsets Stack offsets saved by the matching ReserveCall.
   * \param result_any_frame TVMFFIAny result array, or nullptr to use the ordinary argument stack.
   * \param slot Element index in result_any_frame. A scratch result uses the cell after the
   * arguments.
   * \return The lowered call with its half-open argument range and separate result address.
   */
  Call EmitCall(ffi::Array<Stmt>* body, const CallObj* call, Type type, CallStackOffsets offsets,
                const Var* result_any_frame, int64_t slot) {
    int64_t begin = offsets.args_offset;
    args_stack_.offset = begin;
    dltensor_stack_.offset = offsets.dl_tensor_offset;
    shape_stack_.offset = offsets.shape_offset;
    size_t count = call->args.size() - 1;
    if (!result_any_frame) {
      result_any_frame = &args_stack_.data;
      slot = begin + count;
      StorePackedValue(body, *result_any_frame, slot, GenericConst(NoneType(), nullptr));
    }
    const Op& op =
        call->op.same_as(call_cpacked_op()) ? call_cpacked_lowered_op() : call_packed_lowered_op();
    return Call(type, op,
                {call->args[0], args_stack_.data, IntImm::Int32(begin),
                 IntImm::Int32(begin + count), *result_any_frame, IntImm::Int32(slot)},
                {}, {}, call->loc);
  }

  /*!
   * \brief Encode one argument in the active call's reserved TVMFFIAny array.
   * \param body Statement list for evaluating and storing the argument.
   * \param index Absolute element index in the argument stack.
   * \param arg Rewritten argument, including the address of a prepared DLTensor when applicable.
   * \note Store each argument before rewriting the next to preserve operand evaluation order.
   */
  void StoreArgument(ffi::Array<Stmt>* body, int64_t index, const Expr& arg) {
    StorePackedValue(body, args_stack_.data, index, arg, IsDLTensorAddress(arg));
  }

 private:
  static bool IsDLTensorAddress(const Expr& value) {
    auto call = value.as<CallObj>();
    if (!call || !call->op.same_as(abi_field_get_op())) {
      return false;
    }
    auto field = call->args[2].as<IntImmObj>();
    return field && field->value == kDLTensorAddr;
  }

  /*!
   * \brief Track reservations in one typed scratch array while lowering packed calls.
   *
   * The builder uses one stack each for TVMFFIAny cells, DLTensor descriptors and int64
   * shape or stride elements. Reserve advances the offset beyond the enclosing call's live
   * elements. EmitCall restores the offsets saved by ReserveCall so siblings can reuse
   * the space. The peak capacity sizes the allocation emitted after rewriting the body.
   */
  struct ScratchStack {
    /*! \brief Base pointer bound by Allocate and used to address elements during rewriting. */
    Var data;
    /*! \brief Index of the first available element above the active reservations. */
    int64_t offset = 0;
    /*! \brief Maximum number of simultaneously reserved elements in this array. */
    int64_t capacity = 0;

    int64_t Reserve(int64_t count) {
      int64_t begin = offset;
      offset += count;
      capacity = std::max(capacity, offset);
      return begin;
    }

    void Allocate(ffi::Array<Stmt>* prefix, const char* kind) const {
      if (capacity) {
        prefix->push_back(Bind(
            data, Call(data->ty, stack_alloca_op(), {StringImm(kind), IntImm::Int32(capacity)})));
      }
    }
  };

  // Argument cells and scalar-only result slots share a TVMFFIAny array.
  ScratchStack args_stack_{Var("stack_ffi_any", PtrType::VoidPtrType())};
  // Simultaneously live DLTensor descriptors share an array.
  ScratchStack dltensor_stack_{Var("stack_array", PtrType::VoidPtrType())};
  // Shapes and strides share a flat array of int64 dimensions.
  ScratchStack shape_stack_{Var("stack_shape", PtrType(PrimType::Int(64)))};
};

// -----------------------------------------------------------------------------
// ABI and ownership rewriting
// -----------------------------------------------------------------------------

// ManagedFrame tracks live and reusable slots for one ownership representation.
// It emits the reads, retains, drops, and clears for those slots.
struct ManagedFrame {
  /*! \brief Select the runtime slot representation and its matching access and cleanup operations.
   */
  enum class Kind { kObject, kAny };

  /*!
   * \brief Construct the frame for one ownership layout.
   * \param data Base pointer to use in emitted slot operations.
   * \param kind Runtime representation of each slot.
   */
  ManagedFrame(Var data, Kind kind) : data(std::move(data)), kind_(kind) {}

  // Base pointer used by emitted slot operations.
  Var data;
  // Live local bindings. Temporary call results reserve indices without binding a Var.
  ffi::Map<Var, int64_t> slot_index_map;
  // Peak element count used to allocate this frame at function entry.
  int64_t capacity = 0;

  /*!
   * \brief Reserve an empty slot for a local or temporary result.
   * \return Element index reused from the free list or added to the eventual frame capacity.
   */
  int64_t AllocSlot() {
    if (free_slots_.empty()) {
      return capacity++;
    }
    int64_t index = free_slots_.back();
    free_slots_.pop_back();
    return index;
  }

  /*!
   * \brief Make an index available for reuse during rewriting.
   * \param index Slot whose runtime owner has already been dropped or transferred on this path.
   */
  void FreeSlot(int64_t index) { free_slots_.push_back(index); }

  /*!
   * \brief Construct a borrowed read of an occupied slot.
   * \param type Managed type of the result expression.
   * \param index Element index in this frame.
   * \return An ABI field load of the object handle or tagged Any value.
   */
  Expr Read(const Type& type, int64_t index) const {
    return Call(type, abi_field_get_op(),
                {data, IntImm::Int32(index),
                 IntImm::Int32(kind_ == Kind::kObject ? kTVMFFIObjectHandle : kTVMFFIAnyValue)});
  }

  /*!
   * \brief Construct a runtime release for a slot of this frame's kind.
   * \param index Element index whose value is released.
   * \return A slot operation that clears the cell and releases its owned reference.
   */
  Stmt EmitDrop(int64_t index) const {
    return Evaluate(Call(PrimType::Int(32),
                         kind_ == Kind::kObject ? drop_object_slot_op() : drop_any_slot_op(),
                         {data, IntImm::Int64(index)}));
  }

  void EmitRetain(ffi::Array<Stmt>& body, int64_t slot, Expr value) const {
    if (kind_ == Kind::kAny) {
      body.push_back(Evaluate(
          Call(PrimType::Int(32), retain_any_slot_op(), {data, IntImm::Int64(slot), value})));
      return;
    }
    Var borrowed("borrowed_object", PtrType::VoidPtrType());
    body.push_back(Bind(borrowed, Call(borrowed->ty, reinterpret_op(), {value})));
    body.push_back(Evaluate(
        Call(PrimType::Int(32), call_extern_op(), {StringImm("TVMFFIObjectIncRef"), borrowed})));
    body.push_back(
        Evaluate(Call(PrimType::Int(32), abi_field_set_op(),
                      {data, IntImm::Int32(slot), IntImm::Int32(kTVMFFIObjectHandle), borrowed})));
  }

  /*!
   * \brief Empty a slot after its reference has been transferred.
   * \param body Statement list receiving the stores.
   * \param index Element index to reset to a null object handle or a tagged None.
   */
  void Clear(ffi::Array<Stmt>& body, int64_t index) const {
    if (kind_ == Kind::kObject) {
      body.push_back(Evaluate(
          Call(PrimType::Int(32), abi_field_set_op(),
               {data, IntImm::Int32(index), IntImm::Int32(kTVMFFIObjectHandle), ConstHandle(0)})));
    } else {
      PackedCallBuilder::StorePackedValue(&body, data, index, GenericConst(NoneType(), nullptr));
    }
  }

 private:
  // Object slots hold bare handles and Any slots hold tagged TVMFFIAny values.
  Kind kind_;
  // Vacant elements available for reuse.
  std::vector<int64_t> free_slots_;
};

/*!
 * \brief Lower packed invocations and place ownership operations at lexical and control-flow
 *   boundaries.
 * \note A binding acquires its destination before the sequence drops inputs at their last use.
 *   Control-flow exits emit path-specific drops while sibling paths retain their compile-time
 *   bindings.
 */
class ABIRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  // -----------------------------------------------------------------------------
  // Rewriter state and function entry
  // -----------------------------------------------------------------------------
  // Group the lifetime plan, frames, scope stack, pending statements, scratch,
  // and device context here. Run analyzes last uses, rewrites the function body,
  // then prepends allocations sized from peak usage.

  /*!
   * \brief Rewrite the body and prepend the storage sized by its peak reservations.
   * \param func Eligible host definition. A function using managed frames must return int32
   *   status.
   * \return The function with lowered calls, frame declarations and scratch allocations.
   */
  Function Run(Function func) {
    drops_after_ = ffi::make_object<ManagedLocalLastUse>()->Run(func);
    if (ffi::Optional<Target> target = func->GetAttr(tvm::attr::kTarget)) {
      device_type_ = IntImm::Int32(target.value()->GetTargetDeviceType());
    }
    SeqStmt body =
        scopes_.WithNewScope([&] { return Mutate(func->body).ValueOrUnchanged(func->body); });
    ffi::Array<Stmt> prefix;
    if (object_frame_.capacity || any_frame_.capacity) {
      TVM_FFI_CHECK(ffi::StructuralEqual()(func->ret_type, PrimType::Int(32)), TypeError)
          << "Managed frames require an int32 status-returning function";
    }
    if (object_frame_.capacity) {
      prefix.push_back(
          Bind(object_frame_.data, Call(object_frame_.data->ty, alloc_object_frame_op(),
                                        {IntImm::Int64(object_frame_.capacity)})));
    }
    if (any_frame_.capacity) {
      prefix.push_back(Bind(any_frame_.data, Call(any_frame_.data->ty, alloc_any_frame_op(),
                                                  {IntImm::Int64(any_frame_.capacity)})));
    }
    packed_call_builder_.PrependAllocations(&prefix);
    for (const Stmt& stmt : body->seq) prefix.push_back(stmt);
    func.CopyOnWrite()->body = SeqStmt(prefix);
    return func;
  }

 private:
  /*! \brief Active lexical definitions and the control-flow boundary enclosing them. */
  struct Scope {
    // Mixed Object and Any bindings in source definition order.
    std::vector<Var> definitions;
    // A break or continue unwinds through the innermost loop scope.
    bool is_loop = false;
  };

  // Preparatory statements for the current statement.
  ffi::Array<Stmt> pending_stmts_;
  // Packed argument, shape, and DLTensor scratch for this function or parallel body.
  PackedCallBuilder packed_call_builder_;
  // Lexical device type for DLTensor descriptors, inherited through nested calls.
  ffi::Optional<PrimExpr> device_type_;
  // Lexical device id for descriptors built inside the current device region.
  ffi::Optional<PrimExpr> device_id_;
  // Original statements and the managed locals whose lifetimes end there.
  ffi::Map<Stmt, ffi::List<Var>> drops_after_;
  // Lexical bindings and loop boundaries used for path-specific exit cleanup.
  ScopeStack<Scope> scopes_;
  // Entry-owned slots holding bare object handles.
  ManagedFrame object_frame_{Var("objects", PtrType::VoidPtrType()), ManagedFrame::Kind::kObject};
  // Entry-owned slots holding tagged Any values, including packed results.
  ManagedFrame any_frame_{Var("anys", PtrType::VoidPtrType()), ManagedFrame::Kind::kAny};

  Expr Rewrite(const Expr& value) { return Mutate(value).ValueOrUnchanged(value); }

  // -----------------------------------------------------------------------------
  // Managed bindings and variable reads
  // -----------------------------------------------------------------------------
  // Bind reserves the destination before rewriting its initializer. Var borrows
  // from the live slot. Keep CallAndStoreResult beside Bind for the tagged-result
  // check and the transfer from an Any slot to an Object slot.

  UnchangedOr<Stmt> Mutate_(const BindObj* bind, InplaceMode mode) final {
    if (!bind->var->ty.as<AnyTypeObj>() && !bind->var->ty.as<ObjectTypeObj>()) {
      return StmtExprMutator::Mutate_(bind, mode);
    }
    ManagedFrame& frame = bind->var->ty.as<ObjectTypeObj>() ? object_frame_ : any_frame_;
    int64_t slot = frame.AllocSlot();
    ffi::Array<Stmt> body;
    auto call = bind->value.as<CallObj>();
    if (call && (call->op.same_as(call_packed_op()) || call->op.same_as(call_cpacked_op()))) {
      CallAndStoreResult(body, slot, call, bind->var->ty);
    } else {
      TVM_FFI_CHECK(bind->value.as<VarObj>() || (call && call->op.same_as(abi_field_get_op())),
                    ValueError)
          << "Unsupported managed binding operation: " << bind->value;
      frame.EmitRetain(body, slot, Rewrite(bind->value));
    }
    frame.slot_index_map.Set(bind->var, slot);
    scopes_.Current().definitions.push_back(bind->var);
    return SeqStmt(std::move(body));
  }

  void CallAndStoreResult(ffi::Array<Stmt>& body, int64_t binding, const CallObj* call,
                          const Type& type) {
    // A packed callee always writes TVMFFIAny. Typed results are checked while
    // that Any slot still owns the result, then moved into their object slot.
    bool object = type.as<ObjectTypeObj>();
    int64_t index = object ? any_frame_.AllocSlot() : binding;
    body.push_back(Evaluate(EmitPackedCall(call, PrimType::Void(), &any_frame_.data, index)));
    if (!object) {
      return;
    }
    // Generic ffi.Object accepts every concrete object tag, but never null.
    // Keep this check active even when optional assertions are disabled.
    auto tag = Call(PrimType::Int(32), abi_field_get_op(),
                    {any_frame_.data, IntImm::Int32(index), IntImm::Int32(kTVMFFIAnyTypeIndex)})
                   .as_or_throw<PrimExpr>();
    body.push_back(If(
        tag < IntImm::Int32(kTVMFFIStaticObjectBegin),
        SeqStmt({Evaluate(Call(PrimType::Void(), call_extern_op(),
                               {StringImm("TVMFFIErrorSetRaisedFromCStr"), StringImm("TypeError"),
                                StringImm("Expected non-null ffi.Object")})),
                 Evaluate(Call(PrimType::Void(), throw_last_error_op(), {}))})));
    Expr checked_value =
        Call(type, abi_field_get_op(),
             {any_frame_.data, IntImm::Int32(index), IntImm::Int32(kTVMFFIAnyUnionValue)});
    body.push_back(Evaluate(Call(PrimType::Int(32), abi_field_set_op(),
                                 {object_frame_.data, IntImm::Int32(binding),
                                  IntImm::Int32(kTVMFFIObjectHandle), checked_value})));
    any_frame_.Clear(body, index);
    any_frame_.FreeSlot(index);
  }

  Expr ReadManagedValue(const Var& var) {
    ManagedFrame& frame = var->ty.as<ObjectTypeObj>() ? object_frame_ : any_frame_;
    if (ffi::Optional<int64_t> slot = frame.slot_index_map.Get(var)) {
      return frame.Read(var->ty, *slot);
    }
    return var;
  }

  UnchangedOr<Expr> Mutate_(const VarObj* var, InplaceMode mode) final {
    if (var->ty.as<AnyTypeObj>() || var->ty.as<ObjectTypeObj>()) {
      return ReadManagedValue(ffi::GetRef<Var>(var));
    }
    return StmtExprMutator::Mutate_(var, mode);
  }

  // -----------------------------------------------------------------------------
  // Call lowering
  // -----------------------------------------------------------------------------
  // Dispatch by callee. EmitPackedCall evaluates and stores operands in order,
  // then invokes the callee with the selected result cell. Shape and DLTensor
  // operations construct descriptors in the active call's scratch space.

  UnchangedOr<Expr> Mutate_(const CallObj* call, InplaceMode mode) final {
    if (!call->op.same_as(call_packed_op()) && !call->op.same_as(call_cpacked_op())) {
      if (call->op.same_as(prim::if_then_else_op())) {
        return MutateIfThenElse(call, mode);
      }
      if (call->op.same_as(stack_make_shape_op())) {
        return EmitStackMakeShape(call);
      }
      if (call->op.same_as(stack_make_dltensor_op())) {
        return EmitDLTensor(call);
      }
      return StmtExprMutator::Mutate_(call, mode);
    }
    bool managed_result = call->ty.as<AnyTypeObj>() || call->ty.as<ObjectTypeObj>();
    if (PackedCallBuilder::IsVoidOrNoneType(call->ty)) {
      pending_stmts_.push_back(CallAndDiscard(call));
      return GenericConst(NoneType(), nullptr);
    }
    TVM_FFI_CHECK(!managed_result, ValueError) << "Managed packed results must be bound";
    Call invoke = EmitPackedCall(call, call->ty);
    // Capture the result at its evaluation site before another call reuses the scratch cell.
    Var value("packed_value", call->ty);
    pending_stmts_.push_back(Bind(value, invoke));
    return value;
  }

  Call EmitPackedCall(const CallObj* call, Type type, const Var* result_any_frame = nullptr,
                      int64_t slot = 0) {
    size_t count = call->args.size() - 1;
    PackedCallBuilder::CallStackOffsets offsets =
        packed_call_builder_.ReserveCall(count + (result_any_frame == nullptr));
    for (size_t i = 0; i < count; ++i) {
      const Expr& operand = call->args[i + 1];
      if ((operand->ty.as<AnyTypeObj>() || operand->ty.as<ObjectTypeObj>()) &&
          !operand.as<VarObj>()) {
        auto producer = operand.as<CallObj>();
        TVM_FFI_CHECK(producer && producer->op.same_as(abi_field_get_op()), ValueError)
            << "Managed packed arguments must be bound variables or ABI loads";
      }
      // Store this operand before rewriting the next one, which may prepare a descriptor.
      Expr arg = Rewrite(operand);
      packed_call_builder_.StoreArgument(&pending_stmts_, offsets.args_offset + i, arg);
    }
    return packed_call_builder_.EmitCall(&pending_stmts_, call, type, offsets, result_any_frame,
                                         slot);
  }

  Expr EmitStackMakeShape(const CallObj* call) {
    size_t count = call->args.size();
    Expr data = packed_call_builder_.ReserveShape(count);
    if (count) {
      TensorVar dims("shape", TensorType({IntImm::Int64(count)}, PrimType::Int(64)));
      pending_stmts_.push_back(
          Bind(dims, Call(dims.ty(), decl_tensor_op(), {data}, {}, {dims.ty()})));
      for (size_t i = 0; i < count; ++i) {
        auto dim = Rewrite(call->args[i]).as_or_throw<PrimExpr>();
        pending_stmts_.push_back(
            TensorStore(dims, {IntImm::Int32(i)}, prim::cast(dim, PrimType::Int(64))));
      }
    }
    return data;
  }

  Expr EmitDLTensor(const CallObj* call) {
    auto [frame, index] = packed_call_builder_.ReserveDLTensor();
    TVM_FFI_CHECK(device_id_ && device_type_, ValueError) << "DLTensor needs a device context";
    auto field = [&](int kind, Expr value) {
      pending_stmts_.push_back(
          Evaluate(Call(PrimType::Int(32), abi_field_set_op(),
                        {frame, IntImm::Int32(index), IntImm::Int32(kind), value})));
    };
    field(kDLTensorData, Rewrite(call->args[0]));
    field(kDLTensorShape, Rewrite(call->args[1]));
    Expr strides = Rewrite(call->args[2]);
    if (auto zero = strides.as<PrimExpr>(); zero && prim::IsZero(zero.value())) {
      strides = ConstHandle(0);
    }
    field(kDLTensorStrides, strides);
    field(kDLTensorNDim, Rewrite(call->args[3]));
    PrimType dtype = call->args[4]->ty.as_or_throw<PrimType>();
    field(kDLTensorTypeCode, IntImm(PrimType::UInt(8), dtype.code()));
    field(kDLTensorTypeBits, IntImm(PrimType::UInt(8), dtype.bits()));
    field(kDLTensorTypeLanes, IntImm(PrimType::UInt(16), dtype.lanes()));
    auto offset = Rewrite(call->args[5]).as_or_throw<PrimExpr>();
    PrimExpr bytes = offset * IntImm(offset.ty(), dtype.StorageBytes());
    field(kDLTensorByteOffset, prim::cast(bytes, PrimType::UInt(64)));
    field(kDLTensorDeviceId, prim::cast(device_id_.value(), PrimType::Int(32)));
    field(kDLTensorDeviceType, prim::cast(device_type_.value(), PrimType::Int(32)));
    return Call(PtrType::VoidPtrType(), abi_field_get_op(),
                {frame, IntImm::Int32(index), IntImm::Int32(kDLTensorAddr)});
  }

  // -----------------------------------------------------------------------------
  // Statement boundaries, expression sequencing, and discarded results
  // -----------------------------------------------------------------------------
  // At each original statement, emit preparation, the rewritten statement, then
  // scheduled last-use drops. Keep ReleaseLastUses beside the statement hook.
  // Check let bodies and conditional arms for unsupported setup, and drop
  // discarded packed results when required.

  // Preparation belongs to the statement that evaluates it, including inside nested statements.
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode mode) final {
    if (!input.as<StmtObj>()) {
      return StmtExprMutator::Mutate(input, mode);
    }
    Stmt original = input.as_or_throw<Stmt>();
    ffi::Array<Stmt> outer = std::exchange(pending_stmts_, {});
    Stmt rewritten =
        StmtExprMutator::Mutate(input, mode).ValueOrUnchanged(input).as_or_throw<Stmt>();
    ffi::Array<Stmt> body = std::exchange(pending_stmts_, std::move(outer));
    body.push_back(rewritten);
    ReleaseLastUses(original, body);
    if (body.size() == 1) return rewritten;
    return SeqStmt(std::move(body), original->loc);
  }

  void ReleaseLastUses(const Stmt& original, ffi::Array<Stmt>& body) {
    ffi::Optional<ffi::List<Var>> ending = drops_after_.Get(original);
    if (!ending) {
      return;
    }
    // A return already transferred or dropped every local that ends here.
    bool exit = original.as<ReturnObj>();
    for (const Var& var : *ending) {
      ManagedFrame& frame = var->ty.as<ObjectTypeObj>() ? object_frame_ : any_frame_;
      int64_t slot = frame.slot_index_map.at(var);
      if (!exit) {
        body.push_back(frame.EmitDrop(slot));
      }
      frame.FreeSlot(slot);
      frame.slot_index_map.erase(var);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::LetObj* op, InplaceMode mode) final {
    PrimExpr value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    auto var = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] { return Mutate(op->var, mode); })
                   .as_or_throw<UnchangedOr<Var>>()
                   .ValueOrUnchanged(op->var);
    size_t before_body = pending_stmts_.size();
    PrimExpr body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    TVM_FFI_CHECK(pending_stmts_.size() == before_body, ValueError)
        << "The body of this `let` expression requires setup statements. Bind the variable in a "
           "separate statement, then evaluate the body.";
    return prim::Let(var, value, body, op->loc);
  }

  Expr MutateIfThenElse(const CallObj* op, InplaceMode mode) {
    Expr condition = Mutate(op->args[0], mode).ValueOrUnchanged(op->args[0]);
    size_t before_arms = pending_stmts_.size();
    Expr yes = Mutate(op->args[1], mode).ValueOrUnchanged(op->args[1]);
    Expr no = Mutate(op->args[2], mode).ValueOrUnchanged(op->args[2]);
    TVM_FFI_CHECK(pending_stmts_.size() == before_arms, ValueError)
        << "The true and false values of `if_then_else` must be free of setup statements. "
           "Use a statement `if` and evaluate each value inside its branch.";
    return Call(op->ty, op->op, {condition, yes, no}, op->attrs, op->ty_args, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj* eval, InplaceMode mode) final {
    TVM_FFI_CHECK(!eval->value->ty.as<AnyTypeObj>() && !eval->value->ty.as<ObjectTypeObj>(),
                  ValueError)
        << "Managed results must be bound to a variable";
    auto call = eval->value.as<CallObj>();
    if (call && (call->op.same_as(call_packed_op()) || call->op.same_as(call_cpacked_op()))) {
      // An unannotated packed call has IR type int32 even when its callback returns an
      // object. Drop the tagged result to release an object returned through that form.
      auto scalar = call->ty.as<PrimType>();
      if ((scalar && scalar.value() != PrimType::Int(32) && !scalar.value().IsVoid()) ||
          call->ty.as<PtrTypeObj>()) {
        return Evaluate(EmitPackedCall(call, PrimType::Void()));
      }
      return CallAndDiscard(call);
    }
    if (call && PackedCallBuilder::IsVoidOrNoneType(call->ty)) {
      Call rewritten = Rewrite(eval->value).as_or_throw<Call>();
      return Evaluate(Call(PrimType::Void(), rewritten->op, rewritten->args, rewritten->attrs,
                           rewritten->ty_args, rewritten->loc));
    }
    return StmtExprMutator::Mutate_(eval, mode);
  }

  Stmt CallAndDiscard(const CallObj* call) {
    Call invoke = EmitPackedCall(call, PrimType::Void());
    constexpr size_t kResultFrame = 4;
    constexpr size_t kResultSlot = 5;
    Stmt drop = Evaluate(Call(PrimType::Int(32), drop_any_slot_op(),
                              {invoke->args[kResultFrame], invoke->args[kResultSlot]}));
    return SeqStmt({Evaluate(invoke), drop});
  }

  // -----------------------------------------------------------------------------
  // Scoped statements
  // -----------------------------------------------------------------------------
  // If opens a scope for each arm. For and While own their loop-body scopes.
  // For places parallel scratch inside its body. RegionStmt applies lexical
  // device context while rewriting its body. Use default SeqStmt traversal.

  UnchangedOr<Stmt> Mutate_(const IfObj* op, InplaceMode mode) final {
    PrimExpr condition = Mutate(op->condition, mode).ValueOrUnchanged(op->condition);
    SeqStmt then_case = scopes_.WithNewScope(
        [&] { return Mutate(op->then_case, mode).ValueOrUnchanged(op->then_case); });
    ffi::Optional<SeqStmt> else_case;
    if (op->else_case) {
      const SeqStmt& arm = op->else_case.value();
      else_case = scopes_.WithNewScope([&] { return Mutate(arm, mode).ValueOrUnchanged(arm); });
    }
    return If(condition, then_case, else_case, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode mode) final {
    PrimExpr begin = Mutate(loop->begin, mode).ValueOrUnchanged(loop->begin);
    PrimExpr end = Mutate(loop->end, mode).ValueOrUnchanged(loop->end);
    PrimExpr step = Mutate(loop->step, mode).ValueOrUnchanged(loop->step);
    SeqStmt body = scopes_.WithNewScope([&] {
      scopes_.Current().is_loop = true;
      if (loop->kind != ForKind::kParallel) {
        return Mutate(loop->body, mode).ValueOrUnchanged(loop->body);
      }
      auto outer = std::exchange(packed_call_builder_, PackedCallBuilder{});
      SeqStmt inner = Mutate(loop->body, mode).ValueOrUnchanged(loop->body);
      ffi::Array<Stmt> prefix;
      packed_call_builder_.PrependAllocations(&prefix);
      packed_call_builder_ = std::move(outer);
      prefix.push_back(inner);
      return SeqStmt(std::move(prefix), inner->loc);
    });
    return For(loop->var, begin, end, step, loop->kind, body, loop->annotations, loop->loc);
  }

  UnchangedOr<Stmt> Mutate_(const WhileObj* loop, InplaceMode mode) final {
    PrimExpr condition = Mutate(loop->condition, mode).ValueOrUnchanged(loop->condition);
    TVM_FFI_CHECK(pending_stmts_.empty(), ValueError)
        << "The `while` condition requires setup statements. Use `while True`, evaluate the "
           "condition at the top of the loop, and `break` when it is false.";
    SeqStmt body = scopes_.WithNewScope([&] {
      scopes_.Current().is_loop = true;
      return Mutate(loop->body, mode).ValueOrUnchanged(loop->body);
    });
    return While(condition, body, loop->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* region, InplaceMode mode) final {
    auto args = Mutate(region->args, mode)
                    .as_or_throw<UnchangedOr<ffi::Array<Expr>>>()
                    .ValueOrUnchanged(region->args);
    if (region->op.same_as(device_context_op())) {
      auto type = std::exchange(device_type_, args[0].as_or_throw<PrimExpr>());
      auto id = std::exchange(device_id_, args[1].as_or_throw<PrimExpr>());
      SeqStmt body = scopes_.WithNewScope(
          [&] { return Mutate(region->body, mode).ValueOrUnchanged(region->body); });
      device_id_ = id;
      device_type_ = type;
      return body;
    }
    auto attrs = Mutate(region->attrs, mode)
                     .as_or_throw<UnchangedOr<DictAttrs>>()
                     .ValueOrUnchanged(region->attrs);
    SeqStmt body = scopes_.WithNewScope(
        [&] { return Mutate(region->body, mode).ValueOrUnchanged(region->body); });
    return RegionStmt(region->op, args, region->body_params, attrs, body, region->result_vars,
                      region->loc);
  }

  // -----------------------------------------------------------------------------
  // Returns and loop exits
  // -----------------------------------------------------------------------------
  // Return writes or transfers its result before dropping remaining owners.
  // Break and Continue unwind through the target loop. Keep EmitScopeExitDrops
  // beside these three handlers.

  // For "x = f(); y = g(); return y", move y to the caller, clear its slot, and drop x
  // if it is still live on this path.
  UnchangedOr<Stmt> Mutate_(const ReturnObj* ret, InplaceMode) final {
    ffi::Array<Stmt> body;
    if (auto call = ret->value.as<CallObj>(); call && call->op.same_as(store_return_op())) {
      Var destination = call->args[0].as_or_throw<Var>();
      const Expr& value = call->args[1];
      ffi::Optional<Var> transferred;
      if (value->ty.as<AnyTypeObj>() || value->ty.as<ObjectTypeObj>()) {
        auto var = value.as<Var>();
        TVM_FFI_CHECK(var, ValueError) << "Managed return expressions must be bound variables";
        ManagedFrame& frame = value->ty.as<ObjectTypeObj>() ? object_frame_ : any_frame_;
        ffi::Optional<int64_t> local_slot = frame.slot_index_map.Get(var.value());
        int64_t slot;
        if (!local_slot) {
          slot = frame.AllocSlot();
          frame.EmitRetain(body, slot, value);
        } else {
          slot = *local_slot;
          transferred = var;
        }
        PackedCallBuilder::StorePackedValue(&body, destination, 0, frame.Read(value->ty, slot));
        frame.Clear(body, slot);
        if (!transferred) {
          frame.FreeSlot(slot);
        }
      } else if (value.as<StringImmObj>()) {
        body.push_back(Evaluate(
            Call(PrimType::Int(32), retain_any_slot_op(), {destination, IntImm::Int64(0), value})));
      } else if (PackedCallBuilder::IsVoidOrNoneType(value->ty)) {
        if (!value.as<ConstantObj>() && !value.as<VarObj>()) {
          Stmt effect = Evaluate(value);
          body.push_back(Mutate(effect).ValueOrUnchanged(effect));
        }
        PackedCallBuilder::StorePackedValue(&body, destination, 0,
                                            GenericConst(NoneType(), nullptr));
      } else {
        PackedCallBuilder::StorePackedValue(&body, destination, 0, Rewrite(value));
      }
      EmitScopeExitDrops(body, /*stop_at_loop=*/false, transferred);
      body.push_back(Return(IntImm::Int32(0)));
    } else {
      TVM_FFI_CHECK(!ret->value->ty.as<AnyTypeObj>() && !ret->value->ty.as<ObjectTypeObj>(),
                    ValueError)
          << "Managed returns require packed calling-convention lowering";
      Expr value = Rewrite(ret->value);
      if ((!object_frame_.slot_index_map.empty() || !any_frame_.slot_index_map.empty()) &&
          !value.as<IntImmObj>()) {
        // Evaluate the result while all its inputs are still live.
        Var saved("return_value", value->ty);
        body.push_back(Bind(saved, value));
        value = saved;
      }
      EmitScopeExitDrops(body, /*stop_at_loop=*/false);
      body.push_back(Return(value));
    }
    return SeqStmt(std::move(body));
  }

  // In "x = f(); while cond: y = g(); break/continue", the exit drops a live y;
  // x belongs to the outer scope and survives the loop exit.
  UnchangedOr<Stmt> Mutate_(const BreakObj* stmt, InplaceMode) final {
    ffi::Array<Stmt> body;
    EmitScopeExitDrops(body, /*stop_at_loop=*/true);
    body.push_back(ffi::GetRef<Stmt>(stmt));
    return SeqStmt(std::move(body));
  }

  UnchangedOr<Stmt> Mutate_(const ContinueObj* stmt, InplaceMode) final {
    ffi::Array<Stmt> body;
    EmitScopeExitDrops(body, /*stop_at_loop=*/true);
    body.push_back(ffi::GetRef<Stmt>(stmt));
    return SeqStmt(std::move(body));
  }

  // Emit path-specific drops in reverse definition order. Sibling paths keep their bindings.
  void EmitScopeExitDrops(ffi::Array<Stmt>& body, bool stop_at_loop,
                          ffi::Optional<Var> returned = std::nullopt) {
    for (auto scope = scopes_.rbegin(); scope != scopes_.rend(); ++scope) {
      for (auto it = scope->definitions.rbegin(); it != scope->definitions.rend(); ++it) {
        const Var& var = *it;
        if (returned && var.same_as(returned.value())) {
          continue;
        }
        ManagedFrame& frame = var->ty.as<ObjectTypeObj>() ? object_frame_ : any_frame_;
        if (ffi::Optional<int64_t> slot = frame.slot_index_map.Get(var)) {
          body.push_back(frame.EmitDrop(*slot));
        }
      }
      if (stop_at_loop && scope->is_loop) {
        return;
      }
    }
  }
};
}  // namespace

namespace transform {
/*!
 * \brief Create the function pass that lowers the host packed ABI.
 * \return A pass that leaves declarations and non-host device definitions unchanged and lowers
 *   eligible host bodies.
 */
Pass LowerFFIABI() {
  auto pass = [](Function func, IRModule, PassContext) -> ffi::Optional<Function> {
    if (func->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      return func;
    }
    if (ffi::Optional<Target> target = func->GetAttr(tvm::attr::kTarget);
        target && target.value()->GetTargetDeviceType() != kDLCPU &&
        !func->HasNonzeroAttr(tirx::attr::kIsHostFunc.name)) {
      return func;
    }
    return ffi::make_object<ABIRewriter>()->Run(std::move(func));
  };
  return CreateFunctionPass(pass, 0, "tirx.LowerFFIABI");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tirx.transform.LowerFFIABI", LowerFFIABI);
}
}  // namespace transform
}  // namespace tirx
}  // namespace tvm
