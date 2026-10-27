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

#include <gtest/gtest.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>

#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

namespace tvm {
namespace {

class ScriptExposureTypeObj : public TypeObj {
 public:
  int64_t value{0};
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.ScriptExposureType", ScriptExposureTypeObj, TypeObj);
};

class OtherScriptExposureTypeObj : public TypeObj {
 public:
  int64_t value{0};
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.OtherScriptExposureType", OtherScriptExposureTypeObj,
                                    TypeObj);
};

static_assert(std::is_same_v<decltype(std::declval<const OpDef&>().expose_script()), void>);
static_assert(std::is_same_v<decltype(ExposeScriptType<ScriptExposureTypeObj>()), void>);

struct ExposureState {
  int op_calls{0};
  int type_calls{0};
  bool fail_op{false};
  bool fail_type{false};
  bool reenter_op{false};
  bool reenter_type{false};
};

TEST(ScriptExposure, OptionalReceiversAndCompletedDefinitions) {
  namespace refl = ffi::reflection;
  // This executable loads the compiler without Python. Static registration must
  // succeed without receivers, and missing lookups must remain retryable.
  ASSERT_FALSE(ffi::Function::GetGlobal("script.expose_script_op").has_value());
  ASSERT_FALSE(ffi::Function::GetGlobal("script.expose_script_type").has_value());
  OpDef absent("test.script_exposure.absent");
  absent.signature();
  EXPECT_NO_THROW(ExposeScriptOp(absent.op()->name));
  EXPECT_NO_THROW(absent.expose_script());

  refl::ObjectDef<ScriptExposureTypeObj>().def_ro("value", &ScriptExposureTypeObj::value);
  EXPECT_NO_THROW(ExposeScriptType(ScriptExposureTypeObj::_type_key));
  EXPECT_NO_THROW(ExposeScriptType<ScriptExposureTypeObj>());

  // The successful receiver is retained for the process lifetime. Its captures
  // must therefore outlive this test, including when another test exposes an op.
  auto state = std::make_shared<ExposureState>();
  ffi::Function::SetGlobal("script.expose_script_op",
                           ffi::Function::FromTyped([state](ffi::String name) {
                             if (name != "test.script_exposure.complete") return;
                             ++state->op_calls;
                             TVM_FFI_CHECK(!state->fail_op, ValueError) << "op exposure failed";
                             Op op = Op::Get(name);
                             EXPECT_EQ(op->doc, "A completed operator.");
                             ASSERT_EQ(op->args_info.size(), 1);
                             EXPECT_EQ(op->args_info[0]->name, "value");
                             EXPECT_EQ(op->args_info[0]->doc, "The input value.");
                             ASSERT_EQ(op->ty_args_info.size(), 1);
                             EXPECT_EQ(op->ty_args_info[0]->name, "T");
                             EXPECT_TRUE(op->validator != nullptr);
                             EXPECT_TRUE(Op::GetAttrMap<bool>("test.script_exposure.complete")[op]);
                             if (std::exchange(state->reenter_op, false)) ExposeScriptOp(name);
                           }));
  ffi::Function::SetGlobal(
      "script.expose_script_type", ffi::Function::FromTyped([state](ffi::String key) {
        if (key != ScriptExposureTypeObj::_type_key &&
            key != OtherScriptExposureTypeObj::_type_key) {
          return;
        }
        ++state->type_calls;
        TVM_FFI_CHECK(!state->fail_type, ValueError) << "type exposure failed";
        std::string_view type_key(key.data(), key.size());
        int32_t index = ffi::TypeKeyToIndex(type_key);
        refl::FieldGetter value(type_key, "value");
        refl::TypeAttrColumn initializers(ffi::type_attr::kInit);
        refl::TypeAttrColumn creators(ffi::type_attr::kNew);
        refl::TypeAttrColumn finality(ffi::type_attr::kTypeFinal);
        // ObjectDef finalizes these attributes in its destructor, after fields.
        EXPECT_TRUE(creators[index].as<ffi::Function>().has_value());
        EXPECT_TRUE(finality[index].cast<bool>());
        Type instance = initializers[index].cast<ffi::Function>()(int64_t{7}).cast<Type>();
        EXPECT_EQ(value(instance).cast<int64_t>(), 7);
        if (std::exchange(state->reenter_type, false)) ExposeScriptType(key);
      }));

  // Exposure is explicit and happens only after the signature and attributes.
  OpDef complete("test.script_exposure.complete", "A completed operator.");
  EXPECT_EQ(state->op_calls, 0);
  complete.signature(sig::arg("value", "The input value."), sig::ty_arg("T"))
      .set_attr<bool>("test.script_exposure.complete", true)
      .expose_script();
  EXPECT_EQ(state->op_calls, 1);
  ExposeScriptType<ScriptExposureTypeObj>();
  EXPECT_EQ(state->type_calls, 1);

  // A definition registered after receiver installation sees complete metadata
  // too. Its first template exposure below must share the string helper's cache.
  refl::ObjectDef<OtherScriptExposureTypeObj>().def_ro("value", &OtherScriptExposureTypeObj::value);
  ExposeScriptType(OtherScriptExposureTypeObj::_type_key);
  EXPECT_EQ(state->type_calls, 2);

  state->reenter_op = true;
  state->reenter_type = true;
  complete.expose_script();
  ExposeScriptType<ScriptExposureTypeObj>();
  EXPECT_EQ(state->op_calls, 3);
  EXPECT_EQ(state->type_calls, 4);

  state->fail_op = true;
  state->fail_type = true;
  EXPECT_THROW(complete.expose_script(), ffi::Error);
  EXPECT_THROW(ExposeScriptType<ScriptExposureTypeObj>(), ffi::Error);
  state->fail_op = false;
  state->fail_type = false;

  // Removing registry ownership cannot invalidate the retained receiver. A new
  // template instantiation must use the same cached function as earlier calls.
  ffi::Function::RemoveGlobal("script.expose_script_op");
  ffi::Function::RemoveGlobal("script.expose_script_type");
  ExposeScriptOp(complete.op()->name);
  complete.expose_script();
  ExposeScriptType(ScriptExposureTypeObj::_type_key);
  ExposeScriptType<ScriptExposureTypeObj>();
  ExposeScriptType<OtherScriptExposureTypeObj>();
  EXPECT_EQ(state->op_calls, 6);
  EXPECT_EQ(state->type_calls, 8);
}

}  // namespace
}  // namespace tvm
