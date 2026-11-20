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
 * \file target_kind.cc
 * \brief ROCm compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <cctype>
#include <string>

namespace tvm {

namespace backend {
namespace rocm {

std::string ExtractStringWithPrefix(const std::string& str, const std::string& prefix) {
  if (str.find(prefix) != 0) return "";
  std::size_t pos = prefix.length();
  while (pos < str.length() && (std::isdigit(str[pos]) || std::isalpha(str[pos]))) {
    ++pos;
  }
  return str.substr(prefix.length(), pos - prefix.length());
}

bool DetectDeviceFlag(Device device, const ffi::String& name, ffi::Any* val) {
  using runtime::DeviceAPI;
  DeviceAPI* api = DeviceAPI::Get(device, true);
  if (api == nullptr) {
    return false;
  }
  bool exists = api->GetAttr(device, "exist").cast<bool>();
  if (!exists) {
    return false;
  }
  *val = api->GetAttr(device, name);
  return val->type_index() != kTVMFFINone;
}

void CheckOrSetAttr(ffi::Map<ffi::String, ffi::Any>* attrs, const ffi::String& name,
                    const ffi::String& value) {
  auto iter = attrs->find(name);
  if (iter == attrs->end()) {
    attrs->Set(name, value);
  } else {
    auto str = (*iter).second.try_cast<ffi::String>();
    TVM_FFI_CHECK(str && str.value() == value, ValueError)
        << "Expects \"" << name << "\" to be \"" << value << "\", but gets: " << (*iter).second;
  }
}

ffi::Map<ffi::String, ffi::Any> UpdateROCmAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  CheckOrSetAttr(&target, "mtriple", "amdgcn-amd-amdhsa-hcc");
  std::string arch = "gfx900";
  if (target.count("mcpu")) {
    ffi::String mcpu = target.at("mcpu").as_or_throw<ffi::String>();
    arch = ExtractStringWithPrefix(mcpu, "gfx");
    TVM_FFI_CHECK(!arch.empty(), ValueError)
        << "ROCm target gets an invalid GFX version: -mcpu=" << mcpu;
  } else {
    ffi::Any val;
    if (const auto f_get_rocm_arch = tvm::ffi::Function::GetGlobal("tvm_callback_rocm_get_arch")) {
      arch = (*f_get_rocm_arch)().cast<std::string>();
    }
    target.Set("mcpu", ffi::String(arch));
  }

  // RDNA (gfx10xx, gfx11xx, gfx12xx) executes wave32; GCN/CDNA (gfx9xx) wave64. Respect an
  // explicit thread_warp_size.
  if (!target.count("thread_warp_size")) {
    std::string ver = arch.rfind("gfx", 0) == 0 ? arch.substr(3) : arch;
    bool rdna = ver.rfind("10", 0) == 0 || ver.rfind("11", 0) == 0 || ver.rfind("12", 0) == 0;
    target.Set("thread_warp_size", static_cast<int64_t>(rdna ? 32 : 64));
  }

  ffi::Any val;
  int version;
  if (!DetectDeviceFlag({kDLROCM, 0}, "api_version", &val)) {
    LOG(WARNING) << "Unable to detect ROCm version, assuming >= 3.5";
    version = 305;
  } else {
    version = val.cast<int>();
  }
  if (version < 305) {
    ffi::Array<ffi::String> mattr;
    if (target.count("mattr")) {
      mattr = target.at("mattr").as_or_throw<ffi::Array<ffi::String>>();
    }
    mattr.push_back("-code-object-v3");
    target.Set("mattr", mattr);
  }
  return target;
}

}  // namespace rocm
}  // namespace backend

}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace tvm;
  using namespace tvm::backend::rocm;
  namespace refl = tvm::ffi::reflection;

  TargetKindDef("rocm")
      .set_default_device_type(kDLROCM)
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("mtriple")
      .def_option<ffi::Array<ffi::String>>("mattr")
      .def_option<int64_t>("max_num_threads", refl::DefaultValue(256))
      .def_option<int64_t>("max_threads_per_block", refl::DefaultValue(256))
      .def_option<int64_t>("max_shared_memory_per_block", refl::DefaultValue(65536))
      // Set by UpdateROCmAttrs from mcpu when not given: 32 on RDNA (gfx10/11/12), else 64.
      .def_option<int64_t>("thread_warp_size")
      .set_default_keys({"rocm", "gpu"})
      .set_target_canonicalizer(UpdateROCmAttrs);
}
