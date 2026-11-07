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
 * \file tvm/codegen/llvm/canonicalizer/arm_aprofile.cc
 * \brief Target canonicalizer for Arm(R) Cortex(R) A-Profile CPUs
 */

#include "arm_aprofile.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <memory>
#include <string>

#include "../llvm_instance.h"

namespace tvm {
namespace target {
namespace canonicalizer {
namespace llvm {
namespace aprofile {

bool IsAArch32(ffi::Optional<ffi::String> mtriple, ffi::Optional<ffi::String> mcpu) {
  if (mtriple) {
    bool is_mprofile = mcpu && mcpu.value().starts_with("cortex-m");
    return mtriple.value().starts_with("arm") && !is_mprofile;
  }
  return false;
}

bool IsAArch64(ffi::Optional<ffi::String> mtriple) {
  if (mtriple) {
    return mtriple.value().starts_with("aarch64");
  }
  return false;
}

bool IsArch(ffi::Map<ffi::String, ffi::Any> attrs) {
  ffi::Optional<ffi::String> mtriple =
      attrs.Get("mtriple").value_or(nullptr).as_or_throw<ffi::Optional<ffi::String>>();
  ffi::Optional<ffi::String> mcpu =
      attrs.Get("mcpu").value_or(nullptr).as_or_throw<ffi::Optional<ffi::String>>();

  return IsAArch32(mtriple, mcpu) || IsAArch64(mtriple);
}

bool CheckContains(ffi::Array<ffi::String> array, ffi::String predicate) {
  return std::any_of(array.begin(), array.end(), [&](ffi::String var) { return var == predicate; });
}

static ffi::Map<ffi::String, ffi::Any> GetFeatures(ffi::Map<ffi::String, ffi::Any> target) {
#ifdef TVM_LLVM_VERSION
  ffi::String kind = target.Get("kind").value().as_or_throw<ffi::String>();
  TVM_FFI_ICHECK_EQ(kind, "llvm") << "Expected target kind 'llvm', but got '" << kind << "'";

  ffi::Optional<ffi::String> mtriple =
      target.Get("mtriple").value_or(nullptr).as_or_throw<ffi::Optional<ffi::String>>();
  ffi::Optional<ffi::String> mcpu =
      target.Get("mcpu").value_or(nullptr).as_or_throw<ffi::Optional<ffi::String>>();

  // Check that LLVM has been compiled with the correct target support
  auto llvm_instance = std::make_unique<codegen::LLVMInstance>();
  codegen::LLVMTargetInfo llvm_backend(*llvm_instance, {{"kind", ffi::String("llvm")}});
  ffi::Array<ffi::String> targets = llvm_backend.GetAllLLVMTargets();
  if ((IsAArch64(mtriple) && !CheckContains(targets, "aarch64")) ||
      (IsAArch32(mtriple, mcpu) && !CheckContains(targets, "arm"))) {
    LOG(WARNING) << "Cannot parse target features for target: " << target
                 << ". LLVM was not compiled with support for Arm(R)-based targets.";
    return {};
  }

  codegen::LLVMTargetInfo llvm_target(*llvm_instance, target);
  ffi::Map<ffi::String, ffi::String> features = llvm_target.GetAllLLVMCpuFeatures();

  auto has_feature = [features](const ffi::String& feature) {
    return features.find(feature) != features.end();
  };

  return {{"feature.is_aarch64", static_cast<bool>(IsAArch64(mtriple))},
          {"feature.has_asimd", static_cast<bool>(has_feature("neon"))},
          {"feature.has_sve", static_cast<bool>(has_feature("sve"))},
          {"feature.has_dotprod", static_cast<bool>(has_feature("dotprod"))},
          {"feature.has_matmul_i8", static_cast<bool>(has_feature("i8mm"))},
          {"feature.has_fp16_simd", static_cast<bool>(has_feature("fullfp16"))},
          {"feature.has_sme", static_cast<bool>(has_feature("sme"))}};
#endif

  LOG(WARNING) << "Cannot parse Arm(R)-based target features for target " << target
               << " without LLVM support.";
  return {};
}

static ffi::Array<ffi::String> MergeKeys(ffi::Optional<ffi::Array<ffi::String>> existing_keys) {
  const ffi::Array<ffi::String> kExtraKeys = {"arm_cpu", "cpu"};

  if (!existing_keys) {
    return kExtraKeys;
  }

  ffi::Array<ffi::String> keys = existing_keys.value();
  for (ffi::String key : kExtraKeys) {
    if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
      keys.push_back(key);
    }
  }
  return keys;
}

ffi::Map<ffi::String, ffi::Any> Canonicalize(ffi::Map<ffi::String, ffi::Any> target) {
  ffi::Map<ffi::String, ffi::Any> features = GetFeatures(target);
  for (const auto& kv : features) {
    target.Set(kv.first, kv.second);
  }
  target.Set("keys", MergeKeys((target.Get("keys"))
                                   .value_or(nullptr)
                                   .as_or_throw<ffi::Optional<ffi::Array<ffi::String>>>()));

  return target;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  auto canonicalize = [](ffi::Map<ffi::String, ffi::Any> attrs) {
    // Cortex-M handling remains in the core and does not require LLVM.
    auto default_canonicalize =
        ffi::Function::GetGlobal("codegen.llvm.canonicalize_target.default");
    attrs = (*default_canonicalize)(attrs).cast<ffi::Map<ffi::String, ffi::Any>>();
    return IsArch(attrs) ? Canonicalize(attrs) : attrs;
  };
  // Preserve the Arm architecture spellings accepted by the existing parser.
  for (const char* arch :
       {"aarch64",     "aarch64_be",  "aarch64_32", "arm",      "arm64",         "armeb",
        "armv4",       "armv4t",      "armv5",      "armv5t",   "armv5te",       "armv5tej",
        "armv6",       "armv6k",      "armv6t2",    "armv6kz",  "armv6m",        "armv7",
        "armv7a",      "armv7l",      "armv7ve",    "armv7r",   "armv7m",        "armv7em",
        "armv7s",      "armv7k",      "armv8",      "armv8a",   "armv8l",        "armv8r",
        "armv8m.base", "armv8m.main", "armv8.1",    "armv8.1a", "armv8.1m.main", "armv8.2",
        "armv8.2a",    "armv8.3",     "armv8.3a",   "armv8.4",  "armv8.4a",      "armv8.5",
        "armv8.5a",    "armv8.6",     "armv8.6a",   "armv8.7",  "armv8.7a",      "armv8.8",
        "armv8.8a",    "armv8.9",     "armv8.9a",   "armv9",    "armv9a",        "armv9.1",
        "armv9.1a",    "armv9.2",     "armv9.2a",   "armv9.3",  "armv9.3a",      "armv9.4",
        "armv9.4a",    "armv9.5",     "armv9.5a",   "armv9.6",  "armv9.6a",      "armv9.7",
        "armv9.7a"}) {
    std::string name = std::string("codegen.llvm.canonicalize_target.") + arch;
    ffi::reflection::GlobalDef().def(name.c_str(), canonicalize);
  }
}

}  // namespace aprofile
}  // namespace llvm
}  // namespace canonicalizer
}  // namespace target
}  // namespace tvm
