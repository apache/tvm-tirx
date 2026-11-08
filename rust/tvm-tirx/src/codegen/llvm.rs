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

//! The LLVM backend. Queries of LLVM intrinsics, the host CPU and the CPUs and features of LLVM
//! targets, which `libtvm_tirx_llvm_backend` registers when it is loaded. Target-dependent
//! queries require an explicit target.

// tvm-ffi-stubgen(file): mod(missing_docs = allow)
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import): tvm_ffi::Array
// tvm-ffi-stubgen(import): tvm_ffi::Map
// tvm-ffi-stubgen(import): tvm_ffi::Result
// tvm-ffi-stubgen(import): tvm_ffi::String
// tvm-ffi-stubgen(import): crate::ir::expr::IntImm
// tvm-ffi-stubgen(import): crate::target::Target

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::IntImm;
use crate::target::Target;
use tvm_ffi::Array;
use tvm_ffi::Map;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// The LLVM backend registers these functions only when TIRx is built with LLVM: they are looked up
// when called, and their absence is an error.

/// Return the id of the LLVM intrinsic `name`, or 0 when there is none
/// (`codegen.llvm_lookup_intrinsic_id`).
pub fn lookup_intrinsic_id(name: &str) -> Result<i64> {
    tvm_ffi::Function::get_global("codegen.llvm_lookup_intrinsic_id")?
        .call_tuple((String::from(name),))?
        .try_into()
}

/// Return the name of the LLVM intrinsic `intrin_id` (`codegen.llvm_get_intrinsic_name`).
///
/// # Errors
///
/// Returns an error if `intrin_id` is not the id of an intrinsic.
pub fn get_intrinsic_name(intrin_id: i64) -> Result<String> {
    tvm_ffi::Function::get_global("codegen.llvm_get_intrinsic_name")?
        .call_tuple((intrin_id,))?
        .try_into()
}

/// Return the vendor of the host x86 CPU: `intel`, `amd`, `unknown`, or `unimplemented` on other
/// CPUs (`codegen.llvm_get_system_x86_vendor`).
pub fn get_system_x86_vendor() -> Result<String> {
    tvm_ffi::Function::get_global("codegen.llvm_get_system_x86_vendor")?.call_tuple(())?.try_into()
}

/// Return the default LLVM target triple of the host (`codegen.llvm_get_system_triple`).
pub fn get_system_triple() -> Result<String> {
    tvm_ffi::Function::get_global("codegen.llvm_get_system_triple")?.call_tuple(())?.try_into()
}

/// Return the LLVM name of the host CPU (`codegen.llvm_get_system_cpu`).
pub fn get_system_cpu() -> Result<String> {
    tvm_ffi::Function::get_global("codegen.llvm_get_system_cpu")?.call_tuple(())?.try_into()
}

/// Return the targets LLVM was built with (`codegen.llvm_get_targets`).
pub fn get_targets() -> Result<Array<String>> {
    tvm_ffi::Function::get_global("codegen.llvm_get_targets")?.call_tuple(())?.try_into()
}

/// Return the CPUs of the `-mtriple` of `target` (`codegen.llvm_get_cpu_archlist`).
pub fn get_cpu_archlist(target: &Target) -> Result<Array<String>> {
    tvm_ffi::Function::get_global("codegen.llvm_get_cpu_archlist")?
        .call_tuple((target,))?
        .try_into()
}

/// Return the CPU features of the `-mtriple`, `-mcpu` and `-mattr` of `target`
/// (`codegen.llvm_get_cpu_features`).
pub fn get_cpu_features(target: &Target) -> Result<Map<String, String>> {
    tvm_ffi::Function::get_global("codegen.llvm_get_cpu_features")?
        .call_tuple((target,))?
        .try_into()
}

/// Return whether the CPU of `target` has the feature `feature` (`codegen.llvm_cpu_has_feature`).
pub fn cpu_has_feature(feature: &str, target: &Target) -> Result<bool> {
    tvm_ffi::Function::get_global("codegen.llvm_cpu_has_feature")?
        .call_tuple((String::from(feature), target))?
        .try_into()
}

/// Return the vector width of the CPU of `target` in bits, or -1 for a target that is not `llvm`
/// (`codegen.llvm_get_vector_width`).
///
/// # Examples
///
/// ```
/// use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
/// use tvm_tirx::codegen::llvm;
/// use tvm_tirx::target::Target;
/// use tvm_tirx::tvm_ffi::Module;
///
/// let backend = format!("{DLL_PREFIX}tvm_tirx_llvm_backend{DLL_SUFFIX}");
/// let _backend = Module::load_from_file(backend)?;
/// let cpu = llvm::get_system_cpu()?;
/// let host = Target::new(&format!(r#"{{"kind": "llvm", "mcpu": "{}"}}"#, cpu.as_str()))?;
/// assert!(llvm::get_vector_width(&host)? > 0);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub fn get_vector_width(target: &Target) -> Result<i32> {
    tvm_ffi::Function::get_global("codegen.llvm_get_vector_width")?
        .call_tuple((target,))?
        .try_into()
}

/// Return whether LLVM knows the CPU `cpu` for the target triple `triple`
/// (`codegen.llvm_is_valid_cpu`).
pub fn is_valid_cpu(cpu: &str, triple: &str) -> Result<bool> {
    tvm_ffi::Function::get_global("codegen.llvm_is_valid_cpu")?
        .call_tuple((String::from(cpu), String::from(triple)))?
        .try_into()
}

/// Return the major version of LLVM (`codegen.llvm_version_major`).
pub fn version_major() -> Result<i32> {
    tvm_ffi::Function::get_global("codegen.llvm_version_major")?.call_tuple(())?.try_into()
}

/// Return whether LLVM can create a target machine for the target string `target`, such as
/// `"llvm -mtriple=aarch64-linux-gnu"` (`codegen.llvm_target_enabled`).
pub fn target_enabled(target: &str) -> Result<bool> {
    tvm_ffi::Function::get_global("codegen.llvm_target_enabled")?
        .call_tuple((String::from(target),))?
        .try_into()
}

/// Return the default LLVM target triple of the host (`tvm.codegen.llvm.GetDefaultTargetTriple`).
pub fn get_default_target_triple() -> Result<String> {
    tvm_ffi::Function::get_global("tvm.codegen.llvm.GetDefaultTargetTriple")?
        .call_tuple(())?
        .try_into()
}

/// Return the LLVM target triple of the running process (`tvm.codegen.llvm.GetProcessTriple`).
pub fn get_process_triple() -> Result<String> {
    tvm_ffi::Function::get_global("tvm.codegen.llvm.GetProcessTriple")?.call_tuple(())?.try_into()
}

/// Return the LLVM name of the host CPU (`tvm.codegen.llvm.GetHostCPUName`).
pub fn get_host_cpu_name() -> Result<String> {
    tvm_ffi::Function::get_global("tvm.codegen.llvm.GetHostCPUName")?.call_tuple(())?.try_into()
}

/// Return the features of the host CPU, each with a boolean saying whether it is enabled, or none
/// when LLVM cannot tell (`tvm.codegen.llvm.GetHostCPUFeatures`).
pub fn get_host_cpu_features() -> Result<Map<String, IntImm>> {
    tvm_ffi::Function::get_global("tvm.codegen.llvm.GetHostCPUFeatures")?.call_tuple(())?.try_into()
}
