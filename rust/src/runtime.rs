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

//! The runtime objects the IR refers to, tensor allocation and runtime queries.
//!
//! A [`MemoryScope`] is the storage scope of a [`PtrType`](crate::ir::PtrType).
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::dtype;
//! use tvm_tirx::runtime;
//! use tvm_tirx::tvm_ffi::{DLDevice, DLDeviceType};
//!
//! let cpu = DLDevice::new(DLDeviceType::kDLCPU, 0);
//! let buffer = runtime::empty(&[2, 8], dtype::float(32), cpu, None)?;
//!
//! let bytes = runtime::create_view(&buffer, &[64], dtype::int(8), 0)?;
//! assert_eq!(bytes.shape(), &[64]);
//!
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// tvm-ffi-stubgen(missing-docs): allow
// No `prefix`: only the objects with a block below are bound, not all of `runtime`.
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataType
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDevice
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::Shape
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): tvm_ffi::Tensor

// tvm-ffi-stubgen(begin): import-section
use tvm_ffi::Any;
use tvm_ffi::DLDataType;
use tvm_ffi::DLDevice;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::Shape;
use tvm_ffi::String;
use tvm_ffi::Tensor;

/// Header-only stand-in for the builtin `ffi.Enum`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.Enum"]
struct FfiEnumObj {
    base_: Object,
}
// tvm-ffi-stubgen(end)

// `ir.PtrType.storage_scope` refers to this type.
// tvm-ffi-stubgen(begin): object/runtime.MemoryScope
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct MemoryScope {
    data_: ObjectArc<MemoryScopeObj>,
}

/// The object behind [`MemoryScope`].
///
/// Opaque: parent 'ffi.Enum' is opaque (no-mirror). Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "runtime.MemoryScope"]
#[type_final]
pub struct MemoryScopeObj {
    base_: FfiEnumObj,
}

unsafe impl tvm_ffi::object::ImmutableObject for MemoryScopeObj {}
// tvm-ffi-stubgen(end)

impl MemoryScope {
    /// Register the memory scope `name` (`runtime.MemoryScopeRegister`).
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is empty, contains a null character, or is already registered.
    pub fn register(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("runtime.MemoryScopeRegister")
            .call_tuple((String::from(name),))?
            .try_into()
    }
}

/// Allocate an uninitialized tensor on `device`, in the memory scope `mem_scope` or the global one
/// (`runtime.TensorAllocWithScope`).
///
/// # Errors
///
/// Returns an error if the allocator of `device` does not support `mem_scope`; the CPU one supports
/// only the global scope.
pub fn empty(
    shape: &[i64],
    dtype: DLDataType,
    device: DLDevice,
    mem_scope: Option<&MemoryScope>,
) -> Result<Tensor> {
    tvm_ffi::cached_global_func!("runtime.TensorAllocWithScope")
        .call_tuple((Shape::from(shape), dtype, device, mem_scope.cloned()))?
        .try_into()
}

/// Create a view of the storage of `source`, `relative_byte_offset` bytes past its data, with
/// another shape and data type (`runtime.TensorCreateView`).
///
/// # Errors
///
/// Returns an error if `source` is not contiguous or the view does not fit in its storage.
pub fn create_view(
    source: &Tensor,
    shape: &[i64],
    dtype: DLDataType,
    relative_byte_offset: u64,
) -> Result<Tensor> {
    tvm_ffi::cached_global_func!("runtime.TensorCreateView")
        .call_tuple((source, Shape::from(shape), dtype, relative_byte_offset))?
        .try_into()
}

/// Return the attribute `name` of `device`, such as `exist` or `max_threads_per_block`, or `None`
/// when the device does not report it (`runtime.GetDeviceAttr`).
///
/// # Errors
///
/// Returns an error if the runtime of the device is not enabled and `name` is not `exist`, which is
/// then `false`.
pub fn get_device_attr(device: DLDevice, name: &str) -> Result<Any> {
    tvm_ffi::cached_global_func!("runtime.GetDeviceAttr").call_tuple((
        device.device_type as i32,
        device.device_id,
        String::from(name),
    ))
}

/// Return whether the runtime of the device or target kind `target` (`"cpu"`, `"cuda"`,
/// `"llvm"`...) is enabled (`runtime.RuntimeEnabled`).
pub fn enabled(target: &str) -> Result<bool> {
    tvm_ffi::cached_global_func!("runtime.RuntimeEnabled")
        .call_tuple((String::from(target),))?
        .try_into()
}

/// Return the number of threads the runtime thread pool uses (`runtime.NumThreads`).
pub fn num_threads() -> Result<i32> {
    tvm_ffi::cached_global_func!("runtime.NumThreads").call_tuple(())?.try_into()
}

// The CUDA queries are in the CUDA runtime library, which may not be built or loaded: they are
// looked up when called, and their absence is an error.

/// Return the number of CUDA devices (`runtime.GetCudaDeviceCount`).
///
/// # Errors
///
/// Returns an error if the CUDA runtime is not built or loaded.
pub fn get_cuda_device_count() -> Result<i32> {
    tvm_ffi::Function::get_global("runtime.GetCudaDeviceCount")?.call_tuple(())?.try_into()
}

/// Return the free and total memory of the current CUDA device, as a message
/// (`runtime.GetCudaFreeMemory`).
///
/// # Errors
///
/// Returns an error if the CUDA runtime is not built or loaded.
pub fn get_cuda_free_memory() -> Result<String> {
    tvm_ffi::Function::get_global("runtime.GetCudaFreeMemory")?.call_tuple(())?.try_into()
}
