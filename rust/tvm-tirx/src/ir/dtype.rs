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

//! Data types. A [`DLDataType`] is the DLPack element type of a tensor or a primitive value;
//! these functions build them as plain values, usable in constants, and
//! [`PrimType`](crate::ir::PrimType) is the IR type of one.

use tvm_ffi::DLDataType;
use tvm_ffi::DLDataTypeCode;

/// Return the signed integer type of `bits` bits, such as `int32` for 32.
pub const fn int(bits: u8) -> DLDataType {
    scalar(DLDataTypeCode::kDLInt, bits)
}

/// Return the unsigned integer type of `bits` bits, such as `uint8` for 8.
pub const fn uint(bits: u8) -> DLDataType {
    scalar(DLDataTypeCode::kDLUInt, bits)
}

/// Return the IEEE floating-point type of `bits` bits, such as `float32` for 32.
pub const fn float(bits: u8) -> DLDataType {
    scalar(DLDataTypeCode::kDLFloat, bits)
}

/// Return the brain floating-point type of `bits` bits, such as `bfloat16` for 16.
pub const fn bfloat(bits: u8) -> DLDataType {
    scalar(DLDataTypeCode::kDLBfloat, bits)
}

/// Return the boolean type, `bool`, stored in 8 bits.
pub const fn bool() -> DLDataType {
    scalar(DLDataTypeCode::kDLBool, 8)
}

/// Return the void type, encoded as an opaque handle of 0 bits and 0 lanes.
pub const fn void() -> DLDataType {
    DLDataType { code: DLDataTypeCode::kDLOpaqueHandle as u8, bits: 0, lanes: 0 }
}

/// Return the vector of `lanes` elements of the scalar `dtype`, such as `float16x4`.
///
/// `lanes` is the raw DLPack lane field: values from 32768 up encode scalable vectors.
pub const fn with_lanes(dtype: DLDataType, lanes: u16) -> DLDataType {
    DLDataType { code: dtype.code, bits: dtype.bits, lanes }
}

/// The scalar type of `code` and `bits`.
const fn scalar(code: DLDataTypeCode, bits: u8) -> DLDataType {
    DLDataType { code: code as u8, bits, lanes: 1 }
}
