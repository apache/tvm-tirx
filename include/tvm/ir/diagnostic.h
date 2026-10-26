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

/*! \file tvm/ir/diagnostic.h
 * \brief Shared rendering of IR error locations.
 */
#ifndef TVM_IR_DIAGNOSTIC_H_
#define TVM_IR_DIAGNOSTIC_H_

#include <tvm/ffi/error.h>
#include <tvm/ir/module.h>

namespace tvm {
/*!
 * \brief Append an underlined TVMScript location from an error's visit context.
 *
 * Preserve the original error kind, message, and backtrace. Return the original
 * error if the context cannot be resolved or printed. A rendered error drops
 * its visit context so outer handlers do not append the location again.
 *
 * \param err The original error.
 * \param mod The module containing the visited nodes.
 * \param context The operation description preceding the location.
 * \param func Optional function to search before falling back to the module.
 * \return The enriched error, or the original error on fallback.
 */
TVM_TIRX_DLL ffi::Error EnrichErrorWithContext(
    const ffi::Error& err, const IRModule& mod, ffi::String context,
    ffi::Optional<GlobalVar> func = ffi::Optional<GlobalVar>(std::nullopt));
}  // namespace tvm
#endif  // TVM_IR_DIAGNOSTIC_H_
