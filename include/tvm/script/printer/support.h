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
 * \file tvm/script/printer/support.h
 * \brief Shared helpers for implementing TVMScript translation hooks.
 */
#ifndef TVM_SCRIPT_PRINTER_SUPPORT_H_
#define TVM_SCRIPT_PRINTER_SUPPORT_H_

#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/script/printer/config.h>
#include <tvm/script/printer/doc_translator.h>

namespace tvm {
namespace script {
namespace printer {

/*!
 * \brief Render an IR object for its registered repr callback.
 *
 * Tries to translate \p obj directly; on error falls back to a plain
 * address string. Defined in src/script/printer/printer.cc so that
 * <tvm/runtime/logging.h> is not pulled into this public header.
 */
TVM_TIRX_DLL ffi::String RedirectReprPrinterMethod(const ffi::ObjectRef& obj);

/*!
 * \brief Render visit-error context through the shared IR error renderer.
 * \param error The original callback error, with traversal breadcrumbs.
 * \param root The original structural operation root.
 * \return The enriched error, or the original error if rendering is unavailable.
 */
TVM_TIRX_DLL ffi::Error RedirectVisitErrorEnrichMethod(const ffi::Error& error,
                                                       const ffi::ObjectRef& root);

/*! \brief Register TVMScript repr and visit-error rendering for ObjectType. */
template <typename ObjectType>
void RegisterScriptRepr() {
  namespace refl = ffi::reflection;
  refl::TypeAttrDef<ObjectType>()
      .def(ffi::type_attr::kRepr.name,
           [](ffi::ObjectRef obj, ffi::Function) -> ffi::String {
             return RedirectReprPrinterMethod(obj);
           })
      .def(ffi::type_attr::kEnrichVisitError.name, [](ffi::Error error, ffi::ObjectRef root) {
        return RedirectVisitErrorEnrichMethod(error, root);
      });
}

/*!
 * \brief Allocate or retrieve a variable identifier and record its origin.
 * \param d The translator owning the variable scope and origin table.
 * \param var The variable to name.
 * \param explicit_def True marks a definition and removes its pending entry;
 *        false denotes a reference. This helper does not emit a definition statement.
 * \return The variable's identifier Doc.
 */
TVM_TIRX_DLL IdDoc VarDoc(DocTranslatorObj* d, const Var& var, bool explicit_def = true);

/*!
 * \brief Translate a Var using the normal translation-hook signature.
 * \param d The translator owning variable definitions.
 * \param input The Var or derived node to translate; other inputs are rejected.
 * \param destination The optional destination object.
 * \return Its identifier, or nullopt after emitting a pending definition when
 *         destination is the input Var. Promotion emits its definition and
 *         dependencies. Other destinations do not emit a definition.
 */
TVM_TIRX_DLL ffi::Optional<ExprDoc> VarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                    const ffi::Object* destination);

/*!
 * \brief Store a non-null value in the initialized metadata table.
 * \param d The translator with metadata state and identifier initialized.
 * \param value The value, reusing an equal existing entry of the same type.
 * \return The metadata lookup Doc, recording the value's origin when it is an object.
 */
TVM_TIRX_DLL ExprDoc AddMetadata(DocTranslatorObj* d, ffi::Any value);

/*!
 * \brief Emit an explicit Call constructor preserving all call fields.
 * \param d The translator for the call's fields.
 * \param call The typed call node; this is not the FDocTranslate hook signature.
 * \param translated_args Optional already-translated arguments.
 * \return The explicit constructor Doc.
 */
TVM_TIRX_DLL ExprDoc
RawCallDocTranslate(DocTranslatorObj* d, const CallObj* call,
                    ffi::Optional<ffi::Array<ExprDoc>> translated_args = std::nullopt);

/*!
 * \brief Translate named Op-call syntax, falling back to an explicit call as needed.
 * \param d The translator for the call's fields.
 * \param call The typed call node; operation-specific hook dispatch belongs to its caller.
 * \return The call Doc, or nullopt for a non-Op callee.
 */
TVM_TIRX_DLL ffi::Optional<ExprDoc> StandardCallDocTranslate(DocTranslatorObj* d,
                                                             const CallObj* call);

}  // namespace printer
}  // namespace script
}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_SUPPORT_H_
