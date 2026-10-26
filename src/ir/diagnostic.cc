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

/*! \file src/ir/diagnostic.cc
 * \brief Shared rendering of IR error locations.
 */
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/diagnostic.h>

#include <optional>
#include <sstream>
#include <string>

namespace tvm {
namespace {
/*! \brief Marker line that precedes the TVMScript-rendered location block. */
constexpr const char* kLocationMarker = "Location (TVMScript):";

/*!
 * \brief Render `node` as TVMScript with `path` underlined, via the script
 *        printer global registry, so src/ir does not link against the printer.
 *
 * Returns the rendered snippet, or std::nullopt if the printer entry is
 * unavailable or throws.
 */
std::optional<std::string> RenderScriptWithUnderline(const ffi::ObjectRef& node,
                                                     const ffi::reflection::AccessPath& path) {
  namespace ffi = tvm::ffi;
  auto config_fn = ffi::Function::GetGlobal("node.PrinterConfig");
  auto script_fn = ffi::Function::GetGlobal("node.TVMScriptPrinterScript");
  if (!config_fn.has_value() || !script_fn.has_value()) return std::nullopt;
  try {
    ffi::Map<ffi::String, ffi::Any> config_dict;
    // Error snippets use the same display conventions as script() and repr().
    config_dict.Set("extra_config", ffi::Map<ffi::String, ffi::Any>{
                                        {"ir.comment_imports", true},
                                    });
    config_dict.Set("path_to_underline", ffi::Array<ffi::reflection::AccessPath>{path});
    // Show enough context that a small function renders in full (no
    // "(... N lines skipped ...)" marker), while still bounding a large
    // module. A small TIR/Relax function is ~8-15 lines; 10 lines of context
    // on each side of the underline covers it end-to-end.
    config_dict.Set("num_context_lines", static_cast<int>(10));
    config_dict.Set("render_invisible_path_info", true);
    ffi::Any cfg = (*config_fn)(config_dict);
    ffi::Any rendered = (*script_fn)(node, cfg);
    return rendered.cast<ffi::String>().operator std::string();
  } catch (...) {
    return std::nullopt;
  }
}
}  // namespace

ffi::Error EnrichErrorWithContext(const ffi::Error& err, const IRModule& mod, ffi::String context,
                                  ffi::Optional<GlobalVar> func) {
  namespace ffi = tvm::ffi;
  // No visit-context payload => nothing to resolve; return the error untouched.
  // (The enriched error below drops the payload, so an outer catch that
  // re-enriches lands here and returns unchanged — no double location block.)
  auto visit_ctx = ffi::VisitErrorContext::TryGetFromError(err);
  if (!visit_ctx) return err;

  // Resolve + render against the local function when requested, else the module.
  // Fall back to the module root if the function is missing or yields no path.
  ffi::ObjectRef root = mod;
  if (func.has_value()) {
    if (auto bf = mod->functions.Get(func.value())) {
      root = bf.value();
    } else {
      func = ffi::Optional<GlobalVar>(std::nullopt);
    }
  }

  auto try_render = [&](const ffi::ObjectRef& search_root) -> std::optional<std::string> {
    // Prefer the precise throw site. Prefix matches may include unrelated
    // leaves of an enclosing statement before the full match in traversal order.
    auto paths = ffi::VisitErrorContext::FindAccessPaths(search_root, visit_ctx.value());
    if (paths.empty()) {
      paths = ffi::VisitErrorContext::FindAccessPaths(search_root, visit_ctx.value(),
                                                      /*allow_prefix_match=*/true);
    }
    if (paths.empty()) return std::nullopt;
    return RenderScriptWithUnderline(search_root, paths[0]);
  };

  std::optional<std::string> snippet = try_render(root);
  if (!snippet && func.has_value()) {
    // Function-local resolution failed; retry against the whole module.
    snippet = try_render(mod);
  }
  if (!snippet) return err;

  // Append the operation context + underlined TVMScript location. Preserve kind,
  // original message, and backtrace; drop the VisitErrorContext payload so an
  // outer catch does not re-enrich.
  std::ostringstream suffix;
  suffix << "\n\n" << context << "\n" << kLocationMarker << "\n" << snippet.value();
  return ffi::Error(err.kind(), err.message() + suffix.str(), err.backtrace());
}

}  // namespace tvm
