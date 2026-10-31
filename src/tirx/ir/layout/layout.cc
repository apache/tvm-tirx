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

#include "tile_internal.h"

namespace tvm {
namespace tirx {

namespace {

template <typename F>
decltype(auto) DispatchLayout(const Layout& layout, F&& f) {
  if (const auto* tile = layout.as<TileLayoutObj>()) {
    return f(tile);
  }
  if (const auto* compose = layout.as<ComposeLayoutObj>()) {
    return f(compose);
  }
  TVM_FFI_THROW(TypeError) << "Unsupported layout type: " << layout->GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

}  // namespace

void Layout::CheckWellFormed() const {
  DispatchLayout(*this, [](const auto* node) { layout_detail::CheckWellFormed(node); });
}

PrimExpr Layout::GetSize(ffi::Optional<ffi::String> axis_name) const {
  return DispatchLayout(*this,
                        [&](const auto* node) { return layout_detail::GetSize(node, axis_name); });
}

PrimExpr Layout::GetSpan(ffi::Optional<ffi::String> axis_name) const {
  return DispatchLayout(*this,
                        [&](const auto* node) { return layout_detail::GetSpan(node, axis_name); });
}

ffi::Map<ffi::String, PrimExpr> Layout::Apply(ffi::Array<PrimExpr> coord) const {
  return Apply(std::move(coord), sym::Analyzer());
}

ffi::Map<ffi::String, PrimExpr> Layout::Apply(PrimExpr coord) const {
  return Apply(std::move(coord), sym::Analyzer());
}

ffi::Map<ffi::String, PrimExpr> Layout::Apply(const ffi::Array<PrimExpr>& coord,
                                              const ffi::Array<PrimExpr>& shape) const {
  return Apply(coord, shape, sym::Analyzer());
}

ffi::Map<ffi::String, PrimExpr> Layout::Apply(ffi::Array<PrimExpr> coord,
                                              const sym::Analyzer& analyzer) const {
  return DispatchLayout(*this, [&](const auto* node) {
    return layout_detail::Apply(node, std::move(coord), analyzer);
  });
}

ffi::Map<ffi::String, PrimExpr> Layout::Apply(PrimExpr coord, const sym::Analyzer& analyzer) const {
  return DispatchLayout(*this, [&](const auto* node) {
    return layout_detail::Apply(node, std::move(coord), analyzer);
  });
}

ffi::Map<ffi::String, PrimExpr> Layout::Apply(const ffi::Array<PrimExpr>& coord,
                                              const ffi::Array<PrimExpr>& shape,
                                              const sym::Analyzer& analyzer) const {
  return DispatchLayout(
      *this, [&](const auto* node) { return layout_detail::Apply(node, coord, shape, analyzer); });
}

Layout Layout::Canonicalize() const {
  return DispatchLayout(*this, [](const auto* node) { return layout_detail::Canonicalize(node); });
}

Layout Layout::Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                    const ffi::Array<PrimExpr>& inner_shape) const {
  return DispatchLayout(*this, [&](const auto* node) {
    return layout_detail::Tile(node, outer, outer_shape, inner_shape);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  auto def = refl::GlobalDef();
  def.def("tirx.LayoutCheckWellFormed", [](Layout layout) { return layout.CheckWellFormed(); });
  def.def("tirx.LayoutGetSize", [](Layout layout, ffi::Optional<ffi::String> axis_name) {
    return layout.GetSize(axis_name);
  });
  def.def("tirx.LayoutGetSpan", [](Layout layout, ffi::Optional<ffi::String> axis_name) {
    return layout.GetSpan(axis_name);
  });
  def.def("tirx.LayoutApplyWithShape",
          [](Layout layout, ffi::Array<PrimExpr> coord, ffi::Array<PrimExpr> shape) {
            return layout.Apply(coord, shape);
          });
  def.def("tirx.LayoutApply",
          [](Layout layout, ffi::Array<PrimExpr> coord) { return layout.Apply(coord); });
  def.def("tirx.LayoutApplyLinear",
          [](Layout layout, PrimExpr coord) { return layout.Apply(coord); });
  def.def("tirx.LayoutApplyWithAnalyzer",
          [](Layout layout, ffi::Array<PrimExpr> coord, ffi::Optional<ffi::Array<PrimExpr>> shape,
             const sym::Analyzer& analyzer) {
            if (shape.has_value()) return layout.Apply(coord, shape.value(), analyzer);
            if (coord.size() == 1) return layout.Apply(coord[0], analyzer);
            return layout.Apply(std::move(coord), analyzer);
          });
  def.def("tirx.LayoutCanonicalize", [](Layout layout) { return layout.Canonicalize(); });
  def.def("tirx.LayoutTile", [](Layout layout, TileLayout outer, ffi::Array<PrimExpr> outer_shape,
                                ffi::Array<PrimExpr> inner_shape) {
    return layout.Tile(outer, outer_shape, inner_shape);
  });
}

}  // namespace tirx
}  // namespace tvm
