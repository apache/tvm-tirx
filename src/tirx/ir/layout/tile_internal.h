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

/*
 * Internal helpers for TileLayout implementations.
 * This header is private to the layout implementation files.
 */

#ifndef TVM_TIRX_IR_LAYOUT_TILE_INTERNAL_H_
#define TVM_TIRX_IR_LAYOUT_TILE_INTERNAL_H_

#include "utils.h"

namespace tvm {
namespace tirx {

namespace layout_details {

void CheckWellFormed(const TileLayoutObj* layout);
PrimExpr GetSize(const TileLayoutObj* layout, ffi::Optional<ffi::String> axis_name);
PrimExpr GetSpan(const TileLayoutObj* layout, ffi::Optional<ffi::String> axis_name);
ffi::Map<ffi::String, PrimExpr> Apply(const TileLayoutObj* layout, ffi::Array<PrimExpr> coord);
ffi::Map<ffi::String, PrimExpr> Apply(const TileLayoutObj* layout, PrimExpr coord);
ffi::Map<ffi::String, PrimExpr> Apply(const TileLayoutObj* layout,
                                      const ffi::Array<PrimExpr>& coord,
                                      const ffi::Array<PrimExpr>& shape);
Layout Canonicalize(const TileLayoutObj* layout);
Layout Tile(const TileLayoutObj* layout, const TileLayout& outer,
            const ffi::Array<PrimExpr>& outer_shape, const ffi::Array<PrimExpr>& inner_shape);

void CheckWellFormed(const ComposeLayoutObj* layout);
PrimExpr GetSize(const ComposeLayoutObj* layout, ffi::Optional<ffi::String> axis_name);
PrimExpr GetSpan(const ComposeLayoutObj* layout, ffi::Optional<ffi::String> axis_name);
ffi::Map<ffi::String, PrimExpr> Apply(const ComposeLayoutObj* layout, ffi::Array<PrimExpr> coord);
ffi::Map<ffi::String, PrimExpr> Apply(const ComposeLayoutObj* layout, PrimExpr coord);
ffi::Map<ffi::String, PrimExpr> Apply(const ComposeLayoutObj* layout,
                                      const ffi::Array<PrimExpr>& coord,
                                      const ffi::Array<PrimExpr>& shape);
Layout Canonicalize(const ComposeLayoutObj* layout);
Layout Tile(const ComposeLayoutObj* layout, const TileLayout& outer,
            const ffi::Array<PrimExpr>& outer_shape, const ffi::Array<PrimExpr>& inner_shape);

}  // namespace layout_details

// Group a tile layout's shard by a logical shape, returning the grouped layout and separators.
std::pair<TileLayout, std::vector<int64_t>> Group(TileLayout layout,
                                                  const ffi::Array<PrimExpr>& shape);

// Same as Group but returns std::nullopt instead of fatal-checking when the
// layout cannot be regrouped by ``shape``.
std::optional<std::pair<TileLayout, std::vector<int64_t>>> TryGroup(
    TileLayout layout, const ffi::Array<PrimExpr>& shape);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_IR_LAYOUT_TILE_INTERNAL_H_
