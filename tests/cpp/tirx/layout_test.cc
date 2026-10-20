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
#include <gtest/gtest.h>
#include <tvm/tirx/layout.h>

#include <type_traits>

namespace tvm::tirx {
namespace {

static_assert(!std::is_polymorphic_v<LayoutNode>);
static_assert(!std::is_polymorphic_v<TileLayoutNode>);
static_assert(!std::is_polymorphic_v<ComposeLayoutNode>);
static_assert(!std::is_polymorphic_v<Layout>);

int64_t AsInt(const PrimExpr& expr) {
  PrimExpr simplified = sym::Analyzer()->Simplify(expr);
  return simplified.as_or_throw<IntImm>()->value.as<int64_t>().value();
}

TEST(Layout, WrapperDispatch) {
  const Axis memory = Axis::Get("m");
  const TileLayout tile({Iter(8, 1, memory)}, {}, {});
  const TileLayout outer({Iter(2, 1, memory)}, {}, {});
  for (const Layout& layout : ffi::Array<Layout>{tile, ComposeLayout(0, 1, 1, tile)}) {
    layout.CheckWellFormed();
    EXPECT_EQ(AsInt(layout.GetSize()), 8);
    EXPECT_EQ(AsInt(layout.GetSpan()), 8);
    const int mapped = layout.as<ComposeLayoutNode>() ? 2 : 3;
    EXPECT_EQ(AsInt(layout.Apply(PrimExpr(3))["m"]), mapped);
    EXPECT_EQ(AsInt(layout.Apply(ffi::Array<PrimExpr>{3})["m"]), mapped);
    EXPECT_EQ(AsInt(layout.Apply({0, 3}, {2, 4})["m"]), mapped);
    // A shape that cannot regroup the tile retains flatten-and-split semantics.
    EXPECT_EQ(AsInt(layout.Apply({1, 0}, {2, 3})["m"]), mapped);
    EXPECT_EQ(AsInt(layout.Canonicalize().Apply(PrimExpr(3))["m"]), mapped);
    Layout tiled = layout.Tile(outer, {2}, {8});
    EXPECT_EQ(AsInt(tiled.GetSize()), 16);
    EXPECT_EQ(AsInt(tiled.Apply(PrimExpr(3))["m"]), mapped);
    EXPECT_THROW(layout.Apply({1}, {2, 4}), ffi::Error);
  }
  EXPECT_EQ(AsInt(tile.GetSize(ffi::String("m"))), 8);
  EXPECT_EQ(AsInt(tile.GetSpan(ffi::String("m"))), 8);
  const Layout compose = ComposeLayout(0, 1, 1, tile);
  EXPECT_THROW(compose.GetSize(ffi::String("m")), ffi::Error);
  EXPECT_THROW(compose.GetSpan(ffi::String("m")), ffi::Error);
}

TEST(Layout, ConcreteCopyOnWriteAndLifetime) {
  const Axis memory = Axis::Get("m");
  const Iter iter(8, 1, memory);
  const auto initial_refs = iter->use_count();
  {
    const TileLayout original({iter}, {}, {});
    TileLayout copy = original;
    copy.CopyOnWrite()->offset.Set(memory, PrimExpr(8));
    EXPECT_FALSE(copy.same_as(original));
    EXPECT_EQ(AsInt(original.Apply(PrimExpr(3))["m"]), 3);
    EXPECT_EQ(AsInt(copy.Apply(PrimExpr(3))["m"]), 11);

    const ComposeLayout composed(0, 1, 1, original);
    ComposeLayout composed_copy = composed;
    composed_copy.CopyOnWrite()->tile_layout = copy;
    EXPECT_FALSE(composed_copy.same_as(composed));
    const Layout base = composed;
    const ffi::Any erased = composed_copy;
    EXPECT_EQ(AsInt(base.Apply(PrimExpr(3))["m"]), 2);
    EXPECT_EQ(AsInt(erased.cast<Layout>().Apply(PrimExpr(3))["m"]), 10);
  }
  EXPECT_EQ(iter->use_count(), initial_refs);
}

}  // namespace
}  // namespace tvm::tirx
