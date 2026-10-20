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
#include <tvm/script/printer/doc.h>

#include <type_traits>

namespace tvm::script::printer {
namespace {

static_assert(!std::is_polymorphic_v<DocNode>);
static_assert(!std::is_polymorphic_v<ExprDocNode>);
static_assert(!std::is_polymorphic_v<StmtDocNode>);
static_assert(!std::is_polymorphic_v<ExprStringDocNode>);
static_assert(!std::is_polymorphic_v<StmtBlockDocNode>);
static_assert(!std::is_polymorphic_v<FunctionDocNode>);

TEST(Doc, ConcreteFieldsReleasedThroughBaseAndFFIHandles) {
  const IdDoc child("child");
  const auto initial_refs = child->use_count();
  {
    Doc doc = ExprStringDoc(child, std::nullopt);
    EXPECT_EQ(child->use_count(), initial_refs + 1);
    {
      ffi::Any erased = doc;
      doc = IdDoc("replacement");
      EXPECT_EQ(child->use_count(), initial_refs + 1);
      EXPECT_TRUE(erased.cast<ExprStringDoc>()->value.same_as(child));
    }
    EXPECT_EQ(child->use_count(), initial_refs);
  }
  {
    // Release nested arrays and both statement and expression descendants.
    ffi::Any erased = StmtBlockDoc({ExprStmtDoc(child)});
    EXPECT_EQ(child->use_count(), initial_refs + 1);
    EXPECT_EQ(erased.cast<StmtBlockDoc>()->stmts.size(), 1);
  }
  EXPECT_EQ(child->use_count(), initial_refs);
}

}  // namespace
}  // namespace tvm::script::printer
