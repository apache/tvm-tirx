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
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ir/diagnostic.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace {

TEST(Diagnostic, LeavesUnresolvableErrorsUnchanged) {
  ffi::Error error("ValueError", "original message", "original backtrace");
  IRModule mod;
  EXPECT_TRUE(EnrichErrorWithContext(error, mod, "codegen").same_as(error));
  ffi::detail::UpdateVisitErrorContext(error, IntImm::Int32(1));
  EXPECT_TRUE(EnrichErrorWithContext(error, mod, "codegen").same_as(error));
}

TEST(Diagnostic, PreservesPassContextAndAvoidsDuplicateLocation) {
  auto value = IntImm::Int32(7);
  auto body = SeqStmt({Evaluate(value)});
  tirx::Function func({}, body);
  GlobalVar gv("main");
  IRModule mod({{gv, func}});
  ffi::Error error("ValueError", "original message", "original backtrace");
  ffi::detail::UpdateVisitErrorContext(error, value);
  ffi::detail::UpdateVisitErrorContext(error, body);
  auto enriched = transform::EnrichPassErrorWithContext(error, mod, "TestPass", gv);
  EXPECT_EQ(enriched.kind(), error.kind());
  EXPECT_EQ(enriched.backtrace(), error.backtrace());
  std::string message = enriched.message();
  EXPECT_EQ(message.find("original message"), 0);
  EXPECT_NE(message.find("Error in pass: TestPass"), std::string::npos);
  EXPECT_NE(message.find("Location (TVMScript):"), std::string::npos);
  EXPECT_NE(message.find("^"), std::string::npos);
  EXPECT_FALSE(ffi::VisitErrorContext::TryGetFromError(enriched).has_value());
  EXPECT_TRUE(EnrichErrorWithContext(enriched, mod, "codegen").same_as(enriched));
}

}  // namespace
}  // namespace tvm
