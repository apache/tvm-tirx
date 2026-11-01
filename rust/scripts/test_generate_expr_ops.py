# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Coverage follows reflected ancestry, even for new opaque and indirect nodes."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import generate_expr_ops as generator


class ExpressionCoverageTest(unittest.TestCase):
    def test_new_indirect_and_opaque_nodes_follow_metadata(self):
        infos = {
            "example.Parent": SimpleNamespace(ancestors=["ffi.Object", "ir.Expr"]),
            "example.NewNode": SimpleNamespace(
                ancestors=["ffi.Object", "ir.Expr", "example.Parent"], total_size=None
            ),
            "example.Manual": SimpleNamespace(ancestors=["ffi.Object", "ir.Expr"]),
            "example.NotExpr": SimpleNamespace(ancestors=["ffi.Object"]),
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "ir.rs").write_text("// tvm-ffi-stubgen(begin): object/ir.Expr\n")
            (source / "extension.rs").write_text(
                "\n".join(
                    f"// tvm-ffi-stubgen(begin): object/{key}"
                    for key in infos
                    if key != "example.Manual"
                )
                + "\n"
                + "// tvm-ffi-stubgen(ty-map): example.Manual -> crate::manual::Manual\n"
            )
            with patch.object(
                generator, "object_info_from_type_key", side_effect=infos.__getitem__
            ):
                self.assertEqual(
                    generator.expression_wrappers(source),
                    [
                        "crate::manual::Manual",
                        "crate::extension::NewNode",
                        "crate::extension::Parent",
                        "crate::ir::Expr",
                    ],
                )
            # Missing registration must fail, rather than silently omit the new node.
            del infos["example.NewNode"]
            with patch.object(
                generator, "object_info_from_type_key", side_effect=infos.__getitem__
            ):
                with self.assertRaises(KeyError):
                    generator.expression_wrappers(source)

    def test_duplicate_or_missing_root_declarations_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaisesRegex(ValueError, "Missing object block"):
                generator.expression_wrappers(source)
            for name in ["one.rs", "two.rs"]:
                (source / name).write_text("// tvm-ffi-stubgen(begin): object/ir.Expr\n")
            with self.assertRaisesRegex(ValueError, "Duplicate object block"):
                generator.expression_wrappers(source)


if __name__ == "__main__":
    unittest.main()
