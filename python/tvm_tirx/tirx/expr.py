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
# pylint: disable=redefined-builtin
"""TIR expression nodes.

Each expression node have subfields that can be visited from python side.
For example, you can use addexp.a to get the left operand of an Add node.

.. code-block:: python

  x = tvm_tirx.tirx.Var("n", "int32")
  y = x + 2
  assert(isinstance(y, tvm_tirx.tirx.Add))
  assert(y.a == x)
"""

from tvm_tirx import ir
from tvm_tirx.ir import Expr as Expr
from tvm_tirx.ir import const as const
from tvm_tirx.ir._prim_expr_overload import (
    EqualOp as EqualOp,
)
from tvm_tirx.ir._prim_expr_overload import (
    ExprOp as ExprOp,
)
from tvm_tirx.ir._prim_expr_overload import (
    NotEqualOp as NotEqualOp,
)
from tvm_tirx.ir._prim_expr_overload import (
    div_ambiguity_error as div_ambiguity_error,
)
from tvm_tirx.ir.prim import convert as convert

# Retain historical imports as aliases of the canonical shared definitions.
from tvm_tirx.ir.prim.expr import (
    EQ as EQ,
)
from tvm_tirx.ir.prim.expr import (
    GE as GE,
)
from tvm_tirx.ir.prim.expr import (
    GT as GT,
)
from tvm_tirx.ir.prim.expr import (
    LE as LE,
)
from tvm_tirx.ir.prim.expr import (
    LT as LT,
)
from tvm_tirx.ir.prim.expr import (
    NE as NE,
)
from tvm_tirx.ir.prim.expr import (
    Add as Add,
)
from tvm_tirx.ir.prim.expr import (
    And as And,
)
from tvm_tirx.ir.prim.expr import (
    BinaryOpExpr as BinaryOpExpr,
)
from tvm_tirx.ir.prim.expr import (
    BitwiseAnd as BitwiseAnd,
)
from tvm_tirx.ir.prim.expr import (
    BitwiseNot as BitwiseNot,
)
from tvm_tirx.ir.prim.expr import (
    BitwiseOr as BitwiseOr,
)
from tvm_tirx.ir.prim.expr import (
    BitwiseXor as BitwiseXor,
)
from tvm_tirx.ir.prim.expr import (
    Cast as Cast,
)
from tvm_tirx.ir.prim.expr import (
    CmpExpr as CmpExpr,
)
from tvm_tirx.ir.prim.expr import (
    Div as Div,
)
from tvm_tirx.ir.prim.expr import (
    FloorDiv as FloorDiv,
)
from tvm_tirx.ir.prim.expr import (
    FloorMod as FloorMod,
)
from tvm_tirx.ir.prim.expr import (
    Let as Let,
)
from tvm_tirx.ir.prim.expr import (
    LogicalExpr as LogicalExpr,
)
from tvm_tirx.ir.prim.expr import (
    LShift as LShift,
)
from tvm_tirx.ir.prim.expr import (
    Max as Max,
)
from tvm_tirx.ir.prim.expr import (
    Min as Min,
)
from tvm_tirx.ir.prim.expr import (
    Mod as Mod,
)
from tvm_tirx.ir.prim.expr import (
    Mul as Mul,
)
from tvm_tirx.ir.prim.expr import (
    Not as Not,
)
from tvm_tirx.ir.prim.expr import (
    Or as Or,
)
from tvm_tirx.ir.prim.expr import (
    RShift as RShift,
)
from tvm_tirx.ir.prim.expr import (
    Select as Select,
)
from tvm_tirx.ir.prim.expr import (
    Sub as Sub,
)

Var = ir.Var
