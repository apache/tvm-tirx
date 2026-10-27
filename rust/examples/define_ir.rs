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

//! Operators and a type defined in Rust, in a library of their own.
//!
//! `cargo build --example define_ir` builds this as `target/debug/examples/libdefine_ir.so`,
//! which exports `tirx_register_ir`, as every library built on `tvm_tirx` does.
//! `tests/python/ir/test_op_rust.py` loads it, which registers these operators and gives each its
//! TVMScript builder, as `import tvm_tirx` loads `libtvm_tirx_rust`: `T.example_clamp(i, n)`,
//! `T.example_mark()`, `my_ext.hint(i, attrs={..})` and `with T.example_loop(n) as j:`; and
//! registers the type, with its class `T.ExamplePair`. The test declares a `my_ext` API
//! destination explicitly and supplies that module when parsing printed source; exposing
//! an extension operator does not register a default parser namespace.

use tvm_tirx::ir::op_attr::{CALL_EFFECT_KIND, FIXED_RETURN_TYPE};
use tvm_tirx::ir::sig::{self, IntExpr};
use tvm_tirx::ir::{
    CallEffectKind, CallObj, DictAttrs, Expr, OpDef, PrimType, Type, TypeObj, Var, UNKNOWN_LOC,
};
use tvm_tirx::tvm_ffi::{self, rs_class, Array, ObjectRefCast, Result};

fn first_arg_type(call: &CallObj) -> Result<Type> {
    Ok(call.args.get(0)?.ty.clone())
}

/// The variable a loop binds in its body: an index of the extent's type.
fn loop_index(call: &CallObj) -> Result<Array<Var>> {
    let ty: PrimType = call.args.get(0)?.ty.clone().try_cast()?;
    Ok(Array::new(vec![Var::new("i", ty, UNKNOWN_LOC)]))
}

tvm_tirx::register_ops! {
    OpDef::new("tirx.example_clamp", "Clamp an index to the range `[0, bound)`.")
        .signature((
            sig::arg::<IntExpr>("index", "The index."),
            sig::arg::<IntExpr>("bound", "The exclusive upper bound."),
        ))
        .set_infer_type(first_arg_type)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);

    // `Call.attrs` holds a `DictAttrs`; outside `tirx.`, the calls print as `my_ext.hint(..)`.
    OpDef::new("my_ext.hint", "Return the value, with hints for the code generator.")
        .signature((sig::arg::<Expr>("value", "The value."), sig::call_attrs::<DictAttrs>()))
        .set_infer_type(first_arg_type)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::ExprAnnotation);

    // A region operation, which binds an index in its body.
    OpDef::new("tirx.example_loop", "Run the body for each index in `[0, extent)`.")
        .signature(sig::arg::<IntExpr>("extent", "The number of iterations."))
        .set_region_body_params(loop_index);
}

// A type, which Python constructs from its fields, `T.ExamplePair(T.int32, T.float32)`, as it
// constructs a C++ type.
tvm_tirx::register_types! {
    /// The type of a pair of values.
    #[rs_class("tirx.ExamplePairType", final)]
    pub struct ExamplePairTypeObj {
        base: TypeObj,
        pub first: Type,
        pub second: Type,
    }
    def_object_ref!(pub ExamplePairType, Type, ExamplePairTypeObj);
}

// A block can be anywhere in the library.
mod effects {
    use super::*;

    tvm_tirx::register_ops! {
        OpDef::new("tirx.example_mark", "
            Mark a point in the program.

            The call has no value, so TVMScript emits it as a statement.
        ")
            .signature(())
            .set_attr(FIXED_RETURN_TYPE, PrimType::void())
            .set_attr(CALL_EFFECT_KIND, CallEffectKind::Opaque);
    }
}
