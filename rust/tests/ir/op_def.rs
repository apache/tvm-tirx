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

//! Tests of `ir::op_def`: operators registered from Rust, by an `OpDef` or in static init blocks,
//! which run when the test binary loads, are used by the native registry, `Call` constructor and validation, and their type inference, validators
//! and region contracts are native hooks; and types registered with them.
//!
//! The registry does not synchronize registration, so the tests take `serial()`; the tests
//! register operators of different names.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use tvm_tirx::ir::op_attr::{CALL_EFFECT_KIND, FIXED_RETURN_TYPE};
use tvm_tirx::ir::sig::{self, IntExpr, PrimExpr, PrimVar};
use tvm_tirx::ir::{
    register_type, AttrKey, Attrs, Call, CallEffectKind, CallObj, DictAttrs, Evaluate, Expr,
    FuncType, IntImm, Op, OpDef, PrimType, RegionStmt, RegionStmtObj, StringImm, Type, TypeObj,
    Var, VarObj, UNKNOWN_LOC,
};
use tvm_tirx::tirx::{CallFFIKernelAttr, TensorVar};
use tvm_tirx::tvm_ffi::static_init::{StaticInitBlock, STATIC_INIT_BLOCKS};
use tvm_tirx::tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex;
use tvm_tirx::tvm_ffi::{
    self, def_object_ref, rs_object, run_static_init_blocks, Any, AnyView, Array, Error, ErrorKind,
    Function, Map, ObjectCore, ObjectRefCore, Result, String as FfiString, ATTRIBUTE_ERROR,
    INTERNAL_ERROR, TYPE_ERROR, VALUE_ERROR,
};

use super::{int32, parse_dtype, tensor_var};
use crate::common::{global, serial, structural_equal};

fn var(name: &str, dtype: &str) -> Var {
    Var::new(name, PrimType::from_dtype(parse_dtype(dtype)).unwrap(), UNKNOWN_LOC)
}

/// `Call(op, args)` without a result type, which the constructor infers.
fn call(op: &Op, args: Vec<Expr>) -> Result<Call> {
    Call::new(None, op.clone(), args, None, Vec::<Type>::new(), UNKNOWN_LOC)
}

fn validate(call: &Call) -> Result<()> {
    global("ir.CallValidate").call_tuple((call.clone(),)).map(|_| ())
}

/// `Call(op, args)` with `ty_args`, `attrs` and a result type, so the constructor infers none.
fn call_with(op: &Op, args: Vec<Expr>, ty_args: Vec<Type>, attrs: Option<Attrs>) -> Call {
    let ty: Type = PrimType::int(32).into();
    Call::new(Some(&ty), op.clone(), args, attrs.as_ref(), ty_args, UNKNOWN_LOC).unwrap()
}

/// The message `Call::Validate` raises for `op(args)` with `ty_args` and `attrs`, with the name
/// of `op` replaced by `<op>`.
fn rejection(op: &Op, args: &[Expr], ty_args: &[Type], attrs: Option<Attrs>) -> String {
    let call = call_with(op, args.to_vec(), ty_args.to_vec(), attrs);
    let error = validate(&call).unwrap_err();
    assert_eq!(error.kind(), TYPE_ERROR, "{}", error.message());
    error.message().replace(&format!("`{}`", op.name.as_str()), "`<op>`")
}

fn assert_error(result: Result<impl Sized>, kind: ErrorKind, message: &str) {
    let Err(error) = result else { panic!("expected {kind:?}: {message}") };
    assert_eq!(error.kind(), kind, "{}", error.message());
    assert!(error.message().contains(message), "{}", error.message());
}

/// A Call's `args` and `ty_args`.
type Arguments = (Vec<Expr>, Vec<Type>);

/// A Call's `args` and `attrs`.
type AttrsCall = (Vec<Expr>, Option<Attrs>);

static INFERRED: AtomicUsize = AtomicUsize::new(0);

fn first_arg_type(call: &CallObj) -> Result<Type> {
    INFERRED.fetch_add(1, Ordering::Relaxed);
    Ok(call.args.get(0)?.ty.clone())
}

fn only_variables(call: &CallObj) -> Result<()> {
    match call.args.iter().all(|arg| arg.downcast_ref::<VarObj>().is_some()) {
        true => Ok(()),
        false => Err(Error::new(VALUE_ERROR, "arguments must be variables", "")),
    }
}

fn faulty(_call: &CallObj) -> Result<()> {
    panic!("validator bug")
}

#[test]
fn ops_registered_from_rust_are_used_natively() {
    let _serial = serial();
    let (x, i, j) = (var("x", "float32"), var("i", "int32"), var("j", "int32"));
    let lanes = var("v", "int32x4");
    let text: Expr = StringImm::new("shared", UNKNOWN_LOC).into();
    let one: Expr = int32(1).into();
    let tensor = tensor_var("A");
    let int32_ty: Type = PrimType::int(32).into();

    // A new operator: its description and attributes are in the registry, and both of its hooks
    // are native function pointers.
    let op = OpDef::new("rust_test.identity", "Return the value.")
        .signature(sig::arg::<Expr>("value", "The value."))
        .set_infer_type(first_arg_type)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::ExprAnnotation)
        .op()
        .unwrap();
    assert!(op.same_as(&Op::get("rust_test.identity").unwrap()));
    assert_eq!(op.doc.as_str(), "Return the value.");
    assert_eq!(op.args_info.get(0).unwrap().name.as_str(), "value");
    let effect = Op::attr_map::<i64>("TCallEffectKind").unwrap().unwrap();
    assert_eq!(effect.get(&op).unwrap(), Some(0));
    let opaque = TVMFFITypeIndex::kTVMFFIOpaquePtr as i32;
    assert_eq!(op.validator().type_index(), opaque);
    assert!(!op.validator_is_custom());
    let infer = Op::attr_map::<*mut c_void>("FInferType").unwrap().unwrap();
    assert!(infer.get(&op).unwrap().is_some());

    // The `Call` constructor, in Rust and natively, and `ReinferType` infer the result type with
    // the Rust hook.
    let identity = call(&op, vec![x.clone().into()]).unwrap();
    assert!(structural_equal(&identity.ty, &x.ty));
    let reinferred: Type =
        global("ir.CallReinferType").call_tuple((identity.clone(),)).unwrap().try_into().unwrap();
    assert!(reinferred.same_as(&x.ty));
    let native: Call = global("ir.Call")
        .call_tuple((
            Option::<Type>::None,
            Expr::from(&op),
            Array::<Expr>::from_vec(vec![x.clone().into()]),
            Option::<Attrs>::None,
            Array::<Type>::from_vec(vec![]),
            tvm_tirx::ir::Location::unknown(),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(native.ty.same_as(&x.ty));
    assert_eq!(INFERRED.load(Ordering::Relaxed), 3);
    validate(&identity).unwrap();

    // An error a Rust hook returns reaches the native caller with its kind and message, and a
    // panic becomes an error instead of unwinding into C++.
    let untyped = OpDef::new("rust_test.untyped", "")
        .set_infer_type(|_: &CallObj| Err(Error::new(TYPE_ERROR, "no result type", "")))
        .op()
        .unwrap();
    assert_error(call(&untyped, vec![]), TYPE_ERROR, "no result type");
    let faulty = OpDef::new("rust_test.faulty", "").set_validator(faulty).op().unwrap();
    assert_error(validate(&call(&faulty, vec![]).unwrap()), INTERNAL_ERROR, "validator bug");

    // A generated validator rejects what the C++ one generated from the same signature does,
    // with the same message.
    let cases: Vec<(&str, OpDef, Vec<Arguments>)> = vec![
        (
            "tirx.get_active_lane_mask",
            OpDef::new("rust_test.lane_mask", "")
                .signature((sig::arg::<IntExpr>("base", ""), sig::arg::<IntExpr>("limit", ""))),
            vec![
                (vec![], vec![]),
                (vec![x.clone().into(), i.clone().into()], vec![]),
                (vec![i.clone().into(), lanes.clone().into()], vec![]),
                (vec![i.clone().into(), i.clone().into(), i.clone().into()], vec![]),
            ],
        ),
        (
            "tirx.vector_high",
            OpDef::new("rust_test.vector_high", "").signature(sig::arg::<PrimExpr>("vec", "")),
            vec![(vec![text.clone()], vec![])],
        ),
        (
            "tirx.gpu_thread_filter",
            OpDef::new("rust_test.thread_filter", "")
                .signature((sig::arg::<PrimVar>("var", ""), sig::arg::<PrimExpr>("pred", ""))),
            vec![
                (vec![one.clone(), i.clone().into()], vec![]),
                (vec![j.clone().into(), text.clone()], vec![]),
            ],
        ),
        (
            "tirx.gpu_storage_sync",
            OpDef::new("rust_test.storage_sync", "")
                .signature(sig::arg::<StringImm>("storage_scope", "")),
            vec![(vec![i.clone().into()], vec![]), (vec![text.clone()], vec![int32_ty.clone()])],
        ),
        (
            "tirx.masked_load",
            OpDef::new("rust_test.masked_load", "").signature((
                sig::arg::<TensorVar>("tensor", ""),
                sig::arg::<PrimExpr>("index", ""),
                sig::var_args::<PrimExpr>("args", ""),
            )),
            vec![
                (vec![], vec![]),
                (vec![i.clone().into(), i.clone().into()], vec![]),
                (vec![one.clone(), i.clone().into()], vec![]),
                (vec![tensor.clone().into(), text.clone()], vec![]),
                (
                    vec![tensor.clone().into(), i.clone().into(), j.clone().into(), text.clone()],
                    vec![],
                ),
            ],
        ),
    ];
    for (native, rust, calls) in cases {
        let (native, rust) = (Op::get(native).unwrap(), rust.op().unwrap());
        for (args, ty_args) in calls {
            let (rust, native) = (
                rejection(&rust, &args, &ty_args, None),
                rejection(&native, &args, &ty_args, None),
            );
            assert_eq!(rust, native);
        }
    }

    // A tensor variable with any number of trailing values passes both.
    for name in ["tirx.masked_load", "rust_test.masked_load"] {
        let args = vec![tensor.clone().into(), i.clone().into(), j.clone().into()];
        validate(&call_with(&Op::get(name).unwrap(), args, vec![], None)).unwrap();
    }

    // Variadic tails and type arguments.
    let tails = OpDef::new("rust_test.tails", "")
        .signature((
            sig::arg::<PrimExpr>("index", ""),
            sig::var_args::<PrimExpr>("rest", ""),
            sig::ty_arg::<PrimType>("T", ""),
            sig::var_ty_args::<Type>("more", ""),
        ))
        .op()
        .unwrap();
    let func_ty: Type = FuncType::new(Vec::<Type>::new(), PrimType::int(32), UNKNOWN_LOC).into();
    validate(&call_with(
        &tails,
        vec![i.clone().into(), one.clone()],
        vec![int32_ty.clone(), func_ty.clone()],
        None,
    ))
    .unwrap();
    assert_eq!(
        rejection(&tails, &[], std::slice::from_ref(&int32_ty), None),
        "Op `<op>`: Call.args expected at least 1 argument, got 0"
    );
    assert_eq!(
        rejection(&tails, &[i.clone().into()], &[], None),
        "Op `<op>`: Call.ty_args expected at least 1 type argument, got 0"
    );
    assert_eq!(
        rejection(
            &tails,
            &[i.clone().into(), one.clone(), text.clone()],
            std::slice::from_ref(&int32_ty),
            None,
        ),
        "Op `<op>`: `Call.args[2]` (`rest`) expected `ir.PrimExpr`, got `ir.StringImm[ty=ir.StringType]`."
    );
    assert_eq!(
        rejection(&tails, &[i.clone().into()], std::slice::from_ref(&func_ty), None),
        "Op `<op>`: `Call.ty_args[0]` (`T`) expected `ir.PrimType`, got `ir.FuncType`."
    );

    // `call_attrs`: the operator's attrs type, and the check that `Call.attrs` holds one.
    let dict: Attrs = DictAttrs::new(Map::new()).into();
    let attrs_cases: Vec<(&str, OpDef, Vec<AttrsCall>)> = vec![
        (
            "tirx.device_entry",
            OpDef::new("rust_test.device_entry", "").signature((
                sig::var_args::<Expr>("launch_values", ""),
                sig::call_attrs::<DictAttrs>(),
            )),
            vec![(vec![], None), (vec![i.clone().into()], None)],
        ),
        (
            "tirx.call_ffi_kernel",
            OpDef::new("rust_test.call_ffi_kernel", "").signature((
                sig::arg::<Expr>("kernel", ""),
                sig::var_args::<Expr>("args", ""),
                sig::call_attrs::<CallFFIKernelAttr>(),
            )),
            vec![
                (vec![], Some(dict.clone())),
                (vec![x.clone().into()], Some(dict.clone())),
                (vec![x.clone().into(), i.clone().into()], None),
            ],
        ),
    ];
    for (native, rust, calls) in attrs_cases {
        let (native, rust) = (Op::get(native).unwrap(), rust.op().unwrap());
        assert_eq!(rust.attrs_type_key.as_str(), native.attrs_type_key.as_str());
        assert_eq!(rust.attrs_type_index(), native.attrs_type_index());
        for (args, attrs) in calls {
            let (rust, native) = (
                rejection(&rust, &args, &[], attrs.clone()),
                rejection(&native, &args, &[], attrs),
            );
            assert_eq!(rust, native);
        }
    }
    for name in ["tirx.device_entry", "rust_test.device_entry"] {
        let entry =
            call_with(&Op::get(name).unwrap(), vec![i.clone().into()], vec![], Some(dict.clone()));
        validate(&entry).unwrap();
    }
    // A signature without `call_attrs` leaves the operator without an attrs type.
    let plain = OpDef::new("rust_test.plain", "")
        .signature(sig::call_attrs::<DictAttrs>())
        .signature(sig::var_args::<Expr>("launch_values", ""))
        .op()
        .unwrap();
    assert_eq!(plain.attrs_type_key.as_str(), "");

    // A custom validator replaces the generated one, and a later signature keeps it.
    let checked = OpDef::new("rust_test.checked", "Take variables only.")
        .signature(sig::arg::<Expr>("value", ""))
        .set_validator(only_variables)
        .signature((sig::arg::<Expr>("value", ""), sig::var_args::<Expr>("rest", "")))
        .op()
        .unwrap();
    assert!(checked.validator_is_custom());
    assert_eq!(checked.validator().type_index(), opaque);
    assert_eq!(checked.var_args_info.as_ref().unwrap().name.as_str(), "rest");
    validate(&call(&checked, vec![x.clone().into(), x.clone().into()]).unwrap()).unwrap();
    assert_error(
        validate(&call(&checked, vec![one.clone()]).unwrap()),
        VALUE_ERROR,
        "arguments must be variables",
    );
    // The Rust `Call::validate` runs the same hook.
    assert_error(
        call(&checked, vec![one.clone()]).unwrap().validate(),
        VALUE_ERROR,
        "arguments must be variables",
    );

    // Registering a name twice, an attribute twice, or a second custom validator is an error.
    assert_error(
        OpDef::new("rust_test.identity", "").op(),
        ATTRIBUTE_ERROR,
        "is registered before",
    );
    assert_error(
        OpDef::new("rust_test.twice", "")
            .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure)
            .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure)
            .op(),
        VALUE_ERROR,
        "Attribute TCallEffectKind of rust_test.twice is already registered",
    );
    assert_error(
        OpDef::new("rust_test.validated_twice", "")
            .set_validator(only_variables)
            .set_validator(only_variables)
            .op(),
        VALUE_ERROR,
        "Validator of rust_test.validated_twice is already registered",
    );

    // Replacing and removing an attribute show in a column read before.
    let replaced = OpDef::new("rust_test.replaced", "")
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::ExprAnnotation)
        .replace_attr(CALL_EFFECT_KIND, CallEffectKind::Pure)
        .op()
        .unwrap();
    assert_eq!(effect.get(&replaced).unwrap(), Some(1));
    let reset = OpDef::new("rust_test.reset", "")
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure)
        .reset_attr(CALL_EFFECT_KIND)
        .op()
        .unwrap();
    assert_eq!(effect.get(&reset).unwrap(), None);

    // Argument descriptions and the attrs type.
    let described = OpDef::new("rust_test.described", "")
        .add_arg("lhs", "Left.")
        .add_ty_arg("T", "Result type.")
        .attrs_type_key("ir.DictAttrs")
        .op()
        .unwrap();
    assert_eq!(described.args_info.len(), 1);
    assert_eq!(described.ty_args_info.get(0).unwrap().name.as_str(), "T");
    assert_eq!(described.attrs_type_key.as_str(), "ir.DictAttrs");

    // A key of its own makes a new attribute column.
    const NOTE: AttrKey<FfiString> = AttrKey::new("rust_test.Note");
    let noted = OpDef::new("rust_test.noted", "").set_attr(NOTE, "hint").op().unwrap();
    let note = Op::attr_map::<FfiString>("rust_test.Note").unwrap().unwrap();
    assert_eq!(note.get(&noted).unwrap().unwrap().as_str(), "hint");
}

/// The type of the first argument, as `first_arg_type` but without counting the inference.
fn identity_type(call: &CallObj) -> Result<Type> {
    Ok(call.args.get(0)?.ty.clone())
}

fn accept(_call: &CallObj) -> Result<()> {
    Ok(())
}

/// A type and a count, registered with the operators when the test binary loads.
#[rs_object("rust_test.BlockType", final)]
pub struct BlockTypeObj {
    base: TypeObj,
    pub elem: Type,
    #[reflect(kw_only)]
    pub count: i64,
}
def_object_ref!(pub BlockType, Type, BlockTypeObj);

tvm_ffi::static_init_block! {
    OpDef::new("rust_test.defined_identity", "
        Return the value.

        The documentation is dedented.
    ")
        .signature(sig::arg::<Expr>("value", "The value."))
        .set_infer_type(identity_type)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::ExprAnnotation);

    // `tirx.launch_thread`'s region contract (`src/tirx/op/region.cc`), written in Rust: one
    // thread index of the extent's type.
    OpDef::new("rust_test.launch", "Bind a thread index below an extent.")
        .signature((sig::arg::<StringImm>("tag", ""), sig::arg::<IntExpr>("extent", "")))
        .set_region_body_params(launch_body_params)
        .set_region_validate(no_results_or_attributes);

    OpDef::new("rust_test.panicking_region", "")
        .set_region_body_params(|_: &CallObj| -> Result<Array<Var>> { panic!("hook bug") });

    register_type::<BlockType>()?;
}

mod elsewhere {
    use super::*;

    tvm_ffi::static_init_block! {
        OpDef::new("rust_test.defined_elsewhere", "An operator in another module.")
            .set_attr(FIXED_RETURN_TYPE, PrimType::void());
    }
}

// Two blocks that fail, which `run_static_init_blocks` reports together while it runs the others: a
// name C++ registers, which `.op()?` returns, and a second custom validator, which the chain panics
// with when it is dropped, after which the block stops.
mod failing {
    use super::*;

    tvm_ffi::static_init_block! {
        OpDef::new("tirx.launch_thread", "").op()?;
    }

    tvm_ffi::static_init_block! {
        OpDef::new("rust_test.defined_broken", "").set_validator(accept).set_validator(accept);
        OpDef::new("rust_test.after_broken", "");
    }
}

/// The static init blocks of `module` in this test binary.
fn blocks_of(module: &str) -> Vec<&'static StaticInitBlock> {
    let prefix = format!("ir::op_def{module}:");
    let of_module = |identity: &str| {
        identity
            .strip_prefix(&prefix)
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
    };
    STATIC_INIT_BLOCKS.iter().filter(|block| of_module(&block.identity())).collect()
}

fn is_block_registered(block: &StaticInitBlock) -> bool {
    Function::get_global(&block.marker()).is_ok()
}

fn launch_body_params(call: &CallObj) -> Result<Array<Var>> {
    let tag = StringImm::try_from(call.args.get(0)?)?;
    let extent_ty: PrimType = PrimType::try_from(&call.args.get(1)?.ty)?;
    let dtype = extent_ty.dtype;
    if tag.value.as_str().is_empty() {
        let message = "Check failed: (!tag->value.empty()) is false: launch_thread expects a \
                       nonempty StringImm thread tag";
        return Err(Error::new(VALUE_ERROR, message, ""));
    }
    if dtype.bits <= 1 {
        let message = format!(
            "Check failed: dtype.bits() > 1 ({} vs. 1) : launch_thread extent must have a scalar \
             integer type wider than one bit",
            dtype.bits
        );
        return Err(Error::new(VALUE_ERROR, &message, ""));
    }
    Ok(Array::from_vec(vec![Var::new("", extent_ty, UNKNOWN_LOC)]))
}

fn no_results_or_attributes(region: &RegionStmtObj) -> Result<()> {
    if !region.result_vars.is_empty() || !region.attrs.dict.is_empty() {
        let message = format!(
            "Check failed: (region->result_vars.empty() && region->attrs->dict.empty()) is false: \
             {} expects no results or attributes",
            region.op.name.as_str()
        );
        return Err(Error::new(VALUE_ERROR, &message, ""));
    }
    Ok(())
}

/// `RegionStmt(op, {tag, extent}, body_params, attrs, Evaluate(0), result_vars)`, its error text
/// with the operator's name taken out.
fn region(
    op: &Op,
    tag: &str,
    extent: Expr,
    body_params: Vec<Var>,
    attrs: Vec<(&str, i64)>,
    result_vars: Vec<Var>,
) -> std::result::Result<(), (String, String)> {
    let args: Vec<Expr> = vec![StringImm::new(tag, UNKNOWN_LOC).into(), extent];
    let attrs: Map<FfiString, Any> =
        attrs.into_iter().map(|(key, value)| (FfiString::from(key), Any::from(value))).collect();
    let body = Evaluate::new(0, UNKNOWN_LOC);
    let built = RegionStmt::new(
        op.clone(),
        args,
        body_params,
        DictAttrs::new(attrs),
        body,
        result_vars,
        UNKNOWN_LOC,
    );
    built.map(|_| ()).map_err(|error| {
        let message = error.message().replace(op.name.as_str(), "<op>");
        (error.kind().to_string(), message)
    })
}

/// The text of a panic `catch_unwind` caught.
fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(text) => *text,
        Err(payload) => payload.downcast_ref::<&str>().unwrap_or(&"").to_string(),
    }
}

#[test]
fn a_chain_completes_when_it_is_dropped() {
    let _serial = serial();
    // Dropped at the end of its statement, the chain exposes the operator.
    OpDef::new("rust_test.dropped", "").set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
    let effect = Op::attr_map::<i64>("TCallEffectKind").unwrap().unwrap();
    assert_eq!(effect.get(&Op::get("rust_test.dropped").unwrap()).unwrap(), Some(1));

    // A failed chain panics with its error when it is dropped.
    let caught = std::panic::catch_unwind(|| {
        OpDef::new("tirx.launch_thread", "");
    });
    let text = panic_text(caught.unwrap_err());
    assert!(text.starts_with("AttributeError: "), "{text}");
    assert!(text.ends_with("Operator tirx.launch_thread is registered before"), "{text}");

    // Dropped while another panic unwinds, it does not panic again, which would abort.
    let caught = std::panic::catch_unwind(|| {
        let _def = OpDef::new("tirx.launch_thread", "");
        panic!("the first panic");
    });
    assert_eq!(panic_text(caught.unwrap_err()), "the first panic");
}

#[test]
fn static_init_blocks_define_operators_and_types() {
    let _serial = serial();

    // The blocks ran when the test binary loaded; a block that fails does not keep the others from
    // registering theirs.
    let identity = Op::get("rust_test.defined_identity").unwrap();
    assert_eq!(identity.doc.as_str(), "Return the value.\n\nThe documentation is dedented.");
    assert_eq!(identity.args_info.get(0).unwrap().name.as_str(), "value");
    let effect = Op::attr_map::<i64>("TCallEffectKind").unwrap().unwrap();
    assert_eq!(effect.get(&identity).unwrap(), Some(0));
    let elsewhere = Op::get("rust_test.defined_elsewhere").unwrap();
    assert_eq!(elsewhere.doc.as_str(), "An operator in another module.");
    let fixed = Op::attr_map::<Type>("TFixedReturnType").unwrap().unwrap();
    assert!(structural_equal(&fixed.get(&elsewhere).unwrap().unwrap(), &PrimType::void()));

    // `run_static_init_blocks` reports the blocks that failed together, each with its first error;
    // a block stops at its first failure, and a failed block does not run again.
    let error = run_static_init_blocks().unwrap_err();
    let message = error.message();
    assert!(message.starts_with("2 static init blocks failed:"), "{message}");
    assert!(message.contains("op_def::failing:"), "{message}");
    // A chain that is dropped reports its error as the panic of the block.
    assert!(
        message.contains(
            "InternalError: panicked: ValueError: Validator of rust_test.defined_broken is already \
             registered"
        ),
        "{message}"
    );
    assert!(message.contains("AttributeError: "), "{message}");
    assert!(message.contains("is registered before"), "{message}");
    assert!(Op::get("rust_test.after_broken").is_err(), "the block stopped");
    assert_eq!(run_static_init_blocks().unwrap_err().message(), message);

    // The type `register_type` registers with them: tvm-ffi constructs it from its fields,
    // positionally or by keyword, and reads it back from a JSON graph.
    let init: Function = tvm_ffi::reflection::TypeAttrColumn::new("__ffi_init__")
        .and_then(|column| column.get(BlockTypeObj::type_index()))
        .unwrap()
        .try_into()
        .unwrap();
    let kwargs = global("ffi.GetKwargsObject").call_tuple(()).unwrap();
    let (elem, count, four) = (Type::from(PrimType::int(32)), FfiString::from("count"), 4i64);
    let built = init
        .call_packed(&[
            AnyView::from(&elem),
            AnyView::from(&kwargs),
            AnyView::from(&count),
            AnyView::from(&four),
        ])
        .unwrap();
    let block = BlockType::try_from(built).unwrap();
    assert!(block.elem.same_as(&elem));
    assert_eq!(block.count, 4);
    let positional = init.call_packed(&[AnyView::from(&elem), AnyView::from(&four)]);
    assert!(positional.is_err_and(|error| error.kind() == TYPE_ERROR), "`count` is keyword-only");
    let json = global("ffi.ToJSONGraph").call_packed(&[AnyView::from(&block), AnyView::new()]);
    let read = global("ffi.FromJSONGraph").call_packed(&[AnyView::from(&json.unwrap())]);
    let read = BlockType::try_from(read.unwrap()).unwrap();
    assert!(!read.same_as(&block) && structural_equal(&read, &block));

    // A block that succeeds is marked registered for the other libraries of the process; one that
    // fails is not, so it fails in each of them.
    assert!(blocks_of("").into_iter().chain(blocks_of("::elsewhere")).all(is_block_registered));
    let failing = blocks_of("::failing");
    assert_eq!(failing.len(), 2);
    assert!(!failing.into_iter().any(is_block_registered));

    // The Rust `FInferType` types a Call of the operator. The registry keeps one object per name,
    // for every lookup and thread, so `same_as` tests which operator a Call is to.
    let x = var("x", "int32");
    let call = call(&identity, vec![Expr::from(&x)]).unwrap();
    assert!(call.ty.same_as(&x.ty), "the Rust FInferType types the Call");
    assert!(call.op.same_as(&Op::get("rust_test.defined_identity").unwrap()));
    let native = Op::get("tirx.launch_thread").unwrap();
    assert!(!call.op.same_as(&native));
    let name = std::thread::spawn(|| {
        Op::get("rust_test.defined_identity").unwrap().name.as_str().to_string()
    });
    assert_eq!(name.join().unwrap(), "rust_test.defined_identity");
    assert_error(
        Op::get("rust_test.unregistered"),
        ATTRIBUTE_ERROR,
        "Operator rust_test.unregistered is not registered",
    );

    // The region hooks are native function pointers.
    let launch = Op::get("rust_test.launch").unwrap();
    for key in ["FRegionGetBodyParams", "FRegionValidate"] {
        let hooks = Op::attr_map::<*mut c_void>(key).unwrap().unwrap();
        assert!(hooks.get(&launch).unwrap().is_some(), "{key}");
    }

    // The `RegionStmt` constructor checks a region of the Rust operator as it checks one of
    // `tirx.launch_thread`: the same outcome, and the same error but for the operator's name.
    let extent: Expr = int32(8).into();
    let wide: Expr = IntImm::new(PrimType::int(64), 8, UNKNOWN_LOC).unwrap().into();
    let flag: Expr = IntImm::bool(true).into();
    let i = var("i", "int32");
    let cases = vec![
        ("a well-formed region", "threadIdx.x", extent.clone(), vec![i.clone()], vec![], vec![]),
        ("no body parameter", "threadIdx.x", extent.clone(), vec![], vec![], vec![]),
        ("a body parameter of another type", "threadIdx.x", wide, vec![i.clone()], vec![], vec![]),
        ("an empty tag", "", extent.clone(), vec![i.clone()], vec![], vec![]),
        ("a one-bit extent", "threadIdx.x", flag, vec![var("b", "bool")], vec![], vec![]),
        ("an attribute", "threadIdx.x", extent.clone(), vec![i.clone()], vec![("k", 1)], vec![]),
        (
            "a result",
            "threadIdx.x",
            extent.clone(),
            vec![i.clone()],
            vec![],
            vec![var("r", "int32")],
        ),
        ("a repeated variable", "threadIdx.x", extent, vec![i.clone()], vec![], vec![i]),
    ];
    for (what, tag, extent, params, attrs, results) in cases {
        let rust =
            region(&launch, tag, extent.clone(), params.clone(), attrs.clone(), results.clone());
        let cpp = region(&native, tag, extent, params, attrs, results);
        assert_eq!(rust, cpp, "{what}");
    }

    // A panic in a hook becomes an error instead of unwinding into C++.
    let panicking = Op::get("rust_test.panicking_region").unwrap();
    let body = Evaluate::new(0, UNKNOWN_LOC);
    let empty = DictAttrs::new(Map::new());
    let built = RegionStmt::new(
        panicking,
        Vec::<Expr>::new(),
        Vec::<Var>::new(),
        empty,
        body,
        Vec::<Var>::new(),
        UNKNOWN_LOC,
    );
    assert!(built.is_err(), "a panicking hook fails the constructor");
}
