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

//! Tests of `ir::type`, and of an `ir.Type` defined in Rust with `rs_class` over the generated
//! `TypeObj`: seen through the FFI type table, the C++ engines (structural equality and hashing,
//! deep copy, `repr`, TVMScript), the Rust engines and the IR that uses it.

use std::mem::offset_of;

use tvm_tirx::ir::{
    UNKNOWN_LOC, Location, dtype, Evaluate, FuncType, MissingType, NoneType, PrimType, PtrType,
    TensorRegionType, TupleType, Type, TypeObj, Var, VarObj,
};
use tvm_tirx::tvm_ffi::{self, def_object_ref, rs_class, structural_map};
use tvm_tirx::tvm_ffi::{
    Any, AnyView, Array, DLDataType, DLDataTypeExt, DefRegionKind, FieldGetter, MapDispatch,
    ObjectCore, ObjectRefCore, Result, ShallowCopy, String as FfiString, StructuralView, WalkOrder,
};

use super::{scope};
use crate::common::{
    agree, assert_type_contract, global, native, runtime_type_info, serial, structural_equal,
};

#[test]
fn types() {
    let _serial = serial();
    MissingType::new();
    NoneType::new();
    TensorRegionType::new();
    let i32_ty = PrimType::int(32);
    assert_eq!(i32_ty.dtype, dtype::int(32));
    let ptr = PtrType::new(&i32_ty, &scope()).unwrap();
    assert!(structural_equal(&ptr.element_type, &i32_ty));
    let tuple = TupleType::new([&i32_ty, &i32_ty], UNKNOWN_LOC);
    assert_eq!(tuple.fields.len(), 2);
    let func = FuncType::new([&i32_ty], &i32_ty);
    assert_eq!(func.arg_types.len(), 1);
}

/// The builtin types are the dtypes their names parse to, and the nodes C++ shares.
#[test]
fn builtin_types() {
    let _serial = serial();
    let cases: [(&str, PrimType, DLDataType); 9] = [
        ("int32", PrimType::int(32), dtype::int(32)),
        ("int64", PrimType::int(64), dtype::int(64)),
        ("uint8", PrimType::uint(8), dtype::uint(8)),
        ("float32", PrimType::float(32), dtype::float(32)),
        ("float16", PrimType::float(16), dtype::float(16)),
        ("bfloat16", PrimType::bfloat(16), dtype::bfloat(16)),
        ("bool", PrimType::bool(), dtype::bool()),
        (
            "float16x4",
            PrimType::float(16).with_lanes(4).unwrap(),
            dtype::with_lanes(dtype::float(16), 4),
        ),
        ("int8x32", PrimType::int(8).with_lanes(32).unwrap(), dtype::with_lanes(dtype::int(8), 32)),
    ];
    for (name, ty, dtype) in cases {
        assert_eq!(dtype, DLDataType::try_from_str(name).unwrap(), "{name}");
        assert_eq!(ty.dtype, dtype, "{name}");
        let shared: Type = native("ir.PrimType", &[(&dtype).into()]).unwrap().try_into().unwrap();
        assert!(ty.same_as(&shared), "{name}");
        assert!(ty.same_as(&PrimType::from_dtype(dtype).unwrap()), "{name}");
    }
    let void = PrimType::void();
    assert_eq!(void.dtype, dtype::void());
    assert_eq!((void.dtype.code, void.dtype.bits, void.dtype.lanes), (3, 0, 0));
    assert!(void.with_lanes(4).is_err());
    assert!(PrimType::int(32).with_lanes(40000).is_err());
}

#[test]
fn types_agree_with_cpp() {
    let _serial = serial();
    let int32: Type = PrimType::int(32).into();
    let float32: Type = PrimType::float(32).into();
    let missing: Type = MissingType::new().into();
    for (what, element) in [("PtrType", &int32), ("PtrType(missing)", &missing)] {
        let native_ptr = native("ir.PtrType", &[element.into(), (&scope()).into()]);
        agree(what, PtrType::new(element, &scope()), native_ptr);
    }
    for (what, fields) in
        [("TupleType", vec![int32.clone(), float32.clone()]), ("TupleType(empty)", vec![])]
    {
        let array: Array<Type> = fields.iter().cloned().collect();
        let native_tuple =
            native("ir.TupleType", &[(&array).into(), (&Location::unknown()).into()]);
        agree(what, Ok(TupleType::new(fields, UNKNOWN_LOC)), native_tuple);
    }
    let args: Array<Type> = [int32.clone()].into_iter().collect();
    let native_func = native("ir.FuncType", &[(&args).into(), (&float32).into()]);
    agree("FuncType", Ok(FuncType::new([&int32], &float32)), native_func);

    let none: Type = native("ir.NoneType", &[]).unwrap().try_into().unwrap();
    assert!(NoneType::new().same_as(&none));
    let region: Type = native("ir.TensorRegionType", &[]).unwrap().try_into().unwrap();
    assert!(TensorRegionType::new().same_as(&region));
}

/// A pair of types with a tag, defined in Rust.
#[rs_class("testing.rust.PairType", final)]
pub struct PairTypeObj {
    base: TypeObj,
    /// The first element.
    pub lhs: Type,
    /// The second element.
    pub rhs: Type,
    pub tag: i64,
    /// Not part of structural equality or hashing.
    #[reflect(seq_hash_ignore)]
    pub note: FfiString,
}

def_object_ref!(pub PairType, Type, PairTypeObj);

impl PairType {
    /// A copy with the fields `edit` assigns replaced (C++ `CopyOnWrite()`).
    fn with(&self, edit: impl FnOnce(&mut PairTypeObj)) -> Self {
        let mut node = self.shallow_copy();
        edit(&mut node);
        node.into()
    }
}

fn pair(lhs: Type, rhs: Type, tag: i64, note: &str) -> PairType {
    PairTypeObj::new(TypeObj::new(Location::unknown()), lhs, rhs, tag, note).into()
}

fn int_float_pair(tag: i64, note: &str) -> PairType {
    pair(PrimType::int(32).into(), PrimType::float(32).into(), tag, note)
}

/// How C++ prints a node (`ffi.ReprPrint`).
fn repr<T>(value: &T) -> String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let text = global("ffi.ReprPrint").call_packed(&[AnyView::from(value)]).unwrap();
    FfiString::try_from(text).unwrap().as_str().to_string()
}

/// C++ `ffi::StructuralHash()(value)`.
fn structural_hash<T>(value: &T) -> Result<i64>
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    let flag = <AnyView as From<&bool>>::from;
    global("ffi.StructuralHash")
        .call_packed(&[AnyView::from(value), flag(&false), flag(&false)])?
        .try_into()
}

#[test]
fn rust_type_registers_under_ir_type_with_its_fields_reflected() {
    let _serial = serial();
    // The first use registers the type and its reflection.
    assert_type_contract::<PairTypeObj, TypeObj>(true);

    let info = runtime_type_info::<PairTypeObj>();
    let metadata = unsafe { &*info.metadata };
    assert_eq!(metadata.structural_eq_hash_kind, 1, "TreeNode, as every ir.Type");
    let fields: Vec<(String, i64, i64)> = (0..info.num_fields as usize)
        .map(|index| {
            let field = unsafe { &*info.fields.add(index) };
            (field.name.as_str().to_string(), field.offset, field.flags)
        })
        .collect();
    assert_eq!(
        fields,
        vec![
            ("lhs".to_string(), offset_of!(PairTypeObj, lhs) as i64, 0),
            ("rhs".to_string(), offset_of!(PairTypeObj, rhs) as i64, 0),
            ("tag".to_string(), offset_of!(PairTypeObj, tag) as i64, 0),
            ("note".to_string(), offset_of!(PairTypeObj, note) as i64, 1 << 3),
        ]
    );
}

#[test]
fn rust_type_reflection_reads_own_and_inherited_fields() {
    let _serial = serial();
    let value = int_float_pair(7, "seven");
    let get = |field: &str| -> Any {
        FieldGetter::new(PairTypeObj::type_index(), field).unwrap().get_any(&*value).unwrap()
    };
    assert!(Type::try_from(get("lhs")).unwrap().same_as(&value.lhs));
    assert!(Type::try_from(get("rhs")).unwrap().same_as(&value.rhs));
    assert_eq!(i64::try_from(get("tag")).unwrap(), 7);
    assert_eq!(FfiString::try_from(get("note")).unwrap().as_str(), "seven");
    // `loc` belongs to the parent `ir.Type`.
    assert!(Location::try_from(get("loc")).unwrap().same_as(&value.loc));
}

#[test]
fn rust_type_cpp_structural_equal_and_hash_walk_the_fields() {
    let _serial = serial();
    let a = int_float_pair(7, "a");
    let b = int_float_pair(7, "b");
    let c = int_float_pair(8, "a");
    assert!(structural_equal(&a, &b), "the note is ignored");
    assert!(!structural_equal(&a, &c));
    assert_eq!(structural_hash(&a).unwrap(), structural_hash(&b).unwrap());
    assert_ne!(structural_hash(&a).unwrap(), structural_hash(&c).unwrap());

    // Nested in itself, as any type.
    let int32: Type = PrimType::int(32).into();
    let outer_a = pair(a.clone().into(), int32.clone(), 0, "");
    let outer_b = pair(b.into(), int32.clone(), 0, "");
    let outer_c = pair(c.into(), int32, 0, "");
    assert!(structural_equal(&outer_a, &outer_b));
    assert!(!structural_equal(&outer_a, &outer_c));
    assert_eq!(structural_hash(&outer_a).unwrap(), structural_hash(&outer_b).unwrap());
}

#[test]
fn rust_type_cpp_repr_and_tvmscript_print_the_node() {
    let _serial = serial();
    let value = int_float_pair(7, "seven");
    let text = repr(&value);
    assert!(text.contains("testing.rust.PairType"), "{text}");
    assert!(text.contains("tag=7"), "{text}");
    assert!(text.contains("seven"), "{text}");

    let script = global("script.printer.Script")
        .call_packed(&[AnyView::from(&value), AnyView::new()])
        .unwrap();
    let script = FfiString::try_from(script).unwrap();
    assert!(script.as_str().contains("testing.rust.Pair("), "{}", script.as_str());
    assert!(script.as_str().contains("\"seven\""), "{}", script.as_str());
}

#[test]
fn rust_type_cpp_deep_copy_goes_through_the_shallow_copy_and_the_setters() {
    let _serial = serial();
    let value = int_float_pair(7, "seven");
    let copy = global("ffi.DeepCopy").call_packed(&[AnyView::from(&value)]).unwrap();
    let copy = PairType::try_from(copy).unwrap();
    assert!(!copy.same_as(&value));
    assert!(structural_equal(&copy, &value));
    assert_eq!(copy.tag, 7);
    assert_eq!(copy.note.as_str(), "seven");
}

/// Adds one to every integer leaf.
struct IncrementIntegers;

impl MapDispatch for IncrementIntegers {
    fn dispatch_map(&mut self, value: &StructuralView, _: DefRegionKind) -> Option<Result<Any>> {
        value.cast::<i64>().map(|integer| Ok(Any::from(integer + 1)))
    }
}

#[test]
fn rust_type_rust_engines_descend_through_the_fields() {
    let _serial = serial();
    let value = int_float_pair(7, "seven");
    // Shared with `value`, so the engine copies the node through `__ffi_shallow_copy__`.
    let mapped =
        structural_map(value.clone(), &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    let mapped = PairType::try_from(mapped).unwrap();
    assert!(!mapped.same_as(&value));
    assert_eq!(mapped.tag, 8);
    assert_eq!(mapped.note.as_str(), "seven");
    assert!(mapped.lhs.same_as(&value.lhs));
    assert_eq!(value.tag, 7);
}

#[test]
fn rust_type_with_is_copy_on_write() {
    let _serial = serial();
    let value = int_float_pair(7, "seven");
    let edited = value.with(|node| node.tag = 9);
    assert!(!edited.same_as(&value));
    assert_eq!(edited.tag, 9);
    assert_eq!(value.tag, 7);
    assert!(edited.lhs.same_as(&value.lhs));
    assert!(edited.rhs.same_as(&value.rhs));
    // The generated `ShallowCopy` of `TypeObj` keeps the location.
    assert!(edited.loc.same_as(&value.loc));
    assert_eq!(edited.note.as_str(), "seven");
}

#[test]
fn rust_type_converts_through_any_and_the_type_hierarchy() {
    let _serial = serial();
    let value = int_float_pair(1, "");
    let as_type: Type = value.clone().into();
    assert!(as_type.as_node::<PairTypeObj>().is_some());
    let any = Any::from(as_type);
    assert_eq!(any.type_index(), PairTypeObj::type_index());
    let back = PairType::try_from(any).unwrap();
    assert!(back.same_as(&value));
}

#[test]
fn rust_type_of_a_variable_takes_part_in_ir() {
    let _serial = serial();
    // The type of a variable, as `PrimType` or `TupleType` is one.
    let x = Var::new("x", int_float_pair(7, "seven"), UNKNOWN_LOC);
    assert_eq!(x.ty.as_node::<PairTypeObj>().unwrap().tag, 7);
    assert!(structural_equal(&x.ty, &int_float_pair(7, "other")));
    assert!(!structural_equal(&x.ty, &int_float_pair(8, "seven")));

    // Inside IR, which C++ deep-copies through the shallow copy and the setters.
    let stmt = Evaluate::new(x.clone(), UNKNOWN_LOC);
    let copy = global("ffi.DeepCopy").call_packed(&[AnyView::from(&stmt)]).unwrap();
    let copy = Evaluate::try_from(copy).unwrap();
    let copied = copy.value.as_node::<VarObj>().unwrap();
    let ty = copied.ty.as_node::<PairTypeObj>().unwrap();
    assert_eq!((ty.tag, ty.note.as_str()), (7, "seven"));
    assert!(structural_equal(&copied.ty, &x.ty));
}
