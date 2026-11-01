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
//! An external consumer with a renamed TIRx dependency and no direct TVM-FFI dependency.

pub mod nodes {
    #[ir_api::tvm_ffi::rs_class("testing.overloads.Parent", crate = ir_api::tvm_ffi)]
    pub struct ParentObj {
        base: ir_api::ir::ExprObj,
    }
    ir_api::tvm_ffi::def_object_ref!(
        pub Parent, ir_api::ir::Expr, ParentObj, crate = ir_api::tvm_ffi
    );

    #[ir_api::tvm_ffi::rs_class("testing.overloads.Leaf", final, crate = ir_api::tvm_ffi)]
    pub struct LeafObj {
        base: ParentObj,
    }
    ir_api::tvm_ffi::def_object_ref!(pub Leaf, Parent, LeafObj, crate = ir_api::tvm_ffi);
    ir_api::tvm_ffi::impl_object_upcast!(Leaf => ir_api::ir::Expr);
}

// Invoke from a scope without imported traits/types, using paths and a renamed dependency.
mod collisions {
    use super::nodes::Parent as R;
    #[allow(non_camel_case_types, dead_code)]
    type i32 = ::core::primitive::i64;
    ir_api::def_ex_op_overloads!(R);
}
ir_api::def_ex_op_overloads!(nodes::Leaf);

#[cfg(test)]
mod tests {
    use ir_api::ir::prim::PrimOps;
    use ir_api::ir::{Expr, ExprObj, PrimType, UNKNOWN_LOC};
    use ir_api::tvm_ffi::ObjectRefCore;

    #[test]
    fn downstream_node_inherits_expression_operations() -> ir_api::tvm_ffi::Result<()> {
        let base = ExprObj::new(UNKNOWN_LOC.into(), PrimType::int(64).into());
        let parent = super::nodes::ParentObj::new(base);
        let leaf: super::nodes::Leaf = super::nodes::LeafObj::new(parent).into();
        let same: Expr = (&leaf + 0).try_into()?;
        assert!(same.same_as(&leaf));
        let same: Expr = (0i32 + &leaf).try_into()?;
        assert!(same.same_as(&leaf));
        let same: Expr = (leaf.clone() * 1).try_into()?;
        assert!(same.same_as(&leaf));
        let same: Expr = (1i32 * leaf.clone()).try_into()?;
        assert!(same.same_as(&leaf));
        let _comparison: Expr = leaf.less(1).try_into()?;
        let _sum: Expr = (&leaf + leaf.clone()).try_into()?;
        let parent: super::nodes::Parent = leaf.into();
        let same: Expr = (parent.clone() + 0).try_into()?;
        assert!(same.same_as(&parent));
        Ok(())
    }
}
