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

//! Result-derived constraint facts and shared-engine mutation permissions.

use tvm_tirx::ir::*;
use tvm_tirx::sym::{BindingConstraintObj, Constraint, PredicateConstraintObj, RangeConstraintObj};
use tvm_tirx::tirx::structural_mutate_with_constraint_context::{
    structural_mutate_with_constraint_context as mutate,
    StructuralMutatorWithConstraintContextView as View,
};
use tvm_tirx::tvm_ffi::{
    dispatch, Any, DefRegionKind, MutationView, ObjectRefCore, Result, Unchanged, UnchangedOr,
};

use super::{int32, op};
use crate::common::serial;

fn eval(value: i64) -> Stmt {
    Evaluate::new(int32(value), UNKNOWN_LOC).into()
}

#[derive(Default)]
struct Rewrite {
    facts: Vec<(i64, Vec<Constraint>)>,
    definitions: Vec<DefRegionKind>,
    restored: bool,
}

#[dispatch(mutate, (MutationView<'_>, &mut View<'_, Self>), Result<UnchangedOr<Any>>)]
impl Rewrite {
    fn mutate_integer(
        &mut self,
        node: MutationView<'_, IntImm>,
        _: &mut View<'_, Self>,
    ) -> Result<UnchangedOr<Any>> {
        if node.value.to_i64().unwrap() == 4 { Ok(int32(8).into()) } else { Ok(Unchanged.into()) }
    }
    fn mutate_var(
        &mut self,
        node: MutationView<'_, Var>,
        view: &mut View<'_, Self>,
    ) -> Result<UnchangedOr<Any>> {
        self.definitions.push(view.def_region_kind());
        view.default_mutate(self, node)
    }
    fn mutate_evaluate(
        &mut self,
        node: MutationView<'_, Evaluate>,
        view: &mut View<'_, Self>,
    ) -> Result<UnchangedOr<Any>> {
        let marker = node.value.downcast_ref::<IntImmObj>().unwrap().value.to_i64().unwrap();
        self.facts.push((marker, view.base().constraints().to_array().iter().cloned().collect()));
        view.default_mutate(self, node)
    }
    fn mutate_for(
        &mut self,
        node: MutationView<'_, For>,
        view: &mut View<'_, Self>,
    ) -> Result<UnchangedOr<Any>> {
        let before = view.base().constraints().to_array().len();
        let result = view.default_mutate(self, node);
        self.restored = view.base().constraints().to_array().len() == before;
        result
    }
}

#[test]
fn result_facts_and_scope_restore_for_retained_and_moved_roots() {
    let _serial = serial();
    for moved in [false, true] {
        let x = PrimVar::new("mutated_x", PrimType::int(32), UNKNOWN_LOC);
        let i = PrimVar::new("mutated_i", PrimType::int(32), UNKNOWN_LOC);
        let body = For::new(
            &i,
            int32(0),
            int32(4),
            int32(1),
            ForKind::Default,
            eval(11),
            None,
            UNKNOWN_LOC,
        )
        .unwrap();
        let branch =
            If::new(op("prim._OpLT", &x, int32(4)), body, Some(&eval(12)), UNKNOWN_LOC).unwrap();
        let root = SeqStmt::new(
            [Stmt::from(Bind::new(&x, int32(4), UNKNOWN_LOC).unwrap()), branch.into(), eval(13)],
            UNKNOWN_LOC,
        );
        let retained = (!moved).then(|| root.clone());
        let mut seen = Rewrite::default();
        let result = SeqStmt::try_from(mutate(root, &mut seen).unwrap()).unwrap();
        assert_eq!(
            seen.facts.iter().map(|(n, facts)| (*n, facts.len())).collect::<Vec<_>>(),
            [(11, 5), (12, 2), (13, 1)]
        );
        assert!(seen.restored);
        assert!(seen.definitions.contains(&DefRegionKind::Simple));
        let result_binding = result.seq[0].downcast_ref::<BindObj>().unwrap();
        let result_branch = result.seq[1].downcast_ref::<IfObj>().unwrap();
        let result_loop = result_branch.then_case.seq[0].downcast_ref::<ForObj>().unwrap();
        let binding = seen.facts[0].1[0].downcast_ref::<BindingConstraintObj>().unwrap();
        assert!(binding.var.same_as(&result_binding.var));
        assert!(binding.value.same_as(&result_binding.value));
        assert_eq!(binding.value.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(8));
        let guard = seen.facts[0].1[1].downcast_ref::<PredicateConstraintObj>().unwrap();
        assert!(guard.condition.same_as(&result_branch.condition));
        let range = seen.facts[0].1[2].downcast_ref::<RangeConstraintObj>().unwrap();
        assert!(range.var.same_as(&result_loop.var));
        assert_eq!(range.range.begin.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(0));
        assert_eq!(range.range.extent.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(8));
        assert_eq!(result_loop.end.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(8));
        assert!(seen.facts[0].1[0].same_as(&seen.facts[2].1[0]));
        if let Some(retained) = retained {
            let original = retained.seq[0].downcast_ref::<BindObj>().unwrap();
            assert_eq!(original.value.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(4));
        }
    }
}

#[test]
fn alias_acquired_in_callback_forces_copying_default_descent() {
    let _serial = serial();
    #[derive(Default)]
    struct Alias {
        saved: Option<Any>,
    }
    #[dispatch(mutate, (MutationView<'_>, &mut View<'_, Self>), Result<UnchangedOr<Any>>)]
    impl Alias {
        fn mutate_for(
            &mut self,
            node: MutationView<'_, For>,
            view: &mut View<'_, Self>,
        ) -> Result<UnchangedOr<Any>> {
            self.saved = Some(node.as_any_view().into());
            view.default_mutate(self, node)
        }
        fn mutate_integer(
            &mut self,
            node: MutationView<'_, IntImm>,
            _: &mut View<'_, Self>,
        ) -> Result<UnchangedOr<Any>> {
            Ok(if node.value.to_i64() == Some(4) { int32(8).into() } else { Unchanged.into() })
        }
    }
    let mut alias = Alias::default();
    let i = PrimVar::new("aliased_i", PrimType::int(32), UNKNOWN_LOC);
    let root =
        For::new(&i, int32(0), int32(4), int32(1), ForKind::Default, eval(11), None, UNKNOWN_LOC)
            .unwrap();
    let result = For::try_from(mutate(root, &mut alias).unwrap()).unwrap();
    let saved = alias.saved.unwrap();
    let saved = saved.downcast_ref::<ForObj>().unwrap();
    assert_eq!(saved.end.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(4));
    assert_eq!(result.end.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(8));
    assert!(!std::ptr::eq(saved, &*result));
}
