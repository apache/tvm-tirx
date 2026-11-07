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

//! Compilation targets. A [`Target`] describes the device and options code is compiled for, as an
//! instance of a registered [`TargetKind`]; it is created from a tag, a kind name or a config, and
//! [`Target::scope`] makes it the current one.

// tvm-ffi-stubgen(missing-docs): allow
// Every object registered under `target` has its block in this file.
// tvm-ffi-stubgen(prefix): target
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Module
// tvm-ffi-stubgen(import-object): crate::ir::module::IRModule

// tvm-ffi-stubgen(begin): import-section
use crate::ir::module::IRModule;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Module;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::object::ObjectRef;
// tvm-ffi-stubgen(end)

/// A compilation target: its `kind`, the `keys` that classify it, such as `cpu`, the `attrs` of
/// its options, and an optional `host` target for the code that runs on the host. A target may come
/// from a registered `tag`, such as `nvidia/nvidia-a100`.
///
/// # Examples
///
/// ```
/// use tvm_tirx::target::Target;
/// use tvm_tirx::tvm_ffi::ObjectRefCore;
///
/// let cpu = Target::new(r#"{"kind": "llvm", "mcpu": "skylake"}"#)?;
/// assert!(cpu.as_json()?.as_str().contains(r#""keys":["cpu"]"#));
///
/// let scope = cpu.scope()?;
/// assert!(Target::current(false)?.unwrap().same_as(&cpu));
/// drop(scope);
/// assert!(Target::current(true)?.is_none());
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/target.Target
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Target {
    data_: ObjectArc<TargetObj>,
}

/// The object behind [`Target`].
///
/// Opaque: bytes [72, 88) of [24, 88) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "target.Target"]
#[type_final]
pub struct TargetObj {
    base_: Object,
}

#[allow(missing_docs)]
impl TargetObj {
    pub fn kind(&self) -> Result<TargetKind> {
        FieldGetter::new(Self::type_index(), "kind")?.get(self)
    }

    pub fn tag(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "tag")?.get(self)
    }

    pub fn keys(&self) -> Result<Array<String>> {
        FieldGetter::new(Self::type_index(), "keys")?.get(self)
    }

    pub fn attrs(&self) -> Result<Map<String, Any>> {
        FieldGetter::new(Self::type_index(), "attrs")?.get(self)
    }

    pub fn host(&self) -> Result<Option<ObjectRef>> {
        FieldGetter::new(Self::type_index(), "host")?.get(self)
    }
}
// tvm-ffi-stubgen(end)

impl Target {
    /// Create the target that a tag (`"nvidia/nvidia-a100"`), a kind name (`"llvm"`) or a JSON
    /// config describes (`target.Target`).
    ///
    /// # Errors
    ///
    /// Returns an error if the tag or the kind is not registered, or the JSON config is invalid.
    pub fn new(target: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("target.Target")
            .call_tuple((String::from(target),))?
            .try_into()
    }

    /// Create the target that a config map describes, such as `{"kind": "cuda", "arch": "sm_80"}`
    /// or a `tag` with attribute overrides (`target.Target`).
    ///
    /// The two-target form of `target.Target`, a target with a host, is [`Target::with_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if the config is invalid, as [`new`](Self::new) does.
    pub fn from_config(config: &Map<String, Any>) -> Result<Self> {
        tvm_ffi::cached_global_func!("target.Target").call_tuple((config,))?.try_into()
    }

    /// Make this the current target until the returned [`TargetScope`] is dropped
    /// (`target.TargetEnterScope`, `target.TargetExitScope`).
    pub fn scope(&self) -> Result<TargetScope> {
        tvm_ffi::cached_global_func!("target.TargetEnterScope").call_tuple((self,))?;
        Ok(TargetScope(self.clone(), crate::ThreadBound::default()))
    }

    /// Return the innermost current target, or `None` if there is none and `allow_none` is `true`
    /// (`target.TargetCurrent`).
    ///
    /// # Errors
    ///
    /// Returns an error if there is no current target and `allow_none` is `false`.
    pub fn current(allow_none: bool) -> Result<Option<Self>> {
        tvm_ffi::cached_global_func!("target.TargetCurrent").call_tuple((allow_none,))?.try_into()
    }

    /// Return the config map that describes the target (`target.TargetExport`).
    pub fn export(&self) -> Result<Map<String, Any>> {
        tvm_ffi::cached_global_func!("target.TargetExport").call_tuple((self,))?.try_into()
    }

    /// Return the target as JSON, which [`new`](Self::new) parses back (`target.TargetAsJSON`).
    pub fn as_json(&self) -> Result<String> {
        tvm_ffi::cached_global_func!("target.TargetAsJSON").call_tuple((self,))?.try_into()
    }

    /// Return a copy of the target with `host` as its host, or none (`target.WithHost`).
    pub fn with_host(&self, host: Option<&Target>) -> Result<Self> {
        tvm_ffi::cached_global_func!("target.WithHost")
            .call_tuple((self, host.cloned()))?
            .try_into()
    }

    /// Return the `DLDeviceType` of the devices the target runs on (`target.TargetGetDeviceType`).
    pub fn get_target_device_type(&self) -> Result<i32> {
        tvm_ffi::cached_global_func!("target.TargetGetDeviceType").call_tuple((self,))?.try_into()
    }

    /// Return the `feature.<name>` attribute of the target, or `None` when it has none
    /// (`target.TargetGetFeature`).
    pub fn feature(&self, name: &str) -> Result<Any> {
        tvm_ffi::cached_global_func!("target.TargetGetFeature")
            .call_tuple((self, String::from(name)))
    }

    /// Return the names of the registered target kinds (`target.ListTargetKinds`).
    pub fn list_kinds() -> Result<Array<String>> {
        tvm_ffi::cached_global_func!("target.ListTargetKinds").call_tuple(())?.try_into()
    }
}

/// A scope in which a target is the current one, returned by [`Target::scope`].
///
/// The current targets are a stack of the thread, so a scope stays on the thread that opened it,
/// and the scopes end in the reverse order they were opened: drop the innermost one first.
///
/// ```compile_fail
/// fn send<T: Send>() {}
/// send::<tvm_tirx::target::TargetScope>();
/// ```
#[must_use = "the target stops being the current one when the scope is dropped"]
pub struct TargetScope(Target, crate::ThreadBound);

impl Drop for TargetScope {
    fn drop(&mut self) {
        // A destructor cannot report the error; the scopes end in the order they were opened.
        let _ = tvm_ffi::cached_global_func!("target.TargetExitScope").call_tuple((&self.0,));
    }
}

/// A kind of target, such as `llvm` or `cuda`: its `name`, the options it takes, and the device
/// type and keys of its targets by default.
// tvm-ffi-stubgen(begin): object/target.TargetKind
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TargetKind {
    data_: ObjectArc<TargetKindObj>,
}

/// The object behind [`TargetKind`].
///
/// Opaque: bytes [56, 152) of [24, 152) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "target.TargetKind"]
#[type_final]
pub struct TargetKindObj {
    base_: Object,
}

#[allow(missing_docs)]
impl TargetKindObj {
    pub fn name(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "name")?.get(self)
    }

    pub fn default_device_type(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "default_device_type")?.get(self)
    }

    pub fn default_keys(&self) -> Result<Array<String>> {
        FieldGetter::new(Self::type_index(), "default_keys")?.get(self)
    }
}
// tvm-ffi-stubgen(end)

impl TargetKind {
    /// Return the type key of each option of the kind, by name (`target.ListTargetKindOptions`).
    pub fn options(&self) -> Result<Map<String, String>> {
        tvm_ffi::cached_global_func!("target.ListTargetKindOptions").call_tuple((self,))?.try_into()
    }

    /// Return the options of the kind named `kind_name`, as [`options`](Self::options) does
    /// (`target.ListTargetKindOptionsFromName`).
    ///
    /// # Errors
    ///
    /// Returns an error if no kind is named `kind_name`.
    pub fn options_from_name(kind_name: &str) -> Result<Map<String, String>> {
        tvm_ffi::cached_global_func!("target.ListTargetKindOptionsFromName")
            .call_tuple((String::from(kind_name),))?
            .try_into()
    }
}

/// Return the registered target tags, with their targets (`target.TargetTagListTags`).
pub fn list_tags() -> Result<Map<String, Target>> {
    tvm_ffi::cached_global_func!("target.TargetTagListTags").call_tuple(())?.try_into()
}

/// Register the tag `name` for the target `config` describes, replacing a previous one only if
/// `override_`, and return that target (`target.TargetTagAddTag`).
///
/// # Errors
///
/// Returns an error if `name` is already registered and `override_` is `false`.
///
/// # Examples
///
/// ```
/// use tvm_tirx::target::{self, Target};
/// use tvm_tirx::tvm_ffi::{Any, Map, String};
///
/// let config: Map<String, Any> =
///     [("kind".into(), Any::from(String::from("llvm")))].into_iter().collect();
/// target::register_tag("my_ext/cpu", &config, false)?;
/// assert_eq!(Target::new("my_ext/cpu")?.kind()?.name()?, "llvm");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub fn register_tag(name: &str, config: &Map<String, Any>, override_: bool) -> Result<Target> {
    tvm_ffi::cached_global_func!("target.TargetTagAddTag")
        .call_tuple((String::from(name), config, override_))?
        .try_into()
}

/// Generate the code of `module` with the code generator `target.build.<kind>`, such as `llvm`,
/// `c`, `cuda`, `cuda_host` or `nvptx`.
///
/// [`crate::codegen::build_module`] picks the code generator of the target kind.
///
/// # Errors
///
/// Returns an error if the code generator is not built in, or the error it raises.
pub fn build(kind: &str, module: impl Into<IRModule>, target: impl Into<Target>) -> Result<Module> {
    // The code generators are looked up by name: which ones exist depends on the build.
    tvm_ffi::Function::get_global(&format!("target.build.{kind}"))?
        .call_tuple((&module.into(), &target.into()))?
        .try_into()
}
