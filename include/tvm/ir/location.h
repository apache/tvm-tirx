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
/*!
 * \file tvm/ir/location.h
 * \brief Source locations and source-text lookup.
 */
#ifndef TVM_IR_LOCATION_H_
#define TVM_IR_LOCATION_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief The source name in the Location
 * \sa SourceNameObj, Location
 */
class SourceName;
/*!
 * \brief The name of a source fragment.
 */
class SourceNameObj : public ffi::Object {
 public:
  /*! \brief The source name. */
  ffi::String name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceNameObj>().def_ro("name", &SourceNameObj::name);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceName", SourceNameObj, ffi::Object);
};

/*!
 * \brief The source name of a file loc.
 * \sa SourceNameObj, Location
 */
class SourceName : public ffi::ObjectRef {
 public:
  /*!
   * \brief Get an SourceName for a given operator name.
   *  Will raise an error if the source name has not been registered.
   * \param name Name of the operator.
   * \return SourceName valid throughout program lifetime.
   */
  TVM_TIRX_DLL static SourceName Get(const ffi::String& name);

  explicit SourceName(ffi::UnsafeInit tag, ffi::ObjectPtr<SourceNameObj> n) : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SourceName, ffi::ObjectRef, SourceNameObj);
};

/*! \brief Base class for immutable source-location metadata. */
class LocationObj : public ffi::Object {
 public:
  static void RegisterReflection() { ffi::reflection::ObjectDef<LocationObj>(); }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Location", LocationObj, ffi::Object);
};

/*! \brief Non-nullable source-location metadata. */
class Location : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Location, ffi::ObjectRef, LocationObj);
};

/*! \brief No source-location information is available. */
class UnknownLocObj : public LocationObj {
 public:
  static void RegisterReflection() { ffi::reflection::ObjectDef<UnknownLocObj>(); }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.UnknownLoc", UnknownLocObj, LocationObj);
};

/*! \brief Reference to the shared immutable unknown location. */
class UnknownLoc : public Location {
 public:
  TVM_TIRX_DLL UnknownLoc();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(UnknownLoc, Location, UnknownLocObj);
};

/*! \brief A range in one source, retaining frontend coordinate conventions. */
class SourceLocObj : public LocationObj {
 public:
  explicit SourceLocObj(SourceName source_name) : source_name(std::move(source_name)) {}
  explicit SourceLocObj(ffi::UnsafeInit tag) : source_name(tag) {}
  /*! \brief The source name. */
  SourceName source_name;
  /*! \brief The starting line number. */
  int start_line;
  /*! \brief The starting column offset. */
  int start_column;
  /*! \brief The ending line number. */
  int end_line;
  /*! \brief The ending column offset. */
  int end_column;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceLocObj>()
        .def_ro("source_name", &SourceLocObj::source_name)
        .def_ro("start_line", &SourceLocObj::start_line)
        .def_ro("start_column", &SourceLocObj::start_column)
        .def_ro("end_line", &SourceLocObj::end_line)
        .def_ro("end_column", &SourceLocObj::end_column);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceLoc", SourceLocObj, LocationObj);
};

/*! \brief A concrete source range. */
class SourceLoc : public Location {
 public:
  TVM_TIRX_DLL SourceLoc(SourceName source_name, int start_line, int start_column, int end_line,
                         int end_column);

  /*! \brief Merge two ranges in the same source. */
  TVM_TIRX_DLL SourceLoc Merge(const SourceLoc& other) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SourceLoc, Location, SourceLocObj);
};

/*! \brief The location of a callee together with its caller's provenance. */
class CallSiteLocObj : public LocationObj {
 public:
  Location callee = UnknownLoc();
  Location caller = UnknownLoc();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallSiteLocObj>()
        .def_ro("callee", &CallSiteLocObj::callee)
        .def_ro("caller", &CallSiteLocObj::caller);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.CallSiteLoc", CallSiteLocObj, LocationObj);
};

/*! \brief Explicit inline callee/caller provenance. */
class CallSiteLoc : public Location {
 public:
  TVM_TIRX_DLL CallSiteLoc(Location callee, Location caller);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CallSiteLoc, Location, CallSiteLocObj);
};

}  // namespace tvm

#endif  // TVM_IR_LOCATION_H_
