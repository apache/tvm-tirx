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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {

TVM_FFI_STATIC_INIT_BLOCK() {
  DocObj::RegisterReflection();
  ExprDocObj::RegisterReflection();
  StmtDocObj::RegisterReflection();
  StmtBlockDocObj::RegisterReflection();
  LiteralDocObj::RegisterReflection();
  ExprStringDocObj::RegisterReflection();
  IdDocObj::RegisterReflection();
  AttrAccessDocObj::RegisterReflection();
  IndexDocObj::RegisterReflection();
  CallDocObj::RegisterReflection();
  OperationDocObj::RegisterReflection();
  LambdaDocObj::RegisterReflection();
  TupleDocObj::RegisterReflection();
  ListDocObj::RegisterReflection();
  DictDocObj::RegisterReflection();
  SliceDocObj::RegisterReflection();
  AssignDocObj::RegisterReflection();
  IfDocObj::RegisterReflection();
  WhileDocObj::RegisterReflection();
  BreakDocObj::RegisterReflection();
  ContinueDocObj::RegisterReflection();
  ForDocObj::RegisterReflection();
  ScopeDocObj::RegisterReflection();
  ExprStmtDocObj::RegisterReflection();
  AssertDocObj::RegisterReflection();
  ReturnDocObj::RegisterReflection();
  FunctionDocObj::RegisterReflection();
  ClassDocObj::RegisterReflection();
  CommentDocObj::RegisterReflection();
  DocStringDocObj::RegisterReflection();
}

ExprDoc ExprDocObj::Attr(ffi::String attr) const {
  return AttrAccessDoc(ffi::GetRef<ExprDoc>(this), attr);
}

ExprDoc ExprDocObj::operator[](ffi::Array<Doc> indices) const {
  return IndexDoc(ffi::GetRef<ExprDoc>(this), indices);
}

ExprDoc ExprDocObj::Call(ffi::Array<ExprDoc, void> args) const {
  return CallDoc(ffi::GetRef<ExprDoc>(this), args, ffi::Array<ffi::String>(),
                 ffi::Array<ExprDoc>());
}

ExprDoc ExprDocObj::Call(ffi::Array<ExprDoc, void> args, ffi::Array<ffi::String, void> kwargs_keys,
                         ffi::Array<ExprDoc, void> kwargs_values) const {
  return CallDoc(ffi::GetRef<ExprDoc>(this), args, kwargs_keys, kwargs_values);
}

ExprDoc ExprDoc::operator[](ffi::Array<Doc> indices) const { return (*get())[indices]; }

StmtBlockDoc::StmtBlockDoc(ffi::Array<StmtDoc> stmts) {
  ffi::ObjectPtr<StmtBlockDocObj> n = ffi::make_object<StmtBlockDocObj>();
  n->stmts = stmts;
  this->data_ = std::move(n);
}

LiteralDoc::LiteralDoc(ffi::Any value, const ffi::Optional<AccessPath>& object_path) {
  ffi::ObjectPtr<LiteralDocObj> n = ffi::make_object<LiteralDocObj>();
  n->value = value;
  if (object_path.has_value()) {
    n->source_paths.push_back(object_path.value());
  }
  this->data_ = std::move(n);
}

ExprStringDoc::ExprStringDoc(ExprDoc value, const ffi::Optional<AccessPath>& object_path) {
  ffi::ObjectPtr<ExprStringDocObj> n = ffi::make_object<ExprStringDocObj>();
  n->value = value;
  if (object_path.has_value()) {
    n->source_paths.push_back(object_path.value());
  }
  this->data_ = std::move(n);
}

IdDoc::IdDoc(ffi::String name) {
  ffi::ObjectPtr<IdDocObj> n = ffi::make_object<IdDocObj>();
  n->name = name;
  this->data_ = std::move(n);
}

AttrAccessDoc::AttrAccessDoc(ExprDoc value, ffi::String name) {
  ffi::ObjectPtr<AttrAccessDocObj> n = ffi::make_object<AttrAccessDocObj>();
  n->value = value;
  n->name = name;
  this->data_ = std::move(n);
}

IndexDoc::IndexDoc(ExprDoc value, ffi::Array<Doc> indices) {
  ffi::ObjectPtr<IndexDocObj> n = ffi::make_object<IndexDocObj>();
  n->value = value;
  n->indices = indices;
  this->data_ = std::move(n);
}

CallDoc::CallDoc(ExprDoc callee, ffi::Array<ExprDoc> args, ffi::Array<ffi::String> kwargs_keys,
                 ffi::Array<ExprDoc> kwargs_values) {
  ffi::ObjectPtr<CallDocObj> n = ffi::make_object<CallDocObj>();
  n->callee = callee;
  n->args = args;
  n->kwargs_keys = kwargs_keys;
  n->kwargs_values = kwargs_values;
  this->data_ = std::move(n);
}

OperationDoc::OperationDoc(OperationDocObj::Kind kind, ffi::Array<ExprDoc> operands) {
  ffi::ObjectPtr<OperationDocObj> n = ffi::make_object<OperationDocObj>();
  n->kind = kind;
  n->operands = operands;
  this->data_ = std::move(n);
}

LambdaDoc::LambdaDoc(ffi::Array<IdDoc> args, ExprDoc body) {
  ffi::ObjectPtr<LambdaDocObj> n = ffi::make_object<LambdaDocObj>();
  n->args = args;
  n->body = body;
  this->data_ = std::move(n);
}

TupleDoc::TupleDoc(ffi::Array<ExprDoc> elements) {
  ffi::ObjectPtr<TupleDocObj> n = ffi::make_object<TupleDocObj>();
  n->elements = elements;
  this->data_ = std::move(n);
}

ListDoc::ListDoc(ffi::Array<ExprDoc> elements) {
  ffi::ObjectPtr<ListDocObj> n = ffi::make_object<ListDocObj>();
  n->elements = elements;
  this->data_ = std::move(n);
}

DictDoc::DictDoc(ffi::Array<ExprDoc> keys, ffi::Array<ExprDoc> values) {
  ffi::ObjectPtr<DictDocObj> n = ffi::make_object<DictDocObj>();
  n->keys = keys;
  n->values = values;
  this->data_ = std::move(n);
}

SliceDoc::SliceDoc(ffi::Optional<ExprDoc> start, ffi::Optional<ExprDoc> stop,
                   ffi::Optional<ExprDoc> step) {
  ffi::ObjectPtr<SliceDocObj> n = ffi::make_object<SliceDocObj>();
  n->start = start;
  n->stop = stop;
  n->step = step;
  this->data_ = std::move(n);
}

AssignDoc::AssignDoc(ExprDoc lhs, ffi::Optional<ExprDoc> rhs, ffi::Optional<ExprDoc> annotation) {
  TVM_FFI_CHECK(rhs.has_value() || annotation.has_value(), ValueError)
      << "At least one of rhs and annotation needs to be non-null for AssignDoc.";
  TVM_FFI_CHECK(lhs->IsInstance<IdDocObj>() || annotation == nullptr, ValueError)
      << "annotation can only be nonnull if lhs is an identifier.";

  ffi::ObjectPtr<AssignDocObj> n = ffi::make_object<AssignDocObj>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->annotation = annotation;
  this->data_ = std::move(n);
}

IfDoc::IfDoc(ExprDoc predicate, ffi::Array<StmtDoc> then_branch,
             ffi::Optional<ffi::Array<StmtDoc>> else_branch) {
  ffi::ObjectPtr<IfDocObj> n = ffi::make_object<IfDocObj>();
  n->predicate = predicate;
  n->then_branch = then_branch;
  n->else_branch = else_branch;
  this->data_ = std::move(n);
}

WhileDoc::WhileDoc(ExprDoc predicate, ffi::Array<StmtDoc> body) {
  ffi::ObjectPtr<WhileDocObj> n = ffi::make_object<WhileDocObj>();
  n->predicate = predicate;
  n->body = body;
  this->data_ = std::move(n);
}

BreakDoc::BreakDoc() {
  ffi::ObjectPtr<BreakDocObj> n = ffi::make_object<BreakDocObj>();
  this->data_ = std::move(n);
}

ContinueDoc::ContinueDoc() {
  ffi::ObjectPtr<ContinueDocObj> n = ffi::make_object<ContinueDocObj>();
  this->data_ = std::move(n);
}

ForDoc::ForDoc(ExprDoc lhs, ExprDoc rhs, ffi::Array<StmtDoc> body) {
  ffi::ObjectPtr<ForDocObj> n = ffi::make_object<ForDocObj>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->body = body;
  this->data_ = std::move(n);
}

ScopeDoc::ScopeDoc(ffi::Optional<ExprDoc> lhs, ExprDoc rhs, ffi::Array<StmtDoc> body,
                   bool allow_concise_scoping) {
  ffi::ObjectPtr<ScopeDocObj> n = ffi::make_object<ScopeDocObj>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->body = body;
  n->allow_concise_scoping = allow_concise_scoping;
  this->data_ = std::move(n);
}

ScopeDoc::ScopeDoc(ExprDoc rhs, ffi::Array<StmtDoc> body) {
  ffi::ObjectPtr<ScopeDocObj> n = ffi::make_object<ScopeDocObj>();
  n->lhs = std::nullopt;
  n->rhs = rhs;
  n->body = body;
  this->data_ = std::move(n);
}

ExprStmtDoc::ExprStmtDoc(ExprDoc expr) {
  ffi::ObjectPtr<ExprStmtDocObj> n = ffi::make_object<ExprStmtDocObj>();
  n->expr = expr;
  this->data_ = std::move(n);
}

AssertDoc::AssertDoc(ExprDoc test, ffi::Optional<ExprDoc> msg) {
  ffi::ObjectPtr<AssertDocObj> n = ffi::make_object<AssertDocObj>();
  n->test = test;
  n->msg = msg;
  this->data_ = std::move(n);
}

ReturnDoc::ReturnDoc(ExprDoc value) {
  ffi::ObjectPtr<ReturnDocObj> n = ffi::make_object<ReturnDocObj>();
  n->value = value;
  this->data_ = std::move(n);
}

FunctionDoc::FunctionDoc(IdDoc name, ffi::Array<AssignDoc> args, ffi::Array<ExprDoc> decorators,
                         ffi::Optional<ExprDoc> return_type, ffi::Array<StmtDoc> body,
                         ffi::Array<Doc> type_params) {
  ffi::ObjectPtr<FunctionDocObj> n = ffi::make_object<FunctionDocObj>();
  n->name = name;
  n->args = args;
  n->decorators = decorators;
  n->return_type = return_type;
  n->body = body;
  n->type_params = type_params;
  this->data_ = std::move(n);
}

ClassDoc::ClassDoc(IdDoc name, ffi::Array<ExprDoc> decorators, ffi::Array<StmtDoc> body) {
  ffi::ObjectPtr<ClassDocObj> n = ffi::make_object<ClassDocObj>();
  n->name = name;
  n->decorators = decorators;
  n->body = body;
  this->data_ = std::move(n);
}

CommentDoc::CommentDoc(ffi::String comment) {
  ffi::ObjectPtr<CommentDocObj> n = ffi::make_object<CommentDocObj>();
  n->comment = comment;
  this->data_ = std::move(n);
}

DocStringDoc::DocStringDoc(ffi::String docs) {
  ffi::ObjectPtr<DocStringDocObj> n = ffi::make_object<DocStringDocObj>();
  n->comment = docs;
  this->data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.DocSetSourcePaths",
      [](Doc doc, ffi::Array<AccessPath> source_paths) { doc->source_paths = source_paths; });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("script.printer.ExprDocAttr", &ExprDocObj::Attr)
      .def_method("script.printer.ExprDocIndex", &ExprDocObj::operator[])
      .def_method("script.printer.ExprDocCall",
                  [](ExprDoc doc, ffi::Array<ExprDoc> args, ffi::Array<ffi::String> kwargs_keys,
                     ffi::Array<ExprDoc> kwargs_values) {
                    return doc->Call(args, kwargs_keys, kwargs_values);
                  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.StmtDocSetComment",
      [](StmtDoc doc, ffi::Optional<ffi::String> comment) { doc->comment = comment; });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.StmtBlockDoc",
                        [](ffi::Array<StmtDoc> stmts) { return StmtBlockDoc(stmts); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.printer.LiteralDocNone", LiteralDoc::None)
      .def("script.printer.LiteralDocInt",
           static_cast<LiteralDoc (*)(int64_t, const ffi::Optional<AccessPath>&)>(LiteralDoc::Int))
      .def("script.printer.LiteralDocBoolean", LiteralDoc::Boolean)
      .def("script.printer.LiteralDocFloat", LiteralDoc::Float)
      .def("script.printer.LiteralDocStr", LiteralDoc::Str);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.IdDoc", [](ffi::String name) { return IdDoc(name); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.AttrAccessDoc",
                        [](ExprDoc value, ffi::String attr) { return AttrAccessDoc(value, attr); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.IndexDoc", [](ExprDoc value, ffi::Array<Doc> indices) {
    return IndexDoc(value, indices);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.CallDoc", [](ExprDoc callee,                       //
                                                     ffi::Array<ExprDoc> args,             //
                                                     ffi::Array<ffi::String> kwargs_keys,  //
                                                     ffi::Array<ExprDoc> kwargs_values) {
    return CallDoc(callee, args, kwargs_keys, kwargs_values);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.OperationDoc",
                        [](int32_t kind, ffi::Array<ExprDoc> operands) {
                          return OperationDoc(OperationDocObj::Kind(kind), operands);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.LambdaDoc",
                        [](ffi::Array<IdDoc> args, ExprDoc body) { return LambdaDoc(args, body); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.TupleDoc",
                        [](ffi::Array<ExprDoc> elements) { return TupleDoc(elements); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ListDoc",
                        [](ffi::Array<ExprDoc> elements) { return ListDoc(elements); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.DictDoc",
      [](ffi::Array<ExprDoc> keys, ffi::Array<ExprDoc> values) { return DictDoc(keys, values); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.SliceDoc",
                        [](ffi::Optional<ExprDoc> start, ffi::Optional<ExprDoc> stop,
                           ffi::Optional<ExprDoc> step) { return SliceDoc(start, stop, step); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.AssignDoc", [](ExprDoc lhs, ffi::Optional<ExprDoc> rhs,
                                                       ffi::Optional<ExprDoc> annotation) {
    return AssignDoc(lhs, rhs, annotation);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.IfDoc",
                        [](ExprDoc predicate, ffi::Array<StmtDoc> then_branch,
                           ffi::Optional<ffi::Array<StmtDoc>> else_branch) {
                          return IfDoc(predicate, then_branch, else_branch);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.WhileDoc", [](ExprDoc predicate, ffi::Array<StmtDoc> body) {
    return WhileDoc(predicate, body);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.BreakDoc", []() { return BreakDoc(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ContinueDoc", []() { return ContinueDoc(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.ForDoc",
      [](ExprDoc lhs, ExprDoc rhs, ffi::Array<StmtDoc> body) { return ForDoc(lhs, rhs, body); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ScopeDoc",
                        [](ffi::Optional<ExprDoc> lhs, ExprDoc rhs, ffi::Array<StmtDoc> body) {
                          return ScopeDoc(lhs, rhs, body);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ExprStmtDoc",
                        [](ExprDoc expr) { return ExprStmtDoc(expr); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.AssertDoc",
      [](ExprDoc test, ffi::Optional<ExprDoc> msg = std::nullopt) { return AssertDoc(test, msg); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ReturnDoc", [](ExprDoc value) { return ReturnDoc(value); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.FunctionDoc",
                        [](IdDoc name, ffi::Array<AssignDoc> args, ffi::Array<ExprDoc> decorators,
                           ffi::Optional<ExprDoc> return_type, ffi::Array<StmtDoc> body,
                           ffi::Array<Doc> type_params = {}) {
                          return FunctionDoc(name, args, decorators, return_type, body,
                                             type_params);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ClassDoc",
                        [](IdDoc name, ffi::Array<ExprDoc> decorators, ffi::Array<StmtDoc> body) {
                          return ClassDoc(name, decorators, body);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.CommentDoc",
                        [](ffi::String comment) { return CommentDoc(comment); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.DocStringDoc",
                        [](ffi::String docs) { return DocStringDoc(docs); });
}

NamespaceDoc::NamespaceDoc(ffi::String canonical_name) {
  auto node = ffi::make_object<NamespaceDocObj>();
  node->canonical_name = (canonical_name == "ir" || canonical_name == "prim")
                             ? ffi::String("tirx")
                             : std::move(canonical_name);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() { NamespaceDocObj::RegisterReflection(); }

}  // namespace printer
}  // namespace script
}  // namespace tvm
