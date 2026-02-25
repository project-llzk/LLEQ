/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolExpr.h"

#include "Analysis/SymbolImpls.h"
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Util/ErrorHelper.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace lleq;

Symbol SymbolPool::copy(Symbol s) {
  return llvm::TypeSwitch<Symbol, Symbol>(s)
      .Case<Unknown>(
          [this](Unknown *s) { return alloc.new_object<Unknown>(*this, s->n); })
      .Case<Constant>([this](Constant *s) {
        return alloc.new_object<Constant>(*this, s->value);
      })
      .Case<TemplParam>([this](TemplParam *s) {
        return alloc.new_object<TemplParam>(*this, s->name);
      })
      .Case<OpCall>([this](OpCall *s) {
        llvm::SmallVector<Symbol> copiedArgs;
        for (auto arg : s->arguments) {
          copiedArgs.push_back(copy(arg));
        }
        return alloc.new_object<OpCall>(*this, s->opName, copiedArgs);
      })
      .Case<Index>([this](Index *s) {
        llvm::SmallVector<Symbol> copiedIdx;
        for (auto idx : s->indices) {
          copiedIdx.push_back(copy(idx));
        }
        return alloc.new_object<Index>(*this, s->signal, copiedIdx);
      });
}

Symbol SymbolPool::uninitialized() {
  return alloc.new_object<Uninitialized>(*this);
}

Symbol SymbolPool::fresh_unknown() {
  static std::size_t n;
  return alloc.new_object<Unknown>(*this, n++);
}

Symbol SymbolPool::constant(mlir::DynamicAPInt value) {
  return alloc.new_object<Constant>(*this, value);
}
Symbol SymbolPool::templ_param(llvm::StringRef name) {
  return alloc.new_object<TemplParam>(*this, name);
}
Symbol SymbolPool::index(mlir::Value signal, llvm::ArrayRef<Symbol> ns) {
  if (llvm::any_of(ns, [](Symbol s) { return llvm::isa<Uninitialized>(s); })) {
    return alloc.new_object<Uninitialized>(*this);
  }
  return alloc.new_object<Index>(*this, signal, ns);
}
Symbol SymbolPool::func_call(llvm::StringRef name,
                             llvm::ArrayRef<Symbol> args) {
  if (llvm::any_of(args,
                   [](Symbol s) { return llvm::isa<Uninitialized>(s); })) {
    return alloc.new_object<Uninitialized>(*this);
  }
  return alloc.new_object<OpCall>(*this, name, args);
}

Symbol SymbolPool::pod(const llvm::DenseMap<llvm::StringRef, Symbol> &entries) {
  if (llvm::any_of(entries,
                   [](auto e) { return llvm::isa<Uninitialized>(e.second); })) {
    return alloc.new_object<Uninitialized>(*this);
  }

  return alloc.new_object<Pod>(*this, entries);
}

Symbol SymbolPool::pod(llvm::ArrayRef<llvm::StringRef> keys,
                       llvm::ArrayRef<Symbol> values) {
  llzk::ensure(keys.size() == values.size(),
               "differing number of keys and values in pod!");

  llvm::DenseMap<llvm::StringRef, Symbol> entries;
  for (auto [k, v] : llvm::zip(keys, values)) {
    entries.insert({k, v});
  }
  return pod(entries);
}

Symbol SymbolPool::pod() {
  llvm::DenseMap<llvm::StringRef, Symbol> entries;
  return pod(entries);
}

std::string SymbolPool::_gen_name(mlir::Value value) const {
  static unsigned value_number = 0u;
  static unsigned block_arg_number = 0u;
  std::string result;
  llvm::raw_string_ostream ss(result);
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    ss << llvm::format("%%arg%u", block_arg_number++);
  } else {
    ss << llvm::format("%%val%u", value_number++);
  }
  return result;
}
std::string SymbolPool::getNameForValue(mlir::Value value) const {
  static llvm::DenseMap<mlir::Value, std::string> valueNameMap;
  if (!valueNameMap.contains(value)) {
    valueNameMap[value] = _gen_name(value);
  }
  return valueNameMap[value];
}

llvm::raw_ostream &lleq::operator<<(llvm::raw_ostream &os, Symbol s) {
  if (s == nullptr) {
    os << "(null)";
    return os;
  }
  return s->print(os);
}
