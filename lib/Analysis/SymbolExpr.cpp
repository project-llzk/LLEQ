/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolExpr.h"

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <memory_resource>
#include <mlir/IR/Block.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace lleq;

struct Unknown : impl::SymbolBase {
  size_t n;
  Unknown(SymbolPool *pool, size_t n) : impl::SymbolBase{pool}, n{n} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << '?' << n;
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("unknown", n);
  }
};

struct Constant : impl::SymbolBase {
  mlir::APInt value;
  Constant(SymbolPool *pool, mlir::APInt value)
      : impl::SymbolBase{pool}, value{value} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    // Print it as signed
    value.print(os, true);
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("constant", value);
  }
};

struct TemplParam : impl::SymbolBase {
  std::string name;
  TemplParam(SymbolPool *pool, llvm::StringRef name)
      : impl::SymbolBase{pool}, name{name} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << '@' << name.data();
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("template", name);
  }
};

struct Index : impl::SymbolBase {
  mlir::Value signal;
  llvm::SmallVector<Symbol> indices;

  Index(SymbolPool *pool, mlir::Value signal, llvm::ArrayRef<Symbol> ns)
      : impl::SymbolBase{pool}, signal{signal}, indices{ns} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << pool->getNameForValue(signal);
    for (auto n : indices) {
      os << "[";
      n->print(os) << "]";
    }
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine(
        "indices", signal,
        llvm::hash_combine_range(indices.begin(), indices.end()));
  }
};

struct OpCall : impl::SymbolBase {
  llvm::SmallVector<Symbol> arguments;
  std::string opName;

  OpCall(SymbolPool *pool, llvm::StringRef opName,
         llvm::ArrayRef<Symbol> arguments)
      : impl::SymbolBase{pool}, arguments{arguments}, opName{opName} {}

  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << opName << "(";
    for (unsigned i = 0; i < arguments.size() - 1; i++) {
      os << arguments[i] << ", ";
    }
    os << arguments[arguments.size() - 1];
    os << ")";
    return os;
  }

  unsigned hash_value() const override {
    return llvm::hash_combine(
        opName, llvm::hash_combine_range(arguments.begin(), arguments.end()));
  }
};

Symbol SymbolPool::fresh_unknown() {
  static std::size_t n;
  return alloc.new_object<Unknown>(this, n++);
}

Symbol SymbolPool::constant(mlir::APInt value) {
  return alloc.new_object<Constant>(this, value);
}
Symbol SymbolPool::templ_param(llvm::StringRef name) {
  return alloc.new_object<TemplParam>(this, name);
}
Symbol SymbolPool::index(mlir::Value signal, llvm::ArrayRef<Symbol> ns) {
  return alloc.new_object<Index>(this, signal, ns);
}
Symbol SymbolPool::func_call(llvm::StringRef name,
                             llvm::ArrayRef<Symbol> args) {
  return alloc.new_object<OpCall>(this, name, args);
}

std::string SymbolPool::getNameForValue(mlir::Value value) {
  static unsigned value_number = 0u;
  std::string result;
  llvm::raw_string_ostream ss(result);
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    ss << llvm::format("%%arg%u", blockArg.getArgNumber());
  } else {
    ss << llvm::format("%%val%u", value_number++);
  }
  return result;
}
std::string SymbolPool::_gen_name(mlir::Value value) {
  static mlir::DenseMap<mlir::Value, std::string> valueNameMap;
  if (!valueNameMap.contains(value)) {
    valueNameMap[value] = _gen_name(value);
  }
  return valueNameMap[value];
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Symbol s) {
  return s->print(os);
}
