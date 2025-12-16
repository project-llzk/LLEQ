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
#include <mlir/Support/LLVM.h>

using namespace lleq;

struct Unknown : impl::SymbolBase {
  size_t n;
  Unknown(size_t n) : impl::SymbolBase{}, n{n} {}
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
  Constant(mlir::APInt value) : impl::SymbolBase{}, value{value} {}
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
  TemplParam(llvm::StringRef name) : impl::SymbolBase{}, name{name} {}
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

  Index(mlir::Value signal, llvm::ArrayRef<Symbol> ns)
      : impl::SymbolBase{}, signal{signal}, indices{ns} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    // TODO: print the MLIR value too
    os << "sig";
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

  OpCall(llvm::StringRef opName, llvm::ArrayRef<Symbol> arguments)
      : arguments{arguments}, opName{opName} {}

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
  return alloc.new_object<Unknown>(n++);
}

Symbol SymbolPool::constant(mlir::APInt value) {
  return alloc.new_object<Constant>(value);
}
Symbol SymbolPool::templ_param(llvm::StringRef name) {
  return alloc.new_object<TemplParam>(name);
}
Symbol SymbolPool::index(mlir::Value signal, llvm::ArrayRef<Symbol> ns) {
  return alloc.new_object<Index>(signal, ns);
}
Symbol SymbolPool::func_call(llvm::StringRef name,
                             llvm::ArrayRef<Symbol> args) {
  return alloc.new_object<OpCall>(name, args);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Symbol s) {
  return s->print(os);
}
