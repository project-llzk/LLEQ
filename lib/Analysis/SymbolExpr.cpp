/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolExpr.h"

#include <algorithm>
#include <llvm/ADT/Hashing.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <memory_resource>
#include <mlir/Support/LLVM.h>
#include <ostream>

using namespace lleq;

struct Unknown : impl::SymbolBase {
  size_t n;
  Unknown(size_t n) : impl::SymbolBase{}, n{n} {}
  std::ostream &print(std::ostream &os) const override {
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
  std::ostream &print(std::ostream &os) const override {
    // Print it as signed
    llvm::raw_os_ostream ros(os);
    value.print(ros, true);
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("constant", value);
  }
};

struct TemplParam : impl::SymbolBase {
  std::string name;
  TemplParam(llvm::StringRef name) : impl::SymbolBase{}, name{name} {}
  std::ostream &print(std::ostream &os) const override {
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
  std::ostream &print(std::ostream &os) const override {
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

  std::ostream &print(std::ostream &os) const override {
    os << opName << "(";
    std::copy(arguments.begin(), arguments.end(),
              std::ostream_iterator<Symbol>(os, ","));
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

std::ostream &operator<<(std::ostream &os, Symbol s) { return s->print(os); }
