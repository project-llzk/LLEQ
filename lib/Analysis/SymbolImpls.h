#pragma once

#include "Analysis/SymbolExpr.h"

namespace lleq {
struct SymbolPool;
struct Unknown : impl::SymbolEq<Unknown> {
  size_t n;
  Unknown(SymbolPool *pool, size_t n) : impl::SymbolEq<Unknown>{pool}, n{n} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << '?' << n;
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("unknown", n);
  }
  bool operator==(const Unknown &other) const { return n == other.n; }
};

struct Constant : impl::SymbolEq<Constant> {
  mlir::APInt value;
  Constant(SymbolPool *pool, mlir::APInt value)
      : impl::SymbolEq<Constant>{pool}, value{value} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    // Print it as signed
    value.print(os, true);
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("constant", value);
  }
  bool operator==(const Constant &other) const { return value == other.value; }
};

struct TemplParam : impl::SymbolEq<TemplParam> {
  std::string name;
  TemplParam(SymbolPool *pool, llvm::StringRef name)
      : impl::SymbolEq<TemplParam>{pool}, name{name} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << '@' << name.data();
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("template", name);
  }
  bool operator==(const TemplParam &other) const { return name == other.name; }
};

struct Index : impl::SymbolEq<Index> {
  mlir::Value signal;
  llvm::SmallVector<Symbol> indices;

  Index(SymbolPool *pool, mlir::Value signal, llvm::ArrayRef<Symbol> ns)
      : impl::SymbolEq<Index>{pool}, signal{signal}, indices{ns} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    // TODO: print the MLIR value too
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
  bool operator==(const Index &other) const {
    return signal == other.signal && indices.size() == other.indices.size() &&
           std::equal(indices.begin(), indices.end(), other.indices.begin(),
                      [](auto *a, auto *b) { return *a == *b; });
  }
};

struct OpCall : impl::SymbolEq<OpCall> {
  llvm::SmallVector<Symbol> arguments;
  std::string opName;

  OpCall(SymbolPool *pool, llvm::StringRef opName,
         llvm::ArrayRef<Symbol> arguments)
      : impl::SymbolEq<OpCall>{pool}, arguments{arguments}, opName{opName} {}

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
  bool operator==(const OpCall &other) const {
    return opName == other.opName &&
           arguments.size() == other.arguments.size() &&
           std::equal(arguments.begin(), arguments.end(),
                      other.arguments.begin(),
                      [](auto *a, auto *b) { return *a == *b; });
  }
};
} // namespace lleq
