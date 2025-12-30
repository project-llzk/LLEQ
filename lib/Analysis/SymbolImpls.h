#pragma once

#include "Analysis/SymbolExpr.h"
#include <llvm/ADT/DynamicAPInt.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

namespace lleq {
struct SymbolPool;
struct Unknown : public impl::SymbolEq<Unknown> {
  unsigned n;
  Unknown(SymbolPool &pool, unsigned n)
      : impl::SymbolEq<Unknown>{pool, SymbolKind::SK_Unknown}, n{n} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << '?' << n;
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("unknown", n);
  }
  bool operator==(const Unknown &other) const { return n == other.n; }
  static bool classof(const impl::SymbolBase *sym) {
    return sym->kind == SymbolKind::SK_Unknown;
  }
};

struct Constant : public impl::SymbolEq<Constant> {
  mlir::DynamicAPInt value;
  Constant(SymbolPool &pool, mlir::DynamicAPInt value)
      : impl::SymbolEq<Constant>{pool, SymbolKind::SK_Const}, value{value} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    value.print(os);
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("constant", value);
  }
  bool operator==(const Constant &other) const { return value == other.value; }
  static bool classof(const impl::SymbolBase *sym) {
    return sym->kind == SymbolKind::SK_Const;
  }
};

struct TemplParam : public impl::SymbolEq<TemplParam> {
  std::string name;
  TemplParam(SymbolPool &pool, llvm::StringRef name)
      : impl::SymbolEq<TemplParam>{pool, SymbolKind::SK_TemplParam},
        name{name} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << '@' << name.data();
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("template", name);
  }
  bool operator==(const TemplParam &other) const { return name == other.name; }
  static bool classof(const impl::SymbolBase *sym) {
    return sym->kind == SymbolKind::SK_TemplParam;
  }
};

struct Index : public impl::SymbolEq<Index> {
  mlir::Value signal;
  llvm::SmallVector<Symbol> indices;

  Index(SymbolPool &pool, mlir::Value signal, llvm::ArrayRef<Symbol> ns)
      : impl::SymbolEq<Index>{pool, SymbolKind::SK_Index}, signal{signal},
        indices{ns} {}
  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    // TODO: print the MLIR value too
    os << pool.getNameForValue(signal);
    for (auto n : indices) {
      os << '[';
      n->print(os) << ']';
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
  static bool classof(const impl::SymbolBase *sym) {
    return sym->kind == SymbolKind::SK_Index;
  }
};

struct OpCall : public impl::SymbolEq<OpCall> {
  llvm::SmallVector<Symbol> arguments;
  std::string opName;

  OpCall(SymbolPool &pool, llvm::StringRef opName,
         llvm::ArrayRef<Symbol> arguments)
      : impl::SymbolEq<OpCall>{pool, SymbolKind::SK_Call}, arguments{arguments},
        opName{opName} {}

  llvm::raw_ostream &print(llvm::raw_ostream &os) const override {
    os << opName << '(';
    for (unsigned i = 0; i < arguments.size() - 1; i++) {
      os << arguments[i] << ", ";
    }
    os << arguments[arguments.size() - 1];
    os << ')';
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
  static bool classof(const impl::SymbolBase *sym) {
    return sym->kind == SymbolKind::SK_Call;
  }
};
} // namespace lleq
