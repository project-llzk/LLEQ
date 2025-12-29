/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/raw_ostream.h>
#include <memory_resource>
#include <mlir/IR/Value.h>

namespace lleq {
class SymbolPool;
namespace impl {
struct SymbolBase {
  enum class SymbolKind {
    SK_Unknown,
    SK_Const,
    SK_TemplParam,
    SK_Index,
    SK_Call
  };
  SymbolKind kind;

  SymbolBase(SymbolPool &pool, SymbolKind k) : pool{pool}, kind{k} {}
  virtual llvm::raw_ostream &print(llvm::raw_ostream &os) const = 0;
  virtual unsigned hash_value() const = 0;

  bool operator==(const SymbolBase &other) const {
    if (kind != other.kind) {
      return false;
    }
    return eq(other);
  }

  SymbolPool &pool;
  virtual bool eq(const SymbolBase &other) const = 0;
};

template <class SymbolT> struct SymbolEq : public SymbolBase {
  SymbolEq(SymbolPool &pool, SymbolKind k) : SymbolBase{pool, k} {}
  bool eq(const SymbolBase &other) const override {
    assert(kind == other.kind);
    return static_cast<const SymbolT &>(*this) ==
           static_cast<const SymbolT &>(other);
  }
};

}; // namespace impl

// Represents a symbolic expression assigned to a signal (as defined on pg.11 of
// https://www.cs.utexas.edu/~isil/zequal.pdf)
using Symbol = impl::SymbolBase *;
using SymbolConst = const impl::SymbolBase *;

// A "pool" of symbolic expressions that all refer to each other
class SymbolPool {
  std::pmr::monotonic_buffer_resource memory;
  std::pmr::polymorphic_allocator<Symbol> alloc;

  std::string _gen_name(mlir::Value value) const;

public:
  SymbolPool() : alloc{&memory} {}

  // Create and own a deep-copy of `s` (useful for transferring between pools)
  Symbol copy(Symbol s);
  // Generate a pretty-printable name corresponding to the SSA value
  std::string getNameForValue(mlir::Value value) const;
  // A fresh symbolic variable
  Symbol fresh_unknown();
  // Arbitrary-precision felt
  Symbol constant(mlir::APInt value);
  // Struct template parameter
  Symbol templ_param(llvm::StringRef name);
  // Indexing expression into an N-dimensional array
  Symbol index(mlir::Value signal, llvm::ArrayRef<Symbol> ns);
  // Handles operations like function calls and arithmetic
  Symbol func_call(llvm::StringRef name, llvm::ArrayRef<Symbol> args);
  // The "join" operation defined in Fig. 15 of
  // https://www.cs.utexas.edu/~isil/zequal.pdf
  Symbol join(Symbol a, Symbol b);
};

}; // namespace lleq

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, lleq::Symbol s);

namespace llvm {
inline unsigned hash_value(lleq::Symbol s) { return s->hash_value(); }
} // namespace llvm
