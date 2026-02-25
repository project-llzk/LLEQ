/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <memory_resource>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace lleq {
class SymbolPool;
namespace impl {
struct SymbolBase {
  enum class SymbolKind {
    // Corresponds top the "bottom" element of the symbolic lattice
    SK_Uninitialized,
    // Fresh unknown variables corresponding to the "top" element of
    // the symbolic lattice
    SK_Unknown,
    // A numberic constant, such as one produced by felt.const or arith.constant
    SK_Const,
    // Struct template parameter
    SK_TemplParam,
    // Indexing into a (concrete) multi-dimensional array with a sequence of
    // (symbolic) indices
    SK_Index,
    // Also encodes basic arithmetic, e.g. as "felt.add(sym1, sym2)"
    SK_Call,
    // Represents a plain-old-data object (i.e. a !pod.type) potentially
    // containing multiple symbols
    SK_Pod
  };
  SymbolKind kind;

  SymbolBase(SymbolPool &pool, SymbolKind k) : pool{pool}, kind{k} {}
  virtual llvm::raw_ostream &print(llvm::raw_ostream &os) const = 0;
  void dump() const {
    print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
  virtual unsigned hash_value() const = 0;

  bool operator==(const SymbolBase &other) const {
    if (kind != other.kind) {
      return false;
    }
    return eq(other);
  }

  SymbolPool &pool;
  virtual bool eq(const SymbolBase &other) const = 0;

  // TODO: This is correct but imprecise. For more precision, ask solver
  virtual bool canEqual(const SymbolBase &other) const { return true; }
};

inline bool equal(SymbolBase *a, SymbolBase *b) {
  if (!a || !b) {
    return a == b;
  }
  return *a == *b;
}

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
  // An uninitialized symbol, useful for abstract interpretation
  Symbol uninitialized();
  // A fresh symbolic variable
  Symbol fresh_unknown();
  // Arbitrary-precision felt
  Symbol constant(mlir::DynamicAPInt value);
  // Struct template parameter
  Symbol templ_param(llvm::StringRef name);
  // Indexing expression into an N-dimensional array
  Symbol index(mlir::Value signal, llvm::ArrayRef<Symbol> ns);
  // Handles operations like function calls and arithmetic
  Symbol func_call(llvm::StringRef name, llvm::ArrayRef<Symbol> args);
  // A plain-old-data object (!pod.type)
  Symbol pod(llvm::ArrayRef<llvm::StringRef> keys,
             llvm::ArrayRef<Symbol> values);
  Symbol pod(const llvm::DenseMap<llvm::StringRef, Symbol> &entries);
  // An empty POD
  Symbol pod();
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, lleq::Symbol s);

}; // namespace lleq

namespace llvm {
inline unsigned hash_value(lleq::Symbol s) { return s->hash_value(); }
} // namespace llvm
