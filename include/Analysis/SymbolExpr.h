/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <initializer_list>
#include <llvm/ADT/APInt.h>
#include <memory_resource>
#include <mlir/IR/Value.h>

namespace lleq {

namespace impl {
struct SymbolBase {
  virtual std::ostream &print(std::ostream &os) const = 0;
  virtual unsigned hash_value() const = 0;
};
}; // namespace impl

// Represents a symbolic expression assigned to a signal (as defined on pg.11 of
// https://www.cs.utexas.edu/~isil/zequal.pdf)
using Symbol = impl::SymbolBase *;

// A "pool" of symbolic expressions that all refer to each other
class SymbolPool {
  std::pmr::monotonic_buffer_resource memory;
  std::pmr::polymorphic_allocator<Symbol> alloc;

public:
  SymbolPool() : alloc{&memory} {}

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
};

// The "join" operation defined in Fig. 15 of
// https://www.cs.utexas.edu/~isil/zequal.pdf
Symbol join(Symbol a, Symbol b);

}; // namespace lleq

std::ostream &operator<<(std::ostream &os, lleq::Symbol s);
namespace llvm {
inline unsigned hash_value(lleq::Symbol s) { return s->hash_value(); }
} // namespace llvm
