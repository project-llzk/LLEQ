#pragma once

#include <cstddef>
#include <initializer_list>
#include <llvm/ADT/APInt.h>
#include <memory_resource>
#include <mlir/IR/Value.h>

namespace lleq {

namespace impl {
struct SymbolBase;
};

// Represents a symbolic expression assigned to a signal (as defined on pg.11 of
// https://www.cs.utexas.edu/~isil/zequal.pdf)
using Symbol = impl::SymbolBase *;

static constexpr auto ALLOWED_OPS = {'+', '-', '*'};

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
  Symbol templ_param(std::string_view name);
  // Indexing expression into an N-dimensional array
  Symbol index(mlir::Value signal, std::initializer_list<Symbol> ns);
  // Arithmetic
  Symbol arith(Symbol lhs, Symbol rhs, char op);
};

// The "join" operation defined in Fig. 15 of
// https://www.cs.utexas.edu/~isil/zequal.pdf
Symbol join(Symbol a, Symbol b);

}; // namespace lleq

std::ostream &operator<<(std::ostream &os, lleq::Symbol s);
