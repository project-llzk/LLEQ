#pragma once

#include <cstddef>
#include <memory_resource>
#include <mlir/IR/Value.h>

namespace lleq {

namespace impl {
struct SymbolBase;
};

using Symbol = impl::SymbolBase *;

class SymbolPool {
  std::pmr::monotonic_buffer_resource memory;
  std::pmr::polymorphic_allocator<Symbol> pool;

public:
  SymbolPool() : pool{&memory} {}
  Symbol fresh_unknown();
  Symbol constant(int value);
  Symbol templ_param(std::string_view name);
  Symbol index(mlir::Value signal, Symbol n);
  Symbol arith(Symbol lhs, Symbol rhs, char op);
};

Symbol join(Symbol a, Symbol b);

}; // namespace lleq

std::ostream &operator<<(std::ostream &os, lleq::Symbol s);
