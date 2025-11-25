#pragma once

#include "Util/Arena.h"
#include <cstddef>
#include <mlir/IR/Value.h>

namespace lleq {

namespace impl {
struct SymbolBase;
};

using Symbol = impl::SymbolBase *;

class SymbolPool {
  Arena<2048> pool;

public:
  Symbol fresh_unknown();
  Symbol constant(int value);
  Symbol templ_param(std::string_view name);
  Symbol index(mlir::Value signal, Symbol n);
  Symbol arith(Symbol lhs, Symbol rhs, char op);
};

Symbol join(Symbol a, Symbol b);

}; // namespace lleq

std::ostream &operator<<(std::ostream &os, lleq::Symbol s);
