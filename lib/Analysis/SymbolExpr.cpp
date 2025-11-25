#include "Analysis/SymbolExpr.h"

#include <mlir/Support/LLVM.h>
#include <ostream>

using namespace lleq;

struct impl::SymbolBase {
  virtual std::ostream &print(std::ostream &os) = 0;
};

struct Unknown : impl::SymbolBase {
  size_t n;
  Unknown(size_t n) : impl::SymbolBase{}, n{n} {}
  std::ostream &print(std::ostream &os) override {
    os << '?' << n;
    return os;
  }
};

struct Constant : impl::SymbolBase {
  int value;
  Constant(int value) : impl::SymbolBase{}, value{value} {}
  std::ostream &print(std::ostream &os) override {
    os << value;
    return os;
  }
};

struct TemplParam : impl::SymbolBase {
  std::string_view name;
  TemplParam(std::string_view name) : impl::SymbolBase{}, name{name} {}
  std::ostream &print(std::ostream &os) override {
    os << '@' << name;
    return os;
  }
};

struct Index : impl::SymbolBase {
  mlir::Value signal;
  Symbol n;

  Index(mlir::Value signal, Symbol n)
      : impl::SymbolBase{}, signal{signal}, n{n} {}
  std::ostream &print(std::ostream &os) override {
    // TODO: print the MLIR value too
    os << "sig[";
    n->print(os) << "]";
    return os;
  }
};

struct Arith : impl::SymbolBase {
  Symbol lhs, rhs;
  char op;

  Arith(Symbol lhs, Symbol rhs, char op)
      : impl::SymbolBase{}, lhs{lhs}, rhs{rhs}, op{op} {}
  std::ostream &print(std::ostream &os) {
    os << '(';
    lhs->print(os) << op;
    rhs->print(os) << ')';
    return os;
  }
};

Symbol SymbolPool::fresh_unknown() {
  static std::size_t n;
  return pool.alloc<Unknown>(n++);
}

Symbol SymbolPool::constant(int value) { return pool.alloc<Constant>(value); }
Symbol SymbolPool::templ_param(std::string_view name) {
  return pool.alloc<TemplParam>(name);
}
Symbol SymbolPool::index(mlir::Value signal, Symbol n) {
  return pool.alloc<Index>(signal, n);
}
Symbol SymbolPool::arith(Symbol lhs, Symbol rhs, char op) {
  return pool.alloc<Arith>(lhs, rhs, op);
}

std::ostream &operator<<(std::ostream &os, Symbol s) { return s->print(os); }
