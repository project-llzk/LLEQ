#include "Analysis/SymbolExpr.h"

#include <initializer_list>
#include <memory_resource>
#include <mlir/Support/LLVM.h>
#include <ostream>
#include <vector>

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
  std::pmr::vector<Symbol> indices;

  Index(std::pmr::memory_resource *memory, mlir::Value signal,
        std::initializer_list<Symbol> ns)
      : impl::SymbolBase{}, signal{signal} {
    indices = std::pmr::vector<Symbol>{memory};
    for (Symbol n : ns) {
      indices.push_back(n);
    }
  }
  std::ostream &print(std::ostream &os) override {
    // TODO: print the MLIR value too
    os << "sig";
    for (auto n : indices) {
      os << "[";
      n->print(os) << "]";
    }
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
  return alloc.new_object<Unknown>(n++);
}

Symbol SymbolPool::constant(int value) {
  return alloc.new_object<Constant>(value);
}
Symbol SymbolPool::templ_param(std::string_view name) {
  return alloc.new_object<TemplParam>(name);
}
Symbol SymbolPool::index(mlir::Value signal, std::initializer_list<Symbol> ns) {
  return alloc.new_object<Index>(&memory, signal, ns);
}
Symbol SymbolPool::arith(Symbol lhs, Symbol rhs, char op) {
  return alloc.new_object<Arith>(lhs, rhs, op);
}

std::ostream &operator<<(std::ostream &os, Symbol s) { return s->print(os); }
