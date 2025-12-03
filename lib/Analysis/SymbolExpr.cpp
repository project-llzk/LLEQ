#include "Analysis/SymbolExpr.h"

#include <algorithm>
#include <initializer_list>
#include <llvm/Support/raw_os_ostream.h>
#include <memory_resource>
#include <mlir/Support/LLVM.h>
#include <ostream>
#include <vector>

using namespace lleq;

struct impl::SymbolBase {
  virtual std::ostream &print(std::ostream &os) const = 0;
};

struct Unknown : impl::SymbolBase {
  size_t n;
  Unknown(size_t n) : impl::SymbolBase{}, n{n} {}
  std::ostream &print(std::ostream &os) const override {
    os << '?' << n;
    return os;
  }
};

struct Constant : impl::SymbolBase {
  mlir::APInt value;
  Constant(mlir::APInt value) : impl::SymbolBase{}, value{value} {}
  std::ostream &print(std::ostream &os) const override {
    // Print it as signed
    llvm::raw_os_ostream ros(os);
    value.print(ros, true);
    return os;
  }
};

struct TemplParam : impl::SymbolBase {
  std::string_view name;
  TemplParam(std::string_view name) : impl::SymbolBase{}, name{name} {}
  std::ostream &print(std::ostream &os) const override {
    os << '@' << name;
    return os;
  }
};

struct Index : impl::SymbolBase {
  mlir::Value signal;
  std::pmr::vector<Symbol> indices;

  // Takes a pointer to the memory resource backing the SymbolPool so we can
  // reuse it for the vector of indices
  Index(std::pmr::memory_resource *memory, mlir::Value signal,
        std::initializer_list<Symbol> ns)
      : impl::SymbolBase{}, signal{signal} {
    indices = std::pmr::vector<Symbol>{ns.begin(), ns.end(), memory};
  }
  std::ostream &print(std::ostream &os) const override {
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
  std::ostream &print(std::ostream &os) const override {
    os << '(';
    lhs->print(os) << op;
    rhs->print(os) << ')';
    return os;
  }
};

Symbol SymbolPool::fresh_unknown() const {
  static std::size_t n;
  return alloc.new_object<Unknown>(n++);
}

Symbol SymbolPool::constant(mlir::APInt value) const {
  return alloc.new_object<Constant>(value);
}
Symbol SymbolPool::templ_param(std::string_view name) const {
  return alloc.new_object<TemplParam>(name);
}
Symbol SymbolPool::index(mlir::Value signal,
                         std::initializer_list<Symbol> ns) const {
  return alloc.new_object<Index>(&memory, signal, ns);
}
Symbol SymbolPool::arith(Symbol lhs, Symbol rhs, char op) const {
  if (std::find(ALLOWED_OPS.begin(), ALLOWED_OPS.end(), op) ==
      ALLOWED_OPS.end()) {
    // TODO: What's a better way of signaling an error here? An exception?
    return nullptr;
  }
  return alloc.new_object<Arith>(lhs, rhs, op);
}

std::ostream &operator<<(std::ostream &os, Symbol s) { return s->print(os); }
