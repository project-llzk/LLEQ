#include "Analysis/SymbolExpr.h"

#include <algorithm>
#include <llvm/ADT/Hashing.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <memory_resource>
#include <mlir/Support/LLVM.h>
#include <ostream>
#include <vector>

using namespace lleq;

struct Unknown : impl::SymbolBase {
  size_t n;
  Unknown(size_t n) : impl::SymbolBase{}, n{n} {}
  std::ostream &print(std::ostream &os) const override {
    os << '?' << n;
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("unknown", n);
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
  unsigned hash_value() const override {
    return llvm::hash_combine("constant", value);
  }
};

struct TemplParam : impl::SymbolBase {
  llvm::StringRef name;
  TemplParam(llvm::StringRef name) : impl::SymbolBase{}, name{name} {}
  std::ostream &print(std::ostream &os) const override {
    os << '@' << name.data();
    return os;
  }
  unsigned hash_value() const override {
    return llvm::hash_combine("template", name);
  }
};

struct Index : impl::SymbolBase {
  mlir::Value signal;
  std::pmr::vector<Symbol> indices;

  // Takes a pointer to the memory resource backing the SymbolPool so we can
  // reuse it for the vector of indices
  Index(std::pmr::memory_resource *memory, mlir::Value signal,
        llvm::ArrayRef<Symbol> ns)
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
  unsigned hash_value() const override {
    return llvm::hash_combine(
        "indices", signal,
        llvm::hash_combine_range(indices.begin(), indices.end()));
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
  unsigned hash_value() const override {
    return llvm::hash_combine("arith", lhs, op, rhs);
  }
};

Symbol SymbolPool::fresh_unknown() {
  static std::size_t n;
  return alloc.new_object<Unknown>(n++);
}

Symbol SymbolPool::constant(mlir::APInt value) {
  return alloc.new_object<Constant>(value);
}
Symbol SymbolPool::templ_param(llvm::StringRef name) {
  return alloc.new_object<TemplParam>(name);
}
Symbol SymbolPool::index(mlir::Value signal, llvm::ArrayRef<Symbol> ns) {
  return alloc.new_object<Index>(&memory, signal, ns);
}
Symbol SymbolPool::arith(Symbol lhs, Symbol rhs, char op) {
  if (std::find(ALLOWED_OPS.begin(), ALLOWED_OPS.end(), op) ==
      ALLOWED_OPS.end()) {
    std::string message;
    llvm::raw_string_ostream s(message);
    s << "Illegal operation: " << op;
    llvm::report_fatal_error(message.c_str());
    return nullptr;
  }
  return alloc.new_object<Arith>(lhs, rhs, op);
}

std::ostream &operator<<(std::ostream &os, Symbol s) { return s->print(os); }
