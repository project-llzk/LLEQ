#pragma once

#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"

#include <functional>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace lleq {

class SymbolValue {
  Symbol sym;
  SymbolPool *pool;

public:
  // Needs to be default-constructible
  SymbolValue() : sym{nullptr}, pool{nullptr} {}
  SymbolValue(Symbol sym) : sym{sym} {}

  void setPool(SymbolPool *pool) {
    if (this->pool == nullptr) {
      this->pool = pool;
      sym = pool->uninitialized();
    }
  }

  static SymbolValue join(const SymbolValue &a, const SymbolValue &b) {
    if (!a.sym) {
      return b;
    }
    if (!b.sym) {
      return a;
    }
    auto joined = anti_unify(a.sym, b.sym);
    return {joined};
  }

  bool operator==(const SymbolValue &other) const {
    if (!sym) {
      return !other.sym;
    }
    if (!other.sym) {
      return !sym;
    }
    return *sym == *other.sym;
  }

  void print(llvm::raw_ostream &os) const { sym->print(os); }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SymbolValue &sym) {
    if (!sym.sym) {
      os << "(null)";
      return os;
    }
    sym.sym->print(os);
    return os;
  }

  operator Symbol() const { return sym; }
};

class SignalValueDataflowAnalysis;

class SVALattice : public mlir::dataflow::Lattice<SymbolValue> {
public:
  using Lattice::Lattice;
};

class SignalValueDataflowAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<SVALattice> {
  using Lattice = SVALattice;
  using Base = SparseForwardDataFlowAnalysis<Lattice>;

  std::reference_wrapper<SymbolPool> pool;

public:
  SignalValueDataflowAnalysis(mlir::DataFlowSolver &solver, SymbolPool &pool)
      : SparseForwardDataFlowAnalysis{solver}, pool{pool} {}
  mlir::LogicalResult
  visitOperation(mlir::Operation *op, llvm::ArrayRef<const Lattice *> operands,
                 llvm::ArrayRef<Lattice *> results) override;
  void setToEntryState(Lattice *lattice) override {
    lattice->getValue().setPool(&pool.get());
    propagateIfChanged(
        lattice, lattice->join(pool.get().index(lattice->getAnchor(), {})));
  }
};

} // namespace lleq
