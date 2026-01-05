#pragma once

#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"

#include <functional>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>

namespace lleq {

class SymbolValue {
  Symbol sym;

public:
  // Needs to be default-constructible
  SymbolValue() : sym{nullptr} {}
  SymbolValue(Symbol sym) : sym{sym} {}

  SymbolValue &join(const SymbolValue &other) {
    sym = anti_unify(sym, other.sym);
    return *this;
  }

  static SymbolValue join(const SymbolValue &a, const SymbolValue &b) {
    if (!a.sym) {
      return b;
    }
    if (!b.sym) {
      return a;
    }
    return {anti_unify(a.sym, b.sym)};
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
    sym.sym->print(os);
    return os;
  }

  operator Symbol() const { return sym; }
};

// class SVALatticeValue
//     : public mlir::dataflow::AbstractLatticeValue<SVALatticeValue,
//                                                   SymbolValue> {
// public:
//   using AbstractLatticeValue::AbstractLatticeValue;
// };

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
    pool.get().index(lattice->getAnchor(), {});
  }
};

} // namespace lleq
