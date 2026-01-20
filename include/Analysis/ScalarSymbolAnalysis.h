/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"

#include <functional>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Analysis/SparseAnalysis.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/ErrorHelper.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>

namespace lleq {

class SymbolValue {
  Symbol sym;

public:
  // Needs to be default-constructible
  SymbolValue() : sym{nullptr} {}
  SymbolValue(Symbol sym) : sym{sym} {}

  void initPool(SymbolPool *pool) {
    llzk::ensure(pool != nullptr, "pool cannot be null");
    if (sym == nullptr) {
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

class ScalarSymbolAnalysis;

class ScalarLattice : public mlir::dataflow::Lattice<SymbolValue> {
public:
  using Lattice::Lattice;
};

class ScalarSymbolAnalysis
    : public llzk::dataflow::SparseForwardDataFlowAnalysis<ScalarLattice> {
  using Lattice = ScalarLattice;
  using Base = SparseForwardDataFlowAnalysis<Lattice>;

  std::reference_wrapper<SymbolPool> pool;

public:
  ScalarSymbolAnalysis(mlir::DataFlowSolver &solver, SymbolPool &pool)
      : SparseForwardDataFlowAnalysis{solver}, pool{pool} {}
  mlir::LogicalResult
  visitOperation(mlir::Operation *op, llvm::ArrayRef<const Lattice *> operands,
                 llvm::ArrayRef<Lattice *> results) override;
  void setToEntryState(Lattice *lattice) override {
    lattice->getValue().initPool(&pool.get());
    mlir::Value anchor = lattice->getAnchor();
    if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(anchor)) {
      if (llvm::isa<llzk::function::FuncDefOp>(
              blockArg.getOwner()->getParentOp())) {
        propagateIfChanged(
            lattice, lattice->join(pool.get().index(lattice->getAnchor(), {})));
        return;
      }
      // Apparently setToEntryState gets called on loop induction vars too, but
      // these should just be set to Unknown
      if (llvm::isa<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        propagateIfChanged(lattice, lattice->join(pool.get().fresh_unknown()));
      }
    }
    propagateIfChanged(lattice, lattice->join(pool.get().uninitialized()));
  }
};

} // namespace lleq
