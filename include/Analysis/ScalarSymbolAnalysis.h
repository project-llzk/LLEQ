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

/// This is a wrapper around `Symbol` that provides a `join` method for use in
/// the ScalarSymbolAnalysis lattice
class SymbolValue {
  Symbol sym;

public:
  // Needs to be default-constructible, so we treat `nullptr` as an
  // uninitialized symbol
  SymbolValue() : sym{nullptr} {}
  SymbolValue(Symbol sym) : sym{sym} {}

  bool isInitialized() const { return sym != nullptr; }

  void initPool(SymbolPool *pool) {
    llzk::ensure(pool != nullptr, "pool cannot be null");
    if (sym == nullptr) {
      sym = pool->uninitialized();
    }
  }

  // Joining is just antiunification, once we take care to special-case the
  // nullptr values
  static SymbolValue join(const SymbolValue &a, const SymbolValue &b) {
    if (!a.isInitialized()) {
      return b;
    }
    if (!b.isInitialized()) {
      return a;
    }
    auto joined = anti_unify(a.sym, b.sym);
    return {joined};
  }

  bool operator==(const SymbolValue &other) const {
    if (isInitialized() && other.isInitialized()) {
      return *sym == *other.sym;
    }
    return isInitialized() == other.isInitialized();
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

class ScalarLattice : public mlir::dataflow::Lattice<SymbolValue> {
public:
  using Lattice::Lattice;
};

/// This analysis computes a symbolic expression associated to each *scalar*
/// (i.e., not array-typed) SSA value. It is mutually dependent on the
/// SymbolicStoreAnalysis to handle struct field reads/writes.
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
      // We don't want to treat a function argument as uninitialized/unknown
      // since all the symbolic expressions are relative to these
      if (llvm::isa<llzk::function::FuncDefOp>(
              blockArg.getOwner()->getParentOp())) {
        propagateIfChanged(
            lattice, lattice->join(pool.get().index(lattice->getAnchor(), {})));
        return;
      }
      // Apparently setToEntryState gets called on loop induction vars too, but
      // these should just be set to Unknown because `i` and `i + step` will
      // never antiunify to anything interesting
      if (llvm::isa<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        propagateIfChanged(lattice, lattice->join(pool.get().fresh_unknown()));
      }
    }
    // If nothing else is known, a default of `uninitialized` is fine
    propagateIfChanged(lattice, lattice->join(pool.get().uninitialized()));
  }
};

} // namespace lleq
