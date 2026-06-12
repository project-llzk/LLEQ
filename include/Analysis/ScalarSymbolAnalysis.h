/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/Store.h"
#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"

#include <functional>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Analysis/SparseAnalysis.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/Constants.h>
#include <llzk/Util/ErrorHelper.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>

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
  static SymbolValue join(const SymbolValue &a, const SymbolValue &b,
                          mlir::Value anchor = nullptr) {
    if (!a.isInitialized()) {
      return b;
    }
    if (!b.isInitialized()) {
      return a;
    }

    AUTag tag;
    if (anchor != nullptr) {
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(anchor)) {
        tag = blockArg.getParentBlock()->getParentOp();
      } else {
        tag = anchor.getDefiningOp();
      }
    }

    auto joined = anti_unify(a.sym, b.sym, tag);
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
  SymbolValue value;

public:
  using Lattice::Lattice;

  mlir::Value getAnchor() const { return mlir::cast<mlir::Value>(anchor); }

  SymbolValue &getValue() { return value; }
  const SymbolValue &getValue() const {
    return const_cast<ScalarLattice *>(this)->getValue();
  }

  void print(llvm::raw_ostream &os) const override { value.print(os); }

  mlir::ChangeResult
  join(const mlir::dataflow::AbstractSparseLattice &other) override {
    return join(static_cast<const ScalarLattice &>(other).value);
  }

  mlir::ChangeResult join(SymbolValue otherValue) {
    SymbolValue newValue = SymbolValue::join(value, otherValue, getAnchor());
    if (value == newValue) {
      return mlir::ChangeResult::NoChange;
    }
    value = newValue;
    return mlir::ChangeResult::Change;
  }
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

  void visitExternalCall(mlir::CallOpInterface call,
                         llvm::ArrayRef<const Lattice *> arguments,
                         llvm::ArrayRef<Lattice *> results) override;

  void setToEntryState(Lattice *lattice) override {
    lattice->getValue().initPool(&pool.get());
    mlir::Value anchor = lattice->getAnchor();
    if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(anchor)) {
      // We don't want to treat a function argument as uninitialized/unknown
      // since all the symbolic expressions are relative to these
      if (llvm::isa<llzk::function::FuncDefOp>(
              blockArg.getOwner()->getParentOp())) {
        return propagateIfChanged(
            lattice, lattice->join(pool.get().index(lattice->getAnchor(), {})));
      }
      // Apparently setToEntryState gets called on loop induction vars too, but
      // these should just be set to Unknown because `i` and `i + step` will
      // never antiunify to anything interesting
      if (llvm::isa<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        return propagateIfChanged(lattice,
                                  lattice->join(pool.get().fresh_unknown()));
      }
    }

    // If nothing else is known, a default of `uninitialized` is fine
    return propagateIfChanged(lattice,
                              lattice->join(pool.get().uninitialized()));
  }
};

inline bool isWitnessOp(mlir::Operation *op) {
  return op->hasAttrOfType<mlir::StringAttr>("product_source") &&
         op->getAttrOfType<mlir::StringAttr>("product_source") ==
             llzk::FUNC_NAME_COMPUTE;
}
inline bool isConstraintOp(mlir::Operation *op) {
  return op->hasAttrOfType<mlir::StringAttr>("product_source") &&
         op->getAttrOfType<mlir::StringAttr>("product_source") ==
             llzk::FUNC_NAME_CONSTRAIN;
}

inline bool sourceMatchesOp(mlir::Operation *op, Signal::Source source) {
  return (source == Signal::Source::Constraint && isConstraintOp(op)) ||
         (source == Signal::Source::Witness && isWitnessOp(op));
}

inline bool isSubcmpRead(llzk::component::MemberReadOp read) {
  return read.getComponent().getType() !=
         read->getParentOfType<llzk::component::StructDefOp>().getType();
}

} // namespace lleq
