#pragma once

#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Support/TypeID.h>

namespace lleq {
class ValueStoreLattice : public mlir::dataflow::AbstractDenseLattice {
  llvm::DenseMap<llvm::StringRef, Symbol> valueStore;
  SymbolPool *pool;
  bool initialized = false;

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  void setPool(SymbolPool *pool) {
    if (!this->pool) {
      this->pool = pool;
    }
  }

  Symbol lookup(llvm::StringRef ref) {
    if (!valueStore.contains(ref)) {
      valueStore.insert({ref, pool->fresh_unknown()});
    }
    return valueStore.at(ref);
  }

  Symbol lookupOrNull(llvm::StringRef ref) const {
    if (!valueStore.contains(ref)) {
      return nullptr;
    }
    return valueStore.at(ref);
  }

  mlir::ChangeResult write(llvm::StringRef ref, Symbol sym);

  mlir::ChangeResult
  join(const mlir::dataflow::AbstractDenseLattice &other) override;

  bool operator==(const ValueStoreLattice &other) const {
    for (auto [key, val] : valueStore) {
      if (!other.valueStore.contains(key) ||
          *other.valueStore.at(key) != *val) {
        return false;
      }
    }
    for (auto [key, val] : other.valueStore) {
      if (!valueStore.contains(key) || *valueStore.at(key) != *val) {
        return false;
      }
    }
    return true;
  }

  void print(llvm::raw_ostream &os) const override {
    for (auto [key, val] : valueStore) {
      os << key << ": " << val << "\n";
    }
  }
};
class ValueStoreAnalysis
    : public mlir::dataflow::DenseForwardDataFlowAnalysis<ValueStoreLattice> {
  SymbolPool &pool;

public:
  ValueStoreAnalysis(mlir::DataFlowSolver &solver, SymbolPool &pool)
      : DenseForwardDataFlowAnalysis{solver}, pool{pool} {}
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;
  mlir::LogicalResult visitOperation(mlir::Operation *op,
                                     const ValueStoreLattice &before,
                                     ValueStoreLattice *after) override;
  void setToEntryState(ValueStoreLattice *lattice) override {
    lattice->setPool(&pool);
  }
};
} // namespace lleq
