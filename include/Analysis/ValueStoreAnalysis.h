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

  mlir::ChangeResult write(llvm::StringRef ref, Symbol sym) {
    initialized = true;
    if (valueStore.contains(ref) && *valueStore.at(ref) == *sym) {
      return mlir::ChangeResult::NoChange;
    } else if (valueStore.contains(ref)) {
      valueStore[ref] = anti_unify(valueStore[ref], sym);
    } else {
      valueStore[ref] = sym;
    }
    return mlir::ChangeResult::Change;
  }

  void setPool(SymbolPool *pool) { this->pool = pool; }

  mlir::ChangeResult
  join(const mlir::dataflow::AbstractDenseLattice &other) override {
    const auto *rhs = dynamic_cast<const ValueStoreLattice *>(&other);
    pool = rhs->pool;
    if (!rhs) {
      llvm::report_fatal_error("cannot join incomparable lattices");
    }
    if (!rhs->initialized) {
      return mlir::ChangeResult::NoChange;
    }
    if (!initialized) {
      valueStore = rhs->valueStore;
      initialized = true;
      return mlir::ChangeResult::Change;
    }

    if (*this == *rhs) {
      return mlir::ChangeResult::NoChange;
    }

    for (auto [key, val] : valueStore) {
      if (!rhs->valueStore.contains(key) &&
          val->kind != impl::SymbolBase::SymbolKind::SK_Uninitialized) {
        valueStore[key] = pool->fresh_unknown();
      } else if (rhs->valueStore.contains(key)) {
        valueStore[key] = anti_unify(val, rhs->valueStore.at(key));
      }
    }

    return mlir::ChangeResult::Change;
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
