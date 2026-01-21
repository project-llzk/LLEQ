/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/Store.h"
#include "Analysis/SymbolExpr.h"
#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Support/TypeID.h>

namespace lleq {
class SymbolicStoreAnalysis;
class StoreLattice : public mlir::dataflow::AbstractDenseLattice {
  friend class SymbolicStoreAnalysis;
  std::unique_ptr<ValueStore> valueStore;
  std::unique_ptr<SignalStore> signalStore;

  SymbolPool *pool;
  bool initialized = false;

  template <class T> decltype(auto) store(this auto &&self) {
    static_assert(std::is_same_v<T, mlir::Value> ||
                  std::is_same_v<T, llvm::StringRef> &&
                      "unsupported store type");
    if constexpr (std::is_same_v<T, mlir::Value>) {
      return *self.valueStore;
    } else if constexpr (std::is_same_v<T, llvm::StringRef>) {
      return *self.signalStore;
    }
  }

  template <class T>
  mlir::ChangeResult _write_impl(IndexedLocation<T> ref, Symbol sym);

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  void setPool(SymbolPool *pool) {
    if (!this->pool) {
      this->pool = pool;
      valueStore = std::make_unique<ValueStore>(*pool);
      signalStore = std::make_unique<SignalStore>(*pool);
    }
  }

  template <class T> Symbol lookupOrNull(IndexedLocation<T> ref) const {
    if (!initialized || !store<T>().contains(ref)) {
      return nullptr;
    }
    return store<T>().at(ref);
  }

  template <class T> Symbol lookup(IndexedLocation<T> ref) {
    initialized = true;
    auto symbol = lookupOrNull<T>(ref);
    if (!symbol) {
      return store<T>().write(ref, pool->fresh_unknown(),
                              WriteMode::OverwriteExact);
    }
    return symbol;
  }

  mlir::ChangeResult write(SignalRef ref, Symbol sym) {
    return _write_impl(ref, sym);
  }

  mlir::ChangeResult write(ValueRef ref, Symbol sym) {
    return _write_impl(ref, sym);
  }

  template <class T> auto write(T val, Symbol sym) {
    return _write_impl(IndexedLocation<T>{val, {}}, sym);
  }

  template <class T> auto lookup(T ref) {
    return lookup(IndexedLocation<T>{ref, {}});
  }
  template <class T> auto lookupOrNull(T ref) const {
    return lookupOrNull(IndexedLocation<T>{ref, {}});
  }

  mlir::ChangeResult
  join(const mlir::dataflow::AbstractDenseLattice &other) override;

  bool operator==(const StoreLattice &other) const {
    return *valueStore == *other.valueStore &&
           *signalStore == *other.signalStore;
  }

  void print(llvm::raw_ostream &os) const override;

  std::pair<SignalStore *, ValueStore *> getStores() const {
    return {signalStore.get(), valueStore.get()};
  };
};
class SymbolicStoreAnalysis
    : public mlir::dataflow::DenseForwardDataFlowAnalysis<StoreLattice> {
  SymbolPool &pool;

public:
  SymbolicStoreAnalysis(mlir::DataFlowSolver &solver, SymbolPool &pool)
      : DenseForwardDataFlowAnalysis{solver}, pool{pool} {}
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;
  mlir::LogicalResult visitOperation(mlir::Operation *op,
                                     const StoreLattice &before,
                                     StoreLattice *after) override;
  void setToEntryState(StoreLattice *lattice) override {
    lattice->setPool(&pool);
  }

  // Looks up the symbol SignalValueAnalysis computed for the given SSA value,
  // and subscribes to any updates to the symbol
  Symbol getBoundSymbol(mlir::Value value);
};
} // namespace lleq
