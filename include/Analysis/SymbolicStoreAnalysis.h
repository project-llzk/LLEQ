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
#include <llzk/Analysis/DenseAnalysis.h>
#include <mlir/Support/TypeID.h>

namespace lleq {
class SymbolicStoreAnalysis;

/// Represents a lattice element that tracks a `valueStore`, which maps
/// array-typed SSA values to symbolic expressions, and a `signalStore` which
/// maps struct signals to symbolic expressions. The stores are keyed by
/// `IndexedLocation`s, which track the name of the entry and an optional list
/// of symbolic indices (one per dimension of the array)
class StoreLattice : public mlir::dataflow::AbstractDenseLattice {
  friend class SymbolicStoreAnalysis;
  std::unique_ptr<ValueStore> valueStore;
  std::unique_ptr<SignalStore> signalStore;

  SymbolPool *pool = nullptr;
  bool initialized = false;

  template <class T> decltype(auto) store(this auto &&self) {
    static_assert(std::is_same_v<T, mlir::Value> ||
                  std::is_same_v<T, Signal> && "unsupported store type");
    if constexpr (std::is_same_v<T, mlir::Value>) {
      return *self.valueStore;
    } else if constexpr (std::is_same_v<T, Signal>) {
      return *self.signalStore;
    }
  }

  template <class T>
  mlir::ChangeResult _write_impl(IndexedLocation<T> ref, Symbol sym);

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  void initPool(SymbolPool *pool) {
    if (!this->pool) {
      this->pool = pool;
      valueStore = std::make_unique<ValueStore>(*pool);
      signalStore = std::make_unique<SignalStore>(*pool);
    }
  }

  // Look up the symbol written to a particular location, or `nullptr` if
  // nothing has been written
  template <class T> Symbol lookupOrNull(IndexedLocation<T> ref) const {
    if (!initialized || !store<T>().contains(ref)) {
      return nullptr;
    }
    return store<T>().at(ref);
  }

  // Look up the symbol written to a particular location, or update and return a
  // fresh UNKNOWN
  template <class T> Symbol lookup(IndexedLocation<T> ref) {
    initialized = true;
    auto symbol = lookupOrNull<T>(ref);
    if (!symbol) {
      return store<T>().write(ref, pool->fresh_unknown(),
                              WriteMode::OverwriteExact);
    }
    return symbol;
  }

  mlir::ChangeResult write(IndexedSignal ref, Symbol sym) {
    return _write_impl(ref, sym);
  }

  mlir::ChangeResult write(IndexedValue ref, Symbol sym) {
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

/// This implements a dense analysis that populates the symbolic stores at every
/// program point. It is mutually dependent on `ScalarSymbolAnalysis` for struct
/// field reads/writes
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
    lattice->initPool(&pool);
  }

  // Looks up the symbol SignalValueAnalysis computed for the given SSA value,
  // and subscribes to any updates to the symbol
  Symbol getBoundSymbol(mlir::Value value);
};
} // namespace lleq
