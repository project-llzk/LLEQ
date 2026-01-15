#pragma once

#include "Analysis/SymbolExpr.h"
#include "Analysis/SymbolicStore.h"
#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Support/TypeID.h>

#define DEBUG_TYPE "value-store-analysis"

namespace lleq {
class ValueStoreAnalysis;
class ValueStoreLattice : public mlir::dataflow::AbstractDenseLattice {
  friend class ValueStoreAnalysis;
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
  // template <auto> decltype(auto) store<mlir::Value>(this auto &&self) {
  //   return *self.valueStore;
  // }
  // template <auto> Store<llvm::StringRef> &store(this auto &&self) {
  //   return *self.signalStore;
  // }

  template <class T> mlir::ChangeResult _write_impl(Ref<T> ref, Symbol sym);

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  void setPool(SymbolPool *pool) {
    if (!this->pool) {
      this->pool = pool;
      valueStore = std::make_unique<ValueStore>(*pool);
      signalStore = std::make_unique<SignalStore>(*pool);
    }
  }

  template <class T> Symbol lookupOrNull(Ref<T> ref) const {
    if (!initialized || !store<T>().contains(ref)) {
      return nullptr;
    }
    return store<T>().at(ref);
  }

  template <class T> Symbol lookup(Ref<T> ref) {
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
    return _write_impl(Ref<T>{val, {}}, sym);
  }

  template <class T> auto lookup(T ref) { return lookup(Ref<T>{ref, {}}); }
  template <class T> auto lookupOrNull(T ref) const {
    return lookupOrNull(Ref<T>{ref, {}});
  }

  mlir::ChangeResult
  join(const mlir::dataflow::AbstractDenseLattice &other) override;

  bool operator==(const ValueStoreLattice &other) const {
    return *valueStore == *other.valueStore &&
           *signalStore == *other.signalStore;
  }

  void print(llvm::raw_ostream &os) const override {
    LLVM_DEBUG({
      if (!initialized) {
        os << "(uninit)\n";
        return;
      }
      if (valueStore == nullptr || signalStore == nullptr) {
        os << "(null)\n";
        return;
      }
      os << "--\n";
      if (valueStore->size() == 0) {
        os << "(empty)\n";
      }
      for (auto [key, val] : *valueStore) {
        os << key << ": " << val << "\n";
      }
      os << "--\n";
      if (signalStore->size() == 0) {
        os << "(empty)\n";
      }
    });
    for (auto [key, val] : *signalStore) {
      os << key << ": " << static_cast<Symbol>(val) << "\n";
    }
    // os << "--\n";
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

  // Looks up the symbol SignalValueAnalysis computed for the given SSA value,
  // and subscribes to any updates to the symbol
  Symbol getBoundSymbol(mlir::Value value);
};
} // namespace lleq
