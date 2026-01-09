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

namespace lleq {
class ValueStoreLattice : public mlir::dataflow::AbstractDenseLattice {
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
    if (!store<T>().contains(ref)) {
      return nullptr;
    }
    return store<T>().at(ref);
  }

  template <class T> Symbol lookup(Ref<T> ref) {
    auto symbol = lookupOrNull<T>(ref);
    if (!symbol) {
      return store<T>().write(ref, pool->fresh_unknown(), WriteMode::Overwrite);
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
    for (auto [key, val] : *valueStore) {
      if (!other.valueStore->contains(key) ||
          *other.valueStore->at(key) != *val) {
        return false;
      }
    }
    for (auto [key, val] : *other.valueStore) {
      if (!valueStore->contains(key) || *valueStore->at(key) != *val) {
        return false;
      }
    }
    return true;
  }

  void print(llvm::raw_ostream &os) const override {
    if (!initialized) {
      os << "(uninit)\n";
      return;
    }
    if (valueStore == nullptr || signalStore == nullptr) {
      os << "(null)\n";
      return;
    }
    if (signalStore->size() == 0) {
      os << "(empty)\n";
      return;
    }
    // for (auto [key, val] : *valueStore) {
    //   os << key << ": " << val << "\n";
    // }
    for (auto [key, val] : *signalStore) {
      os << key << ": " << static_cast<Symbol>(val) << "\n";
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
