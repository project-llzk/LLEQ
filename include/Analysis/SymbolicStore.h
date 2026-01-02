/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/Store.h"
#include "Analysis/SymbolExpr.h"

#include <cassert>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/PointerUnion.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace lleq {
/// @brief Represents a mapping between circuit signals and symbolic
/// expressions. Each entry in the store is keyed by both the signal, which is
/// assumed to be a (possibly multidimensional) array, and a vector of symbolic
/// indices into the array, one per dimension. This is later used to statically
/// prove equivalence between pairs of witness/constraint signals.
class SymbolicStore {
  std::unique_ptr<SymbolPool> pool = std::make_unique<SymbolPool>();
  SignalStore signalStore;
  ValueStore valueStore;
  llzk::component::StructDefOp component;

  // Copy (owned) symbol(s) named `src` to local name `dest`, while correctly
  // handling scalar and array values
  template <class S, class T>
  void copy_value(S dest, T src, WriteMode mode = WriteMode::Overwrite) {
    copy_value(_get<S>(), dest, src, mode);
  }

  // Copy (owned) symbol(s) named `src` to name `dest` inside `destStore`, while
  // correctly handling scalar and array values
  template <class S, class T>
  void copy_value(Store<S> &destStore, S dest, T src,
                  WriteMode mode = WriteMode::Overwrite);

  template <class T> mlir::Type _lookup_type(T val);
  template <class T> Store<T> &_get();

  Symbol lookup(llvm::StringRef sig) {
    if (!signalStore.contains({sig, {}})) {
      signalStore.write({sig, {}}, pool->fresh_unknown());
    }
    return signalStore.at({sig, {}});
  }

public:
  SymbolicStore() : signalStore{*pool.get()}, valueStore{*pool.get()} {}
  SymbolicStore(const SymbolicStore &other);
  SymbolicStore &operator=(const SymbolicStore &other);
  bool operator==(const SymbolicStore &other) const {
    return signalStore == other.signalStore && valueStore == other.valueStore;
  }

  /// @brief Build a store from a given circuit component (struct)
  /// @param structDef
  void build_store(llzk::component::StructDefOp structDef);

  /// @brief Update the signalStore and valueStore based on a single operation
  /// @param op
  void process_operation(mlir::Operation *op);

  /// @brief Update the signalStore and valueStore based on the operations in a
  /// block. Optionally takes a vector of values that capture any values yielded
  /// by the block
  /// @param block
  /// @param yielded
  void process_block(mlir::Block *block,
                     llvm::ArrayRef<mlir::Value> yielded = {});

  /// @brief Generate a symbolic expression corresponding to an MLIR SSA value,
  /// possibly looking up values in the store to do so
  /// @param value
  Symbol lookup(mlir::Value value);

  /// @brief Pretty-print the contents of the store
  /// @param os
  void dump(llvm::raw_ostream &os) const;

  /// @brief Compute a store that represents entries from both `a` and `b`
  /// @param a
  /// @param b
  static SymbolicStore join(const SymbolicStore &a, const SymbolicStore &b);
};

} // namespace lleq
