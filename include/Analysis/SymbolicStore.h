/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/SymbolExpr.h"

#include <algorithm>
#include <concepts>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/PointerUnion.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace lleq {

template <std::equality_comparable NameT> struct Ref {
  NameT name;
  llvm::SmallVector<Symbol> indices;
};

template <std::equality_comparable T>
static inline bool operator==(const Ref<T> &a, const Ref<T> &b) {
  return a.name == b.name && a.indices.size() == b.indices.size() &&
         std::equal(a.indices.begin(), a.indices.end(), b.indices.begin(),
                    [](auto *a, auto *b) { return *a == *b; });
}

// Indexing into an ordinary MLIR value
using ValueRef = Ref<mlir::Value>;
// Indexing into a struct signal (either fieldName or blockArgIndex)
using SignalRef = Ref<llvm::StringRef>;
} // namespace lleq

namespace llvm {

inline unsigned hash_value(mlir::Value val) {
  return llvm::hash_value(val.getAsOpaquePointer());
}

template <std::equality_comparable T> struct RefInfo {
  static inline lleq::Ref<T> getEmptyKey() {
    return {llvm::DenseMapInfo<T>::getEmptyKey(), {}};
  }

  static inline lleq::Ref<T> getTombstoneKey() {
    return {llvm::DenseMapInfo<T>::getTombstoneKey(), {}};
  }

  static unsigned getHashValue(const lleq::Ref<T> &Val) {
    return llvm::hash_combine(Val.name, Val.indices.size());
  }
  static bool isEqual(const lleq::Ref<T> &LHS, const lleq::Ref<T> &RHS) {
    return LHS == RHS;
  }
};

template <>
struct DenseMapInfo<lleq::ValueRef> : public RefInfo<mlir::Value> {};

template <>
struct DenseMapInfo<lleq::SignalRef> : public RefInfo<llvm::StringRef> {};

} // namespace llvm

namespace lleq {

/// @brief Represents a mapping between circuit signals and symbolic
/// expressions. Each entry in the store is keyed by both the signal, which is
/// assumed to be a (possibly multidimensional) array, and a vector of symbolic
/// indices into the array, one per dimension. This is later used to statically
/// prove equivalence between pairs of witness/constraint signals.
class SymbolicStore {
  std::unique_ptr<SymbolPool> pool = std::make_unique<SymbolPool>();
  llvm::DenseMap<SignalRef, Symbol> signalStore;
  llvm::DenseMap<ValueRef, Symbol> valueStore;

public:
  SymbolicStore() {}
  SymbolicStore(const SymbolicStore &other);
  SymbolicStore &operator=(const SymbolicStore &other);

  /// @brief Build a store from a given circuit component (struct)
  /// @param structDef
  void build_store(llzk::component::StructDefOp structDef);

  /// @brief Update the signalStore and valueStore based on a single operation
  /// @param op
  void process_operation(mlir::Operation *op);

  /// @brief Update the signalStore and valueStore based on the operations in a
  /// block. If the block yields a value, returns the corresponding symbol.
  /// @param block
  void process_block(mlir::Block *block,
                     std::optional<mlir::Value> yielded = std::nullopt);

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
