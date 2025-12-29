/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"

#include <algorithm>
#include <cassert>
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

enum class WriteMode { Overwrite, AntiUnify };
template <class T> struct Store {
  auto begin() const { return _store.begin(); }
  auto end() const { return _store.end(); }
  decltype(auto) at(Ref<T> ref) const { return _store.at(ref); }
  bool contains(Ref<T> ref) const { return _store.contains(ref); }
  auto size() const { return _store.size(); }

  void write(Ref<T> ref, Symbol val, WriteMode mode = WriteMode::Overwrite) {
    if (&val->pool != &_pool)
      val = _pool.copy(val);
    if (!contains(ref) || mode == WriteMode::Overwrite)
      _store[ref] = val;
    _store[ref] = anti_unify(_store[ref], val);
  }

  Store(llvm::DenseMap<Ref<T>, Symbol> entries, SymbolPool &pool)
      : _store{entries}, _pool{pool} {}
  Store(SymbolPool &pool) : _store{}, _pool{pool} {}
  Store(const Store<T> &other) : _pool{other._pool}, _store{other._store} {}
  Store &operator=(const Store<T> &other) {
    assert(&_pool == &other._pool &&
           "cannot assign to store backed by different SymbolPool");
    _store = other._store;
    return *this;
  }

  void widen(const Store<T> &other) {
    // TODO: update when _join_stores works in-place
    *this = _join_stores(*this, other, _pool);
  }

private:
  llvm::DenseMap<Ref<T>, Symbol> _store;
  SymbolPool &_pool;
};

// template <class T> using Store = llvm::DenseMap<Ref<T>, Symbol>;
using SignalStore = Store<llvm::StringRef>;
using ValueStore = Store<mlir::Value>;

// template struct Store<llvm::StringRef>;
// template struct Store<mlir::Value>;

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

  // Utilities for metaprogramming
  template <class S, class T>
  void copy_value(S dest, T src, WriteMode mode = WriteMode::Overwrite) {
    copy_value(_get<S>(), dest, src, mode);
  }
  template <class S, class T>
  void copy_value(Store<S> &destStore, S dest, T src,
                  WriteMode mode = WriteMode::Overwrite);
  template <class T>
  void write_value(Store<T> &store, Ref<T> ref, Symbol value, WriteMode mode);
  template <class T> mlir::Type _lookup_type(T val);
  template <class T> Store<T> &_get();
  Symbol lookup(llvm::StringRef sig) {
    if (!signalStore.contains({sig, {}}))
      signalStore.write({sig, {}}, pool->fresh_unknown());
    return signalStore.at({sig, {}});
  }

public:
  SymbolicStore() : signalStore{*pool.get()}, valueStore{*pool.get()} {}
  SymbolicStore(const SymbolicStore &other);
  SymbolicStore &operator=(const SymbolicStore &other);

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

// Helpful utilities
template <class T>
Store<T> _join_stores(const Store<T> &a, const Store<T> &b, SymbolPool &);
} // namespace lleq
