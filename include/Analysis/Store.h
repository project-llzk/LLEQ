/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/SymbolExpr.h"
#include "Analysis/SymbolImpls.h"
#include "Analysis/Unification.h"
#include <algorithm>
#include <concepts>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Debug.h>

namespace lleq {

// A symbolic multidimensional index into an array
using StoreIndex = llvm::SmallVector<Symbol>;
} // namespace lleq

namespace llvm {
template <> struct DenseMapInfo<lleq::StoreIndex> {
  static inline lleq::StoreIndex getEmptyKey() {
    return {DenseMapInfo<lleq::Symbol>::getEmptyKey()};
  }

  static inline lleq::StoreIndex getTombstoneKey() {
    return {DenseMapInfo<lleq::Symbol>::getTombstoneKey()};
  }

  static unsigned getHashValue(const lleq::StoreIndex &index) {
    return llvm::hash_combine(
        index.size(), llvm::hash_combine_range(index.begin(), index.end()));
  }

  static bool isEqual(const lleq::StoreIndex &lhs,
                      const lleq::StoreIndex &rhs) {
    auto emptyKey = DenseMapInfo<lleq::Symbol>::getEmptyKey();
    auto tombstoneKey = DenseMapInfo<lleq::Symbol>::getTombstoneKey();
    auto isSentinel = [&](const lleq::StoreIndex &index, lleq::Symbol key) {
      return index.size() == 1 && index.front() == key;
    };

    if (isSentinel(lhs, emptyKey) || isSentinel(lhs, tombstoneKey) ||
        isSentinel(rhs, emptyKey) || isSentinel(rhs, tombstoneKey)) {
      return lhs == rhs;
    }

    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(), lleq::impl::equal);
  }
};
} // namespace llvm
namespace lleq {

// Represents a location in the store with a "name" pointing to a
// multidimensional array, and a symbolic index into each dimension
template <std::equality_comparable NameT> struct IndexedLocation {
  NameT name;
  StoreIndex index;

  // Returns true if the two locations might alias each other (i.e.
  // have the same name, and the symbolic indices could be the same under some
  // valuation)
  bool canAlias(const IndexedLocation<NameT> &other) const {
    return name == other.name && index.size() == other.index.size() &&
           std::equal(index.begin(), index.end(), other.index.begin(),
                      [](auto a, auto b) { return a->canEqual(*b); });
  }
};

template <std::equality_comparable T>
static inline bool operator==(const IndexedLocation<T> &a,
                              const IndexedLocation<T> &b) {
  return a.name == b.name && a.index.size() == b.index.size() &&
         std::equal(a.index.begin(), a.index.end(), b.index.begin(),
                    impl::equal);
}

template <class T>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const IndexedLocation<T> &ref) {
  os << ref.name;
  for (auto idx : ref.index) {
    os << '[' << idx << ']';
  }
  return os;
}

// Indexing into an ordinary MLIR value
using IndexedValue = IndexedLocation<mlir::Value>;

// Indexing into a struct member (either memberName or blockArgIndex)
// Note that we use the term "signal" here generically--the referenced member
// may not indeed be marked with the `signal` attr.
struct Signal {
  enum class Source { Witness, Constraint };
  Source source; // compute or constrain
  llvm::StringRef name;
  bool operator==(const Signal &other) const = default;
  operator std::string() const {
    llvm::Twine twine = name;
    return twine.concat(source == Source::Witness ? "_w" : "_c").str();
  }
};

inline unsigned hash_value(Signal sig) { return llvm::hash_value(sig.name); }

using IndexedSignal = IndexedLocation<Signal>;
} // namespace lleq

namespace llvm {

inline unsigned hash_value(mlir::Value val) {
  return llvm::hash_value(val.getAsOpaquePointer());
}

template <> struct DenseMapInfo<lleq::Signal> {
  static inline auto getEmptyKey() {
    return lleq::Signal{lleq::Signal::Source::Witness,
                        DenseMapInfo<StringRef>::getEmptyKey()};
  }
  static inline auto getTombstoneKey() {
    return lleq::Signal{lleq::Signal::Source::Witness,
                        DenseMapInfo<StringRef>::getTombstoneKey()};
  }
  static bool isEqual(auto LHS, auto RHS) { return LHS == RHS; }
};

template <std::equality_comparable T> struct IndexedLocationInfo {
  static inline lleq::IndexedLocation<T> getEmptyKey() {
    return {llvm::DenseMapInfo<T>::getEmptyKey(), {}};
  }

  static inline lleq::IndexedLocation<T> getTombstoneKey() {
    return {llvm::DenseMapInfo<T>::getTombstoneKey(), {}};
  }

  static unsigned getHashValue(const lleq::IndexedLocation<T> &Val) {
    return llvm::hash_combine(Val.name, Val.index.size());
  }
  static bool isEqual(const lleq::IndexedLocation<T> &LHS,
                      const lleq::IndexedLocation<T> &RHS) {
    return LHS == RHS;
  }
};

template <>
struct DenseMapInfo<lleq::IndexedValue>
    : public IndexedLocationInfo<mlir::Value> {};

template <>
struct DenseMapInfo<lleq::IndexedSignal>
    : public IndexedLocationInfo<lleq::Signal> {};

} // namespace llvm

namespace lleq {
template <class T> struct Store;

// For every pair of entries (arr, phi_1) :- x; (arr, phi_2) :- y from `a` and
// `b`, add an entry (arr, AU(phi_1, phi_2)) :- AU(x, y) to the result, where
// AU represents antiunification
template <class T>
Store<T> _join_stores(const Store<T> &a, const Store<T> &b, SymbolPool &);

// What action to take when writing to an existing entry: `OverwriteExact`
// overwrites the entry, `AntiUnify` performs anti-unification with the existing
// entry and the new value, and `HavocAliases` overwrites the existing entry
// and sets all other entries that could alias it to Unknown
enum class WriteMode { OverwriteExact, AntiUnify, HavocAliases };

/// Represents a single mapping of refs (names + indices) to symbols.
/// Lightweight wrapper around a DenseMap<Ref<T>, Symbol> that provides some
/// convenience methods
template <class T> struct Store {
  // Whether this store has ever been written to
  bool initialized = false;

  // Support iterating over the underlying store
  auto begin() const { return _store.begin(); }
  auto end() const { return _store.end(); }
  decltype(auto) at(const IndexedLocation<T> &ref) const {
    return _store.at(ref);
  }
  decltype(auto) find(const IndexedLocation<T> ref) const {
    return _store.find(ref);
  }
  bool contains(const IndexedLocation<T> &ref) const {
    return _store.contains(ref);
  }
  bool canContain(IndexedLocation<T> ref) const {
    for (auto [key, _] : _store) {
      if (key.canAlias(ref)) {
        return true;
      }
    }
    return false;
  }
  auto size() const { return _store.size(); }

  // Configure behavior when writing to a name that is already present
  // (overwrite vs. anti-unify)
  Symbol write(const IndexedLocation<T> &ref, Symbol val,
               WriteMode mode = WriteMode::OverwriteExact) {
    initialized = true;
    if (&val->pool != &_pool.get()) {
      val = _pool.get().copy(val);
    }

    if (mode == WriteMode::HavocAliases) {
      // Start by clobbering any possible aliases
      llvm::SmallVector<IndexedLocation<T>> keys;
      keys.reserve(_store.size());
      for (auto [key, _] : _store) {
        keys.push_back(key);
      }
      for (auto key : keys) {
        if (ref.canAlias(key)) {
          _store[key] = _pool.get().fresh_unknown();
        }
      }
      _store[ref] = val;
    } else if (!contains(ref) || mode == WriteMode::OverwriteExact) {
      _store[ref] = val;
    } else if (mode == WriteMode::AntiUnify) {
      _store[ref] = anti_unify(_store[ref], val);
    } else {
      assert(false && "unsupported write mode");
    }
    return _store[ref];
  }

  Store(llvm::DenseMap<IndexedLocation<T>, Symbol> entries, SymbolPool &pool)
      : _store{entries}, _pool{pool} {}
  Store(SymbolPool &pool) : _store{}, _pool{pool} {}

  // Create a deep copy backed by `pool`
  Store<T> clone(SymbolPool &pool) const {
    Store<T> cloned{pool};
    for (const auto &[ref, val] : _store) {
      IndexedLocation<T> clonedRef{ref.name, {}};
      for (auto idx : ref.indices) {
        clonedRef.index.push_back(pool.copy(idx));
      }
      cloned.write(clonedRef, val);
    }
    return cloned;
  }

  bool operator==(const Store<T> &other) const {
    for (auto [ref, val] : _store) {
      if (!other.contains(ref) || *other.at(ref) != *val) {
        return false;
      }
    }

    for (auto [ref, val] : other._store) {
      if (!contains(ref) || *at(ref) != *val) {
        return false;
      }
    }

    return true;
  }

  void join_with(const Store<T> &other) {
    if (!initialized || !other.initialized) {
      return;
    }

    for (auto [key, val] : _store) {
      if (!other.contains(key) && !llvm::isa<Uninitialized>(val)) {
        _store.erase(key);
      } else if (other.canContain(key)) {
        auto aliasedEntries = llvm::filter_to_vector(
            other, [key](auto entry) { return key.canAlias(entry.first); });
        for (auto [otherKey, otherVal] : aliasedEntries) {
          write(key, otherVal, WriteMode::AntiUnify);
        }
      }
    }
  }

  void clear() { _store.clear(); }

private:
  llvm::DenseMap<IndexedLocation<T>, Symbol> _store;
  std::reference_wrapper<SymbolPool> _pool;
};

using SignalStore = Store<Signal>;
using ValueStore = Store<mlir::Value>;

} // namespace lleq
