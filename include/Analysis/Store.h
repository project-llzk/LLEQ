
#include "Analysis/SymbolExpr.h"
#include "Analysis/SymbolImpls.h"
#include "Analysis/Unification.h"
#include <algorithm>
#include <concepts>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>

namespace lleq {

template <std::equality_comparable NameT> struct Ref {
  NameT name;
  llvm::SmallVector<Symbol> indices;
  bool canEqual(const Ref<NameT> &other) const {
    return name == other.name && indices.size() == other.indices.size() &&
           std::equal(indices.begin(), indices.end(), other.indices.begin(),
                      [](auto a, auto b) { return a->canEqual(*b); });
  }
};

template <std::equality_comparable T>
static inline bool operator==(const Ref<T> &a, const Ref<T> &b) {
  return a.name == b.name && a.indices.size() == b.indices.size() &&
         std::equal(a.indices.begin(), a.indices.end(), b.indices.begin(),
                    impl::equal);
}

template <class T>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Ref<T> &ref) {
  os << ref.name;
  for (auto idx : ref.indices) {
    os << '[' << idx << ']';
  }
  return os;
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
template <class T> struct Store;

// For every pair of entries (arr, phi_1) :- x; (arr, phi_2) :- y from `a` and
// `b`, add an entry (arr, AU(phi_1, phi_2)) :- AU(x, y) to the result, where
// AU represents antiunification
template <class T>
Store<T> _join_stores(const Store<T> &a, const Store<T> &b, SymbolPool &);

// What action to take when writing to an existing entry: `OverwriteExact`
// overwrites the entry, `AntiUnify` performs anti-unification with the existing
// entry and the new value, and `HavocEquivalent` overwrites the existing entry
// and sets all other entries that could alias it to Unknown
enum class WriteMode { OverwriteExact, AntiUnify, HavocEquivalent };

/// Represents a single mapping of refs (names + indices) to symbols.
/// Lightweight wrapper around a DenseMap<Ref<T>, Symbol> that provides some
/// convenience methods
template <class T> struct Store {
  // Support iterating over the underlying store
  auto begin() const { return _store.begin(); }
  auto end() const { return _store.end(); }
  decltype(auto) at(const Ref<T> &ref) const { return _store.at(ref); }
  bool contains(const Ref<T> &ref) const { return _store.contains(ref); }
  bool canContain(Ref<T> ref) const {
    for (auto [key, _] : _store) {
      if (key.canEqual(ref)) {
        return true;
      }
    }
    return false;
  }
  auto size() const { return _store.size(); }

  // Configure behavior when writing to a name that is already present
  // (overwrite vs. anti-unify)
  Symbol write(const Ref<T> &ref, Symbol val,
               WriteMode mode = WriteMode::OverwriteExact) {
    if (&val->pool != &_pool.get()) {
      val = _pool.get().copy(val);
    }

    if (mode == WriteMode::HavocEquivalent) {
      // Start by clobbering any possible aliases
      for (auto [key, old] : _store) {
        if (ref.canEqual(key)) {
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

  Store(llvm::DenseMap<Ref<T>, Symbol> entries, SymbolPool &pool)
      : _store{entries}, _pool{pool} {}
  Store(SymbolPool &pool) : _store{}, _pool{pool} {}

  // Create a deep copy backed by `pool`
  Store<T> clone(SymbolPool &pool) const {
    Store<T> cloned{pool};
    for (const auto &[ref, val] : _store) {
      Ref<T> clonedRef{ref.name, {}};
      for (auto idx : ref.indices) {
        clonedRef.indices.push_back(pool.copy(idx));
      }
      cloned.write(clonedRef, val);
    }
    return cloned;
  }

  // Widen to include symbols from another store
  void widen(const Store<T> &other) {
    // TODO: update when _join_stores works in-place
    *this = _join_stores(*this, other, _pool);
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
    for (auto [key, val] : _store) {
      if (!other.contains(key) && !llvm::isa<Uninitialized>(val)) {
        write(key, _pool.get().fresh_unknown());
      } else if (other.canContain(key)) {
        auto aliasedEntries = llvm::filter_to_vector(
            other, [key](auto entry) { return key.canEqual(entry.first); });
        for (auto [otherKey, otherVal] : aliasedEntries) {
          write(key, otherVal, WriteMode::AntiUnify);
        }
      }
    }
  }

private:
  llvm::DenseMap<Ref<T>, Symbol> _store;
  std::reference_wrapper<SymbolPool> _pool;
};

using SignalStore = Store<llvm::StringRef>;
using ValueStore = Store<mlir::Value>;

template <class T>
void _join_stores_simple(Store<T> &a, const Store<T> &b, SymbolPool &pool) {
  for (auto [key, val] : a) {
    if (!b.contains(key) && !llvm::isa<Uninitialized>(val)) {
      a.write(key, pool.fresh_unknown());
    } else if (b.contains(key)) {
      a.write(key, b.at(key), WriteMode::AntiUnify);
    }
  }
}

} // namespace lleq
