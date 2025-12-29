
#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"
#include <algorithm>
#include <concepts>

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
template <class T> struct Store;

// For every pair of entries (arr, phi_1) :- x; (arr, phi_2) :- y from `a` and
// `b`, add an entry (arr, AU(phi_1, phi_2)) :- AU(x, y) to the result, where
// AU represents antiunification
template <class T>
Store<T> _join_stores(const Store<T> &a, const Store<T> &b, SymbolPool &);

enum class WriteMode { Overwrite, AntiUnify };

/// Represents a single mapping of refs (names + indices) to symbols.
/// Lightweight wrapper around a DenseMap<Ref<T>, Symbol> that provides some
/// convenience methods
template <class T> struct Store {
  // Support iterating over the underlying store
  auto begin() const { return _store.begin(); }
  auto end() const { return _store.end(); }
  decltype(auto) at(Ref<T> ref) const { return _store.at(ref); }
  bool contains(Ref<T> ref) const { return _store.contains(ref); }
  auto size() const { return _store.size(); }

  // Configure behavior when writing to a name that is already present
  // (overwrite vs. anti-unify)
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
  // Creates a shallow copy. For a deep copy, use `.clone()`
  Store(const Store<T> &other) : _pool{other._pool}, _store{other._store} {}
  Store &operator=(const Store<T> &other) {
    assert(&_pool == &other._pool &&
           "cannot assign to store backed by different SymbolPool");
    _store = other._store;
    return *this;
  }

  // Create a deep copy backed by `pool`
  Store<T> clone(SymbolPool &pool) const {
    Store<T> cloned{pool};
    for (const auto [ref, val] : _store) {
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

private:
  llvm::DenseMap<Ref<T>, Symbol> _store;
  SymbolPool &_pool;
};

using SignalStore = Store<llvm::StringRef>;
using ValueStore = Store<mlir::Value>;

} // namespace lleq
