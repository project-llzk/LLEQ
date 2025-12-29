#include "Analysis/SymbolExpr.h"
#include "Analysis/SymbolicStore.h"
#include "Analysis/Unification.h"
#include <llzk/Dialect/Array/IR/Types.h>

namespace lleq {
// Group written indices by array reference
template <class T> auto _group_idx(const Store<T> &store) {
  llvm::DenseMap<T, llvm::SmallVector<llvm::SmallVector<Symbol>>> grouped;

  for (auto [k, _] : store) {
    grouped[k.name].push_back(k.indices);
  }

  return grouped;
}

// TODO: Rewrite all the joins to be in-place to avoid allocating
template <class T>
Store<T> _join_stores(const Store<T> &a, const Store<T> &b, SymbolPool &pool) {
  // For every pair of entries (arr, phi_1) :- x; (arr, phi_2) :- y from `a` and
  // `b`, add an entry (arr, AU(phi_1, phi_2)) :- AU(x, y) to the result, where
  // AU represents antiunification
  Store<T> result{pool};
  auto groupedA = _group_idx(a);
  auto groupedB = _group_idx(b);

  for (auto [arrA, idxAs] : groupedA) {
    if (!groupedB.contains(arrA)) {
      // If `arrA` is only written to in `a`, then just copy all the entries
      for (auto idxA : idxAs) {
        Ref<T> ref{.name = arrA, .indices = idxA};
        result.write(ref, a.at(ref));
      }
      continue;
    }

    // If `arrA` is written to in both `a` and `b`, pairwise anti-unify the
    // written indices
    auto idxBs = groupedB[arrA];
    for (auto idxA : idxAs) {
      for (auto idxB : idxBs) {
        llvm::SmallVector<Symbol> idxUnif;
        for (auto [iA, iB] : llvm::zip(idxA, idxB)) {
          idxUnif.push_back(anti_unify(iA, iB));
        }
        Ref<T> ref{arrA, idxUnif};
        // and anti-unify all values written to these anti-unified indices
        result.write(ref, a.at({arrA, idxA}), WriteMode::AntiUnify);
        result.write(ref, b.at({arrA, idxB}), WriteMode::AntiUnify);
      }
    }
  }

  // Finally, just copy over all entries only written to in `b`
  for (auto [arrB, idxBs] : groupedB) {
    if (!groupedA.contains(arrB)) {
      for (auto idxB : idxBs) {
        Ref<T> ref{arrB, idxB};
        result.write(ref, b.at(ref));
      }
    }
  }

  return result;
}

template ValueStore _join_stores(const ValueStore &a, const ValueStore &b,
                                 SymbolPool &);
template SignalStore _join_stores(const SignalStore &a, const SignalStore &b,
                                  SymbolPool &);

template <> mlir::Type SymbolicStore::_lookup_type(mlir::Value val) {
  return val.getType();
}
template <> mlir::Type SymbolicStore::_lookup_type(llvm::StringRef val) {
  return component
      .getFieldDef(mlir::StringAttr::get(component->getContext(), val))
      .getType();
}

template <> ValueStore &SymbolicStore::_get() { return valueStore; }
template <> SignalStore &SymbolicStore::_get() { return signalStore; }

template <class S, class T>
void SymbolicStore::copy_value(Store<S> &destStore, S dest, T src,
                               WriteMode mode) {
  if (llvm::isa<llzk::array::ArrayType>(_lookup_type(src))) {
    // Copy all written indices
    for (auto [ref, val] : _get<T>()) {
      if (ref.name == src) {
        destStore.write({.name = dest, .indices = ref.indices}, val, mode);
      }
    }
  } else {
    destStore.write({.name = dest, .indices = {}}, lookup(src), mode);
  }
}

template void SymbolicStore::copy_value(ValueStore &, mlir::Value, mlir::Value,
                                        WriteMode);
template void SymbolicStore::copy_value(ValueStore &, mlir::Value,
                                        llvm::StringRef, WriteMode);
template void SymbolicStore::copy_value(SignalStore &, llvm::StringRef,
                                        mlir::Value, WriteMode);
template void SymbolicStore::copy_value(SignalStore &, llvm::StringRef,
                                        llvm::StringRef, WriteMode);

} // namespace lleq
