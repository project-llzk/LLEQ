#pragma once

#include "Analysis/SymbolExpr.h"

#include <concepts>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/PointerUnion.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace lleq {

template <std::equality_comparable NameT> struct Ref {
  NameT name;
  std::vector<Symbol> indices;
};

template <std::equality_comparable T>
static inline bool operator==(const Ref<T> &a, const Ref<T> &b) {
  return a.name == b.name && a.indices == b.indices;
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
    return llvm::hash_combine(
        llvm::hash_value(Val.name),
        llvm::hash_combine_range(Val.indices.begin(), Val.indices.end()));
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

class SymbolicStore {
  SymbolPool pool;
  mlir::DenseMap<SignalRef, Symbol> signalStore;
  mlir::DenseMap<ValueRef, Symbol> valueStore;

public:
  void build_store(llzk::component::StructDefOp structDef);
  // Update the signalStore and valueStore based on a single operation
  void process_operation(mlir::Operation *op);

  // Update the signalStore and valueStore based on the operations in a block
  void process_block(mlir::Block *block);

  Symbol lookup(mlir::Value value);
};
} // namespace lleq
