#include "Analysis/SymbolicStore.h"

#include <llzk/Dialect/Array/IR/Ops.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace lleq;

void SymbolicStore::process_block(mlir::Block *block) {}

SignalRef SymbolicStore::getSignal(mlir::Value value) {
  if (mlir::dyn_cast<mlir::BlockArgument>(value)) {
    // An input is a signal by itself
    return {value, {}};
  }
}

// Symbol SymbolicStore::lookup(mlir::Value value) {
//   if (!store.contains(value)) {
//     store[value] = _build_sym(value);
//   }
//   return store[value];
// }

// Symbol SymbolicStore::_build_sym(mlir::Value value) {
//   if (mlir::dyn_cast<mlir::BlockArgument>(value)) {
//     return pool.index(value, {});
//   }
//   mlir::Operation *defining = value.getDefiningOp();
//   if (auto read = mlir::dyn_cast<llzk::array::ReadArrayOp>(defining)) {
//     auto indices = read.getIndices();

//     Symbol arraySym = lookup(read.getArrRef());
//   }
// }
