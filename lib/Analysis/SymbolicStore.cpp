#include "Analysis/SymbolicStore.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace lleq;

Symbol SymbolicStore::lookup(mlir::Value value) {
  // Input signal
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    // TODO: what do we do here??
    llvm::report_fatal_error("fuck");
  }

  mlir::Operation *definingOp = value.getDefiningOp();
  return llvm::TypeSwitch<mlir::Operation *, Symbol>(definingOp)
      // A template parameter
      .Case<llzk::polymorphic::ConstReadOp>([&](auto templRead) {
        return pool.templ_param(templRead.getConstName());
      })
      // Read from a value array into a value scalar
      .Case<llzk::array::ReadArrayOp>([&](auto arrRead) {
        auto array = arrRead.getArrRef();
        // Get the indices as symbols
        std::vector<Symbol> indices;
        for (auto index : arrRead.getIndices()) {
          indices.push_back(lookup(index));
        }
        auto ref = ValueRef{array, indices};

        // Is the array an input?
        if (mlir::dyn_cast<mlir::BlockArgument>(array)) {
          return pool.index(array, indices);
        }

        // If the index hasn't been written to before, mark it
        if (!valueStore.contains(ref)) {
          valueStore[ref] = pool.fresh_unknown();
        }
        return valueStore[ref];
      })
      .Default([&](auto _) { return pool.fresh_unknown(); });
}
