#include "Analysis/SymbolicStore.h"

#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace lleq;

Symbol SymbolicStore::lookup(mlir::Value value) {
  // Input signal
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    // TODO: what do we do here??
    llvm::report_fatal_error("ruh-roh");
  }

  if (mlir::isa<mlir::TypedValue<llzk::array::ArrayType>>(value)) {
    llvm::report_fatal_error("cannot generate a symbol for a non-scalar value");
  }

  mlir::Operation *definingOp = value.getDefiningOp();
  return llvm::TypeSwitch<mlir::Operation *, Symbol>(definingOp)
      // A template parameter
      .Case<llzk::polymorphic::ConstReadOp>(
          [&](llzk::polymorphic::ConstReadOp templRead) {
            return pool.templ_param(templRead.getConstName());
          })
      // Read from a value array into a value scalar
      .Case<llzk::array::ReadArrayOp>([&](llzk::array::ReadArrayOp arrRead) {
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
      .Case<llzk::felt::FeltBinaryOpInterface>(
          [&](llzk::felt::FeltBinaryOpInterface binop) {
            char op = mlir::isa<llzk::felt::AddFeltOp>(binop)   ? '+'
                      : mlir::isa<llzk::felt::SubFeltOp>(binop) ? '-'
                      : mlir::isa<llzk::felt::MulFeltOp>(binop) ? '*'
                                                                : '?';
            if (op == '?')
              return pool.fresh_unknown();
            return pool.arith(lookup(binop.getLhs()), lookup(binop.getRhs()),
                              op);
          })
      .Case<llzk::felt::FeltConstantOp>([&](llzk::felt::FeltConstantOp op) {
        // TODO: this won't work for "arith.constant"
        return pool.constant(op.getValue());
      })
      .Case<llzk::component::FieldReadOp>(
          [&](llzk::component::FieldReadOp read) {
            // Reading from a scalar field should copy from the symbol store
            SignalRef ref{.value = read.getFieldName(), .indices = {}};
            if (!signalStore.contains(ref)) {
              signalStore[ref] = pool.fresh_unknown();
            }
            return signalStore[ref];
          })
      .Default([&](auto) { return pool.fresh_unknown(); });
}
