/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolicStore.h"
#include "Analysis/SymbolExpr.h"

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LLVM.h>

#define DEBUG_TYPE "symbolic-store"

using namespace lleq;

SymbolicStore::SymbolicStore(const SymbolicStore &other)
    : pool{std::make_unique<SymbolPool>()},
      signalStore{other.signalStore.clone(*pool.get())},
      valueStore{other.valueStore.clone(*pool.get())} {}

SymbolicStore &SymbolicStore::operator=(const SymbolicStore &other) {
  auto new_pool = std::make_unique<SymbolPool>();
  signalStore = other.signalStore.clone(*new_pool.get());
  valueStore = other.valueStore.clone(*new_pool.get());
  pool.reset(new_pool.release());
  return *this;
}

void SymbolicStore::dump(llvm::raw_ostream &os) const {
  for (const auto &[sig, sym] : signalStore) {
    os << sig.name;
    for (auto idx : sig.indices) {
      os << "[" << idx << "]";
    }
    os << ": " << sym << "\n";
  }
}

void SymbolicStore::build_store(llzk::component::StructDefOp structDef) {
  component = structDef;
  auto computeFunc = structDef.getComputeFuncOp();
  for (auto &block : computeFunc.getFunctionBody().getBlocks()) {
    process_block(&block);
  }
}

Symbol SymbolicStore::lookup(mlir::Value value) {
  if (llvm::isa<mlir::TypedValue<llzk::array::ArrayType>>(value)) {
    llvm::report_fatal_error("cannot generate a symbol for a non-scalar value");
  }
  if (valueStore.contains({.name = value, .indices = {}})) {
    return valueStore.at({.name = value, .indices = {}});
  }

  // Input signal
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    return pool->index(value, {});
  }

  mlir::Operation *definingOp = value.getDefiningOp();
  return llvm::TypeSwitch<mlir::Operation *, Symbol>(definingOp)
      // A template parameter
      .Case<llzk::polymorphic::ConstReadOp>(
          [this](llzk::polymorphic::ConstReadOp templRead) {
            return pool->templ_param(templRead.getConstName());
          })
      // Read from a value array into a value scalar
      .Case<llzk::array::ReadArrayOp>([this](llzk::array::ReadArrayOp arrRead) {
        auto array = arrRead.getArrRef();
        // Get the indices as symbols
        llvm::SmallVector<Symbol> indices;
        for (auto index : arrRead.getIndices()) {
          indices.push_back(lookup(index));
        }
        auto ref = ValueRef{array, indices};

        // Is the array an input?
        if (llvm::dyn_cast<mlir::BlockArgument>(array)) {
          return pool->index(array, indices);
        }

        // If the index hasn't been written to before, mark it
        if (!valueStore.contains(ref)) {
          valueStore.write(ref, pool->fresh_unknown());
        }
        return valueStore.at(ref);
      })
      .Case<llzk::felt::FeltBinaryOpInterface>(
          [this](llzk::felt::FeltBinaryOpInterface binop) {
            return pool->func_call(
                binop->getName().getStringRef(),
                {lookup(binop.getLhs()), lookup(binop.getRhs())});
          })
      .Case<llzk::felt::FeltConstantOp>([this](llzk::felt::FeltConstantOp op) {
        return pool->constant(mlir::DynamicAPInt{op.getValue().getSExtValue()});
      })
      .Case<mlir::arith::ConstantOp>([this](mlir::arith::ConstantOp op) {
        return pool->constant(mlir::DynamicAPInt{
            llvm::dyn_cast<mlir::IntegerAttr>(op.getValue()).getInt()});
      })
      .Case<llzk::component::FieldReadOp>(
          [this](llzk::component::FieldReadOp read) {
            // Reading from a scalar field should copy from the symbol store
            SignalRef ref{.name = read.getFieldName(), .indices = {}};
            if (!signalStore.contains(ref)) {
              signalStore.write(ref, pool->fresh_unknown());
            }
            return signalStore.at(ref);
          })
      .Case<llzk::function::CallOp>([this](llzk::function::CallOp call) {
        llvm::SmallVector<Symbol> args;
        for (auto arg : call.getArgOperands()) {
          args.push_back(lookup(arg));
        }
        return pool->func_call(call.getCallee().getLeafReference(), args);
      })
      .Default([this](mlir::Operation *op) {
        if (op->getNumResults() != 1) {
          return pool->fresh_unknown();
        }
        // Treat it as an uninterpreted function
        llvm::SmallVector<Symbol> args;
        for (auto arg : op->getOperands()) {
          args.push_back(lookup(arg));
        }
        return pool->func_call(op->getName().getStringRef(), args);
      });
}

void SymbolicStore::process_operation(mlir::Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Processing op: " << *op << "\n");
  // TODO: handle constraint ops

  llvm::TypeSwitch<mlir::Operation *>(op)
      .Case<mlir::scf::IfOp>([this](mlir::scf::IfOp ifOp) {
        SymbolicStore thenStore{*this};
        SymbolicStore elseStore{*this};
        llvm::SmallVector<mlir::Value> results{ifOp.getResults()};
        for (auto &block : ifOp.getThenRegion()) {
          thenStore.process_block(&block, results);
        }
        for (auto &block : ifOp.getElseRegion()) {
          elseStore.process_block(&block, results);
        }

        LLVM_DEBUG({
          llvm::dbgs() << "Then store:\n";
          thenStore.dump(llvm::dbgs());
          llvm::dbgs() << "Else store:\n";
          elseStore.dump(llvm::dbgs());
        });
        *this = SymbolicStore::join(thenStore, elseStore);
      })
      .Case<mlir::scf::ForOp>([this](mlir::scf::ForOp forOp) {
        SymbolicStore bodyStore{*this};

        // Start by initializing any loop-carried deps
        bodyStore.copy_value(forOp.getInductionVar(),
                             llvm::dyn_cast<mlir::Value>(forOp.getLowerBound()),
                             WriteMode::AntiUnify);
        llvm::SmallVector<mlir::Value> loopCarriedDeps{
            llvm::drop_begin(forOp.getBody()->getArguments())};
        for (auto [arg, val] : llvm::zip(loopCarriedDeps, forOp.getInits())) {
          bodyStore.copy_value(llvm::dyn_cast<mlir::Value>(arg), val,
                               WriteMode::AntiUnify);
        }

        // Run the bodyStore on the loop body, capturing the yielded values
        // If the store changes, run it again (up to N times)
        // Otherwise, we've hit a fixpoint so break
        constexpr unsigned MAX_ITERS = 4;
        SymbolicStore oldBody;
        unsigned num_iters = 0;
        for (; num_iters < MAX_ITERS; num_iters++) {
          // Execute one iteration
          oldBody = bodyStore;
          bodyStore.process_block(forOp.getBody(), loopCarriedDeps);

          // Update the loop induction var
          Symbol newIVar = pool->func_call(
              "felt.add", {bodyStore.lookup(forOp.getInductionVar()),
                           bodyStore.lookup(forOp.getLowerBound())});
          bodyStore.valueStore.write({forOp.getInductionVar(), {}}, newIVar,
                                     WriteMode::AntiUnify);

          // If we hit a fixpoint, break early
          if (bodyStore == oldBody) {
            LLVM_DEBUG(llvm::dbgs() << "Fixpoint reached after " << num_iters
                                    << " iterations\n");
            break;
          }
        }
        LLVM_DEBUG(if (num_iters == MAX_ITERS) llvm::dbgs()
                   << "No fixpoint reached\n");

        // Finally, save the results of the loop:
        // 1. Widen signalStore to accomodate bodyStore.signalStore
        // 2. Copy the loop args into the forOp's results
        // TODO: calling `widen` twice fails (use-after-free)
        // signalStore.widen(bodyStore.signalStore);
        // valueStore.widen(bodyStore.valueStore);
        *this = SymbolicStore::join(*this, bodyStore);
        for (auto [result, arg] :
             llvm::zip(forOp.getResults(),
                       llvm::drop_begin(forOp.getBody()->getArguments()))) {
          bodyStore.copy_value<mlir::Value, mlir::Value>(valueStore, result,
                                                         arg);
        }
      })
      .Case<llzk::component::FieldReadOp>(
          [this](llzk::component::FieldReadOp read) {
            copy_value(read.getResult(), read.getFieldName());
          })
      .Case<llzk::component::FieldWriteOp>(
          [this](llzk::component::FieldWriteOp write) {
            copy_value(write.getFieldName(), write.getVal());
          })
      .Case<llzk::array::WriteArrayOp>([this](llzk::array::WriteArrayOp write) {
        // Havoc every possible array index that could be clobbered
        auto array = write.getArrRef();
        for (const auto &[valueRef, value] : valueStore) {
          // TODO: actually check if the index could be clobbered
          if (valueRef.name == array) {
            valueStore.write(valueRef, pool->fresh_unknown());
          }
        }

        // Write the new symbol
        llvm::SmallVector<Symbol> indices;
        for (auto index : write.getIndices()) {
          indices.push_back(lookup(index));
        }

        valueStore.write(ValueRef{array, indices}, lookup(write.getRvalue()));
      })
      .Default([](auto) {});
}

void SymbolicStore::process_block(mlir::Block *block,
                                  llvm::ArrayRef<mlir::Value> yielded) {
  for (auto &op : block->getOperations()) {
    if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(op)) {
      // If the block yields any values, explicitly copy them into the captures
      assert(yieldOp->getNumOperands() == yielded.size() &&
             "wrong number of values captured");
      for (auto [result, yielded] : llvm::zip(yieldOp.getOperands(), yielded)) {
        copy_value(yielded, result, WriteMode::AntiUnify);
      }
    }
    process_operation(&op);
  }
}

SymbolicStore SymbolicStore::join(const SymbolicStore &a,
                                  const SymbolicStore &b) {
  SymbolicStore result;
  result.signalStore =
      _join_stores(a.signalStore, b.signalStore, *result.pool.get());
  result.valueStore =
      _join_stores(a.valueStore, b.valueStore, *result.pool.get());
  return result;
}
