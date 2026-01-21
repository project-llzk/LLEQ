/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/ScalarSymbolAnalysis.h"

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <llzk/Util/ErrorHelper.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>

#define DEBUG_TYPE "scalar-symbol-analysis"

namespace lleq {

mlir::LogicalResult
ScalarSymbolAnalysis::visitOperation(mlir::Operation *op,
                                     llvm::ArrayRef<const Lattice *> operands,
                                     llvm::ArrayRef<Lattice *> results) {
  LLVM_DEBUG({
    llvm::dbgs() << "Operation: " << *op << "\n";
    for (auto operand : operands) {
      llvm::dbgs() << "* operand: " << operand->getValue() << "\n";
    }
  });
  if (operands.empty() && results.empty()) {
    return mlir::success();
  }

  auto symbols =
      llvm::TypeSwitch<mlir::Operation *, llvm::SmallVector<Symbol>>(op)
          .Case<mlir::arith::ConstantOp>([this](mlir::arith::ConstantOp op)
                                             -> llvm::SmallVector<Symbol> {
            return {pool.get().constant(mlir::DynamicAPInt{
                llvm::dyn_cast<mlir::IntegerAttr>(op.getValue()).getInt()})};
          })
          .Case<llzk::polymorphic::ConstReadOp>(
              [this](llzk::polymorphic::ConstReadOp op)
                  -> llvm::SmallVector<Symbol> {
                return {pool.get().templ_param(op.getConstName())};
              })
          .Case<llzk::felt::FeltConstantOp>(
              [this](
                  llzk::felt::FeltConstantOp op) -> llvm::SmallVector<Symbol> {
                return {pool.get().constant(
                    mlir::DynamicAPInt{op.getValue().getSExtValue()})};
              })
          .Case<llzk::felt::FeltBinaryOpInterface>(
              [this, operands](llzk::felt::FeltBinaryOpInterface binop)
                  -> llvm::SmallVector<Symbol> {
                llvm::SmallVector<Symbol> args{operands[0]->getValue(),
                                               operands[1]->getValue()};
                return {pool.get().func_call(binop->getName().getStringRef(),
                                             args)};
              })
          .Case<llzk::function::CallOp>(
              [this, operands](
                  llzk::function::CallOp call) -> llvm::SmallVector<Symbol> {
                llvm::SmallVector<Symbol> args;
                for (auto *arg : operands) {
                  args.push_back(arg->getValue());
                }
                return {pool.get().func_call(
                    call.getCallee().getLeafReference(), args)};
              })
          .Case<llzk::component::FieldReadOp>(
              [this](auto) -> llvm::SmallVector<Symbol> {
                return {pool.get().uninitialized()};
              })
          .Case<llzk::array::ReadArrayOp>(
              [this, operands](llzk::array::ReadArrayOp readArr)
                  -> llvm::SmallVector<Symbol> {
                if (llvm::isa<mlir::BlockArgument>(readArr.getArrRef())) {
                  llvm::SmallVector<Symbol> indices;
                  for (auto *arg : llvm::drop_begin(operands)) {
                    indices.push_back(arg->getValue());
                  }
                  return {pool.get().index(readArr.getArrRef(), indices)};
                }
                return {pool.get().uninitialized()};
              })
          .Case<mlir::scf::YieldOp>([this, operands](mlir::scf::YieldOp yield)
                                        -> llvm::SmallVector<Symbol> {
            mlir::Operation *parent = yield->getParentOp();
            for (auto [yielded, result] :
                 llvm::zip(operands, parent->getResults())) {
              auto *resultLattice = getLatticeElement(result);
              propagateIfChanged(resultLattice,
                                 resultLattice->join(yielded->getValue()));
            }
            if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(parent)) {
              for (auto [yielded, iterArg] :
                   llvm::zip(operands, forOp.getRegionIterArgs())) {
                auto *argLattice = getLatticeElement(iterArg);
                propagateIfChanged(argLattice,
                                   argLattice->join(yielded->getValue()));
              }
            }
            return {};
          })
          .Default([this, operands,
                    results](mlir::Operation *op) -> llvm::SmallVector<Symbol> {
            if (results.empty()) {
              return {};
            }
            llvm::SmallVector<Symbol> args;
            for (auto *arg : operands) {
              args.push_back(arg->getValue());
            }
            return {pool.get().func_call(op->getName().getStringRef(), args)};
          });
  llzk::ensure(results.size() == symbols.size(),
               "unsupported: expression with multiple results");
  for (auto [result, sym] : llvm::zip(results, symbols)) {
    LLVM_DEBUG(llvm::dbgs() << "Symbol: " << sym << "\n");
    propagateIfChanged(result, result->join(sym));
  }
  return mlir::success();
}

} // namespace lleq
