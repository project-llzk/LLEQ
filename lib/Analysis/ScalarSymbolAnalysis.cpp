/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/ScalarSymbolAnalysis.h"

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SlowDynamicAPInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/DynamicAPIntHelper.h>
#include <llzk/Util/ErrorHelper.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>

#define DEBUG_TYPE "scalar-symbol-analysis"

namespace lleq {

void ScalarSymbolAnalysis::visitExternalCall(
    mlir::CallOpInterface _call, llvm::ArrayRef<const Lattice *> arguments,
    llvm::ArrayRef<Lattice *> results) {
  auto call = dyn_cast<llzk::function::CallOp>(_call.getOperation());
  if (!call || !(call.calleeIsCompute() || call.calleeIsConstrain())) {
    return Base::visitExternalCall(_call, arguments, results);
  }

  const auto &[subcmp, args] =
      call.calleeIsCompute()
          ? std::tuple{call.getResult(0), arguments}
          : std::tuple{call.getArgOperands().front(), arguments.drop_front()};

  std::string name;
  llvm::raw_string_ostream ss{name};
  ss << subcmp.getType();

  Symbol x = args.front()->getValue();

  Symbol subcmpSym = pool.get().func_call(
      name, llvm::map_to_vector(args, [this](const Lattice *lat) -> Symbol {
        return lat->getValue();
      }));

  auto lat = getOrCreate<ScalarLattice>(subcmp);
  propagateIfChanged(lat, lat->join(subcmpSym));
}

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
                auto value = op.getValue().getValue();
                return {pool.get().constant(llzk::toDynamicAPInt(value))};
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
          .Case<llzk::component::MemberReadOp>(
              [this, operands](llzk::component::MemberReadOp readOp)
                  -> llvm::SmallVector<Symbol> {
                // If this is a subcomponent read, it won't be populated later
                // so just fill it with a symbol representing the subcomponent
                // member
                if (isSubcmpRead(readOp)) {
                  return {pool.get().func_call(
                      ("read-@" + readOp.getMemberName()).str(),
                      {operands[0]->getValue()})};
                }

                // We don't have any store information yet, so just set it to
                // uninitialized. When this op is picked up by
                // SymbolicStoreAnalysis it will read from the store and
                // propagate the correct lattice value
                return {pool.get().uninitialized()};
              })
          .Case<llzk::array::ReadArrayOp>(
              [this, operands](llzk::array::ReadArrayOp readArr)
                  -> llvm::SmallVector<Symbol> {
                if (llvm::isa<mlir::BlockArgument>(readArr.getArrRef())) {
                  // If the array we're reading from is an input, we don't need
                  // a store to construct the right symbol
                  llvm::SmallVector<Symbol> indices;
                  for (auto *arg : llvm::drop_begin(operands)) {
                    indices.push_back(arg->getValue());
                  }
                  return {pool.get().index(readArr.getArrRef(), indices)};
                }
                // Otherwise, do the same thing as MemberReadOp
                return {pool.get().uninitialized()};
              })
          .Case<mlir::scf::YieldOp>([this, operands](mlir::scf::YieldOp yield)
                                        -> llvm::SmallVector<Symbol> {
            // Propagate yielded values to the parent's results
            mlir::Operation *parent = yield->getParentOp();
            for (auto [yielded, result] :
                 llvm::zip(operands, parent->getResults())) {
              auto *resultLattice = getLatticeElement(result);
              propagateIfChanged(resultLattice,
                                 resultLattice->join(yielded->getValue()));
            }
            // If necessary, propagate yielded values to the next iteration's
            // iter_args
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
            // Bit weird to get called on an op with no results, but that's fine
            if (results.empty()) {
              return {};
            }
            // Conservatively just build a symbol out of the op name and
            // operands
            llvm::SmallVector<Symbol> args;
            for (auto *arg : operands) {
              args.push_back(arg->getValue());
            }
            return {pool.get().func_call(op->getName().getStringRef(), args)};
          });
  if (results.empty()) {
    return mlir::success();
  }
  llzk::ensure(results.size() == symbols.size(),
               "unsupported: expression with multiple results");
  for (auto [result, sym] : llvm::zip(results, symbols)) {
    LLVM_DEBUG(llvm::dbgs() << "Symbol: " << sym << '\n');
    propagateIfChanged(result, result->join(sym));
  }
  return mlir::success();
}

} // namespace lleq
