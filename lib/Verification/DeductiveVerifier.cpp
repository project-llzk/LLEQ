/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/DeductiveVerifier.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/SMTAPI.h>
#include <llzk/Dialect/SMT/IR/SMTOps.h>

using namespace mlir;
using namespace llzk::smt;

namespace lleq {

template <class SMTOp>
llvm::SMTExprRef (llvm::SMTSolver::*mkBVOp)(const llvm::SMTExprRef &,
                                            const llvm::SMTExprRef &);

template <> auto mkBVOp<llzk::smt::IntAddOp> = &llvm::SMTSolver::mkBVAdd;
template <> auto mkBVOp<llzk::smt::IntSubOp> = &llvm::SMTSolver::mkBVSub;
template <> auto mkBVOp<llzk::smt::IntMulOp> = &llvm::SMTSolver::mkBVMul;
template <> auto mkBVOp<llzk::smt::IntModOp> = &llvm::SMTSolver::mkBVURem;

LogicalResult DeductiveVerifier::_process_op(mlir::Operation *op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<llzk::smt::IntAddOp>([this](llzk::smt::IntAddOp addOp) {
        SmallVector<Value> args = addOp.getInputs();
        smtExprs[addOp.getResult()] =
            solver->mkBVAdd(smtExprs[args[0]], smtExprs[args[1]]);
      })
      .Case<llzk::smt::IntSubOp>([this](llzk::smt::IntSubOp subOp) {

      });
  return success();
}

} // namespace lleq
