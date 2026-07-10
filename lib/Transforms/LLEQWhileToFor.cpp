/**
 * Copyright 2026 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Transforms/LLEQWhileToFor.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Bool/IR/Enums.h>
#include <llzk/Dialect/Bool/IR/Ops.h>
#include <llzk/Dialect/Cast/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>

#define DEBUG_TYPE "lleq-while-to-for"
namespace lleq::transform {

llvm::LogicalResult transformWhileToFor(llzk::function::FuncDefOp funcDef) {
  auto result = funcDef.getBody().walk([](mlir::scf::WhileOp op) {
    mlir::IRRewriter rewriter{op->getContext()};
    if (auto forOp = transformWhileToFor(op, rewriter);
        llvm::succeeded(forOp)) {
      return mlir::WalkResult::advance();
    }
    return mlir::WalkResult::interrupt();
  });
  return llvm::failure(result.wasInterrupted());
}

ForOpInfo parseInfo(mlir::scf::WhileOp op) {
  ForOpInfo info;

  auto condition = op.getConditionOp().getCondition();
  if (auto cmp = condition.getDefiningOp<llzk::boolean::CmpOp>();
      cmp.getPredicate() == llzk::boolean::FeltCmpPredicate::LT) {
    // We found the ivar and the ub
    info.ivar = cmp.getLhs();
    info.ub = cmp.getRhs();
  } else {
    return info;
  }

  // Find which # block arg the ivar is
  for (auto [i, arg] : llvm::enumerate(op.getConditionOp().getArgs())) {
    if (arg == *info.ivar) {
      info.ivar_index = i;
      break;
    }
  }
  if (info.ivar_index == -1) {
    return info;
  }

  // Now, look for the lb as the corresponding init arg
  info.lb = *op.getInits().drop_front(info.ivar_index).begin();

  // Finally, look for the step
  auto nextIvar = *op.getYieldedValues().drop_front(info.ivar_index).begin();
  if (auto incOp = nextIvar.getDefiningOp<llzk::felt::AddFeltOp>()) {
    if (incOp.getLhs().getDefiningOp<llzk::felt::FeltConstantOp>()) {
      info.step = incOp.getLhs();
    } else if (incOp.getRhs().getDefiningOp<llzk::felt::FeltConstantOp>()) {
      info.step = incOp.getRhs();
    }
  }

  return info;
}

llvm::raw_ostream &ForOpInfo::print(llvm::raw_ostream &os) const {
  if (!success()) {
    return os << "[Failed to parse info]";
  }
  return os << "For loop from " << lb << " to " << ub << " with a step of "
            << step;
}

llvm::FailureOr<mlir::scf::ForOp>
transformWhileToFor(mlir::scf::WhileOp op, mlir::RewriterBase &rewriter) {
  ForOpInfo info = parseInfo(op);
  if (!info.success()) {
    LLVM_DEBUG(llvm::dbgs() << "[failed to parse info]\n");
    return llvm::failure();
  }

  rewriter.setInsertionPointAfter(op);
  mlir::IRMapping mapping;

  auto copyValue = [op, &rewriter, &mapping](mlir::Value val) -> mlir::Value {
    if (auto definingOp = val.getDefiningOp();
        definingOp && definingOp->getParentOfType<mlir::scf::WhileOp>() == op) {
      return rewriter.clone(*definingOp, mapping)->getResult(0);
    }
    return val;
  };

  // Emit a prelude setting up the loop bounds
  auto lbOpFelt = copyValue(*info.lb);
  auto ubOpFelt = copyValue(*info.ub);
  auto stepOpFelt = copyValue(*info.step);

  llvm::SmallVector<mlir::Value> inits;
  for (auto [i, init] : llvm::enumerate(op.getInits())) {
    if (i == info.ivar_index) {
      continue;
    }
    inits.push_back(init);
  }

  auto forOp = rewriter.create<mlir::scf::ForOp>(op->getLoc(), lbOpFelt,
                                                 ubOpFelt, stepOpFelt, inits);

  mapping.map(*info.ivar, forOp.getInductionVar());
  rewriter.setInsertionPointToStart(forOp.getBody());

  auto *whileBody = op.getAfterBody();
  for (size_t i = 0; i < whileBody->getNumArguments(); i++) {
    if (i == info.ivar_index) {
      mapping.map(whileBody->getArgument(i), forOp.getInductionVar());
      continue;
    }
    mapping.map(whileBody->getArgument(i),
                forOp.getRegionIterArg(i > info.ivar_index ? i - 1 : i));
  }

  for (auto &bodyOp : *op.getAfterBody()) {
    if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
      llvm::SmallVector<mlir::Value> valuesToYield;
      for (auto [i, val] : llvm::enumerate(yieldOp.getResults())) {
        if (i == info.ivar_index) {
          continue;
        }
        valuesToYield.push_back(mapping.lookupOrDefault(val));
      }
      rewriter.create<mlir::scf::YieldOp>(yieldOp.getLoc(), valuesToYield);
      continue;
    }
    rewriter.clone(bodyOp, mapping);
  }

  llvm::SmallVector<mlir::Value> replacedValues;
  for (auto [i, result] : llvm::enumerate(op.getResults())) {
    if (i == info.ivar_index) {
      replacedValues.push_back(*info.ub);
      continue;
    }
    replacedValues.push_back(forOp.getResult(i > info.ivar_index ? i - 1 : i));
  }

  rewriter.replaceOp(op, replacedValues);

  return forOp;
}
} // namespace lleq::transform
