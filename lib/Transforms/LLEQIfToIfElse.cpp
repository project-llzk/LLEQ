/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Transforms/LLEQIfToIfElse.h"
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>

namespace lleq::transform {

using namespace llvm;
using namespace mlir;
using namespace llzk;

FailureOr<scf::IfOp> addElseToIf(scf::IfOp op, RewriterBase &rewriter) {
  if (op.getElseRegion().empty()) {
    auto elseBlock = rewriter.createBlock(&op.getElseRegion());
    rewriter.setInsertionPointToStart(elseBlock);
    rewriter.create<scf::YieldOp>(op->getLoc());
  }
  return op;
}

LogicalResult transformIfToIfElse(function::FuncDefOp funcDef) {
  auto result = funcDef.walk([](scf::IfOp op) {
    IRRewriter rewriter{op.getContext()};
    if (failed(addElseToIf(op, rewriter))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

} // namespace lleq::transform
