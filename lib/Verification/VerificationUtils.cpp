/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/VerificationUtils.h"

#include <llvm/Support/LogicalResult.h>
#include <llzk/Analysis/LightweightSignalEquivalenceAnalysis.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Transforms/LLZKComputeConstrainToProductPass.h>

using namespace mlir;
using namespace llzk;

namespace lleq {
llvm::LogicalResult ensureProductFunc(ModuleOp module,
                                      component::StructDefOp structDef) {
  if (structDef.getProductFuncOp()) {
    return success();
  }

  auto computeFunc = structDef.getComputeFuncOp();
  auto constrainFunc = structDef.getConstrainFuncOp();
  if (!computeFunc || !constrainFunc) {
    return structDef.emitError()
           << "expected the selected struct to define either @product or both "
              "@compute and @constrain";
  }

  SymbolTableCollection tables;
  LightweightSignalEquivalenceAnalysis equivalence(module);
  ProductAligner aligner(tables, equivalence);
  auto productFunc = aligner.alignFuncs(structDef, computeFunc, constrainFunc);
  if (!productFunc) {
    return structDef.emitError()
           << "failed to align @compute/@constrain into @product";
  }

  return aligner.alignCalls(productFunc);
}

} // namespace lleq
