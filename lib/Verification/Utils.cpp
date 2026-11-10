/**
 * Copyright 2026 Project LLZK.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/Utils.h"

#include <llzk/Analysis/LightweightSignalEquivalenceAnalysis.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Transforms/LLZKComputeConstrainToProductPass.h>
#include <llzk/Transforms/LLZKTransformationPasses.h>
#include <mlir/Pass/PassManager.h>

using namespace mlir;
using namespace llzk;

namespace lleq::util {
function::FuncDefOp requirePureFieldFunction(function::CallOp call) {
  SymbolTableCollection tables;
  auto target = call.getCalleeTarget(tables);
  if (failed(target)) {
    throw UnsupportedConstruct(call.getLoc(), "unresolved free-function call");
  }
  auto callee = target->get();
  auto reject = [&]() {
    throw UnsupportedConstruct(call.getLoc(),
                               "free-function UF encoding requires a total "
                               "straight-line field function");
  };
  if (callee->getParentOfType<component::StructDefOp>() ||
      !llvm::hasSingleElement(callee.getFunctionBody())) {
    reject();
  }
  auto signature = callee.getFunctionType();
  if (signature.getNumResults() == 0) {
    reject();
  }
  auto fieldType = signature.getResult(0);
  if (!isa<felt::FeltType>(fieldType) ||
      llvm::any_of(signature.getInputs(),
                   [&](Type t) { return t != fieldType; }) ||
      llvm::any_of(signature.getResults(),
                   [&](Type t) { return t != fieldType; }) ||
      !llvm::equal(call.getArgOperands().getTypes(), signature.getInputs()) ||
      !llvm::equal(call.getResultTypes(), signature.getResults())) {
    reject();
  }
  auto &body = callee.getFunctionBody().front();
  auto ret = dyn_cast<function::ReturnOp>(body.getTerminator());
  if (!ret || !llvm::equal(ret.getOperandTypes(), signature.getResults())) {
    reject();
  }
  // This whitelist proves totality and purity. In particular, never summarize
  // RAM, nondeterminism, assertions, calls, loops, or division by this rule.
  for (auto &op : body.without_terminator()) {
    if (!isa<felt::FeltConstantOp, felt::AddFeltOp, felt::SubFeltOp,
             felt::MulFeltOp, felt::NegFeltOp>(op) ||
        llvm::any_of(op.getResultTypes(),
                     [&](Type t) { return t != fieldType; })) {
      reject();
    }
  }
  // Reject captures: formal input values must determine all returned values.
  for (auto &op : body) {
    for (Value operand : op.getOperands()) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getOwner() != &body) {
          reject();
        }
      } else if (operand.getDefiningOp()->getBlock() != &body) {
        reject();
      }
    }
  }
  return callee;
}

void ensureProductFunc(ModuleOp module, component::StructDefOp structDef) {
  if (structDef.getProductFuncOp()) {
    return;
  }

  auto computeFunc = structDef.getComputeFuncOp();
  auto constrainFunc = structDef.getConstrainFuncOp();
  if (!computeFunc || !constrainFunc) {
    structDef.emitError()
        << "expected the selected struct to define either @product or both "
           "@compute and @constrain";
    return;
  }

  SymbolTableCollection tables;
  LightweightSignalEquivalenceAnalysis equivalence(module);
  ProductAligner aligner(tables, equivalence);
  auto productFunc = aligner.alignFuncs(structDef, computeFunc, constrainFunc);
  if (!productFunc) {
    structDef.emitError()
        << "failed to align @compute/@constrain into @product";
    return;
  }

  ensure(succeeded(aligner.alignCalls(productFunc)),
         "failed to align subcomponent calls");

  // Now, try fusing loops
  PassManager pm{module->getContext()};
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ensure(succeeded(pm.run(module)), "failed to align loops");
}
} // namespace lleq::util
