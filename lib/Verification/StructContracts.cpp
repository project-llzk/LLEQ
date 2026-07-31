/**
 * Copyright 2026 Project LLZK.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/StructContracts.h"
#include "Verification/Utils.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llzk/Dialect/Bool/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Dialect/Verif/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

using namespace llzk;
using namespace llzk::component;
using namespace mlir;

namespace lleq {

static inline FailureOr<CreateStructOp>
getAllocation(function::FuncDefOp func) {
  StructType parentType = func->getParentOfType<StructDefOp>().getType();
  auto funcReturn = dyn_cast<function::ReturnOp>(
      func.getFunctionBody().front().getTerminator());

  auto alloc = funcReturn.getOperand(0).getDefiningOp<CreateStructOp>();
  if (!alloc || alloc.getType() != parentType) {
    return failure();
  }
  return alloc;
}

void applyContractToStruct(verif::ContractOp contract) {
  // TODO: bail out if contract is inside template
  if (!contract.hasStructTarget()) {
    return;
  }

  // Find the struct the contract is targeting
  auto target = contract.getStructTarget();
  if (failed(target)) {
    return;
  }

  // Make sure we have a @product function
  auto structDef = target->get();
  util::ensureProductFunc(structDef->getParentOfType<ModuleOp>(), structDef);

  // Lower preconditions to bool.assert
  auto productFunc = structDef.getProductFuncOp();

  auto allocation = getAllocation(productFunc);
  ensure(succeeded(allocation), "failed to find `struct.new` to condition");

  auto &contractBody = contract.getFunctionBody();

  // Map contract args to struct inputs
  IRMapping mapping;
  mapping.map(contractBody.getArgument(0), allocation->getResult());
  for (auto [contractArg, functionArg] :
       llvm::zip(llvm::drop_begin(contractBody.getArguments()),
                 productFunc.getArguments())) {
    mapping.map(contractArg, functionArg);
  }

  IRRewriter rewriter{structDef->getContext()};
  // Don't wanna just call cloneRegion or something, since
  // `verif.require_compute` and `verif.require_constrain` have to also be
  // updated to `bool.assert`
  rewriter.setInsertionPointAfter(*allocation);
  for (auto &contractOp :
       contractBody.getBlocks().front().without_terminator()) {
    llvm::TypeSwitch<Operation &, void>(contractOp)
        .Case<verif::RequireComputeOp, verif::RequireConstrainOp>(
            [&rewriter, &mapping](auto op) {
              rewriter.create<boolean::AssertOp>(
                  op.getLoc(), mapping.lookupOrDefault(op.getCondition()),
                  StringAttr::get(op.getContext()));
            })
        .Default(
            [&rewriter, &mapping](auto &op) { rewriter.clone(op, mapping); });
  }
}

} // namespace lleq
