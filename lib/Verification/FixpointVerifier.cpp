/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/FixpointVerifier.h"
#include "Verification/DeductiveVerifier.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Config/abi-breaking.h>
#include <llvm/Support/LogicalResult.h>
#include <llzk/Dialect/Array/Transforms/TransformationPasses.h>
#include <llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace lleq {
using namespace llzk;
using namespace mlir;

LogicalResult FixpointVerifier::init(bool enableStore) {
  if (failed(symbolicVerifier.buildStore())) {
    return failure();
  }

  if (failed(deductiveVerifier.generateBaseQuery())) {
    return failure();
  }

  for (auto memberDef : structDef.getMemberDefs()) {
    currentResult.unknownMembers.insert(memberDef.getSymName());
  }

  if (enableStore) {
    auto extraAssertions = symbolicVerifier.generateAssertions(field);
    deductiveVerifier.addExtraAssertions(extraAssertions);
  }

  return success();
}

ChangeResult FixpointVerifier::runIteration() {
  StructVerificationResult result =
      deductiveVerifier.verifyStruct(currentResult.unknownMembers);

  currentResult.update(result);
  if (result.equivalentMembers.empty() && result.inequivalentMembers.empty()) {
    return ChangeResult::NoChange;
  }
  return ChangeResult::Change;
}

void FixpointVerifier::report(raw_ostream &os) {
  auto isSignal = [this](auto memberName) -> bool {
    return structDef
        .getMemberDef(StringAttr::get(structDef.getContext(), memberName))
        .getSignal();
  };

  // Without this, `filter_to_vector` iterates over a non-const view of the
  // containers and somehow blows up
  const StructVerificationResult &result = currentResult;
  auto equivalentSignals = filter_to_vector(result.equivalentMembers, isSignal);
  auto unknownSignals = filter_to_vector(result.unknownMembers, isSignal);
  auto inequivalentSignals =
      filter_to_vector(result.inequivalentMembers,
                       [&isSignal](auto elem) { return isSignal(elem.first); });

  if (!equivalentSignals.empty()) {
    for (auto member : equivalentSignals) {
      llvm::outs() << "+ @" << structDef.getSymName() << "::" << member << "\n";
    }
  }
  if (!inequivalentSignals.empty()) {
    for (auto [member, counterexample] : inequivalentSignals) {
      auto [w, c] = counterexample;
      llvm::outs() << "- @" << structDef.getSymName() << "::" << member << "\n";
      llvm::outs() << "\twitness: " << w << "\n";
      llvm::outs() << "\tconstraint: " << c << "\n";
    }
  }
  if (!unknownSignals.empty()) {
    for (auto member : unknownSignals) {
      llvm::outs() << "* @" << structDef.getSymName() << "::" << member << "\n";
    }
  }
}

void FixpointVerifier::dumpSmt(raw_ostream &os) {
  deductiveVerifier.dumpQuery(os);
}

} // namespace lleq
