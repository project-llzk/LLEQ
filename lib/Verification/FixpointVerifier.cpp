/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/FixpointVerifier.h"
#include "Verification/DeductiveVerifier.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <ranges>

namespace lleq {
using namespace llzk;
using namespace mlir;

ChangeResult FixpointVerifier::runIteration() {
  ensure(succeeded(deductiveVerifier.generateBaseQuery()),
         "failed to generate SMT struct semantics");

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
}

} // namespace lleq
