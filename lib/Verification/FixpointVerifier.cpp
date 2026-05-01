/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/FixpointVerifier.h"
#include "Verification/DeductiveVerifier.h"
#include <llvm/ADT/iterator_range.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>

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

} // namespace lleq
