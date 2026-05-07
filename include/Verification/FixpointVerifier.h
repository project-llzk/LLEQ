/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "Verification/DeductiveVerifier.h"
#include "Verification/SymbolicVerifier.h"
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Analysis/DataFlowFramework.h>
namespace lleq {

/// This class manages both a deductive and symbolic verifier, and runs them in
/// a loop adding facts until reaching a fixpoint
class FixpointVerifier {
  llzk::component::StructDefOp structDef;
  llzk::Field field;

  DeductiveVerifier deductiveVerifier;
  SymbolicVerifier symbolicVerifier;

  StructVerificationResult currentResult;

public:
  FixpointVerifier(llzk::component::StructDefOp structDef, llzk::Field field)
      : structDef{structDef}, field{field}, deductiveVerifier{structDef, field},
        symbolicVerifier{structDef} {}

  mlir::ChangeResult runIteration();
  llvm::LogicalResult init(bool enableStore);
  void report(llvm::raw_ostream &os);
  void dumpSmt(llvm::raw_ostream &os);
  StructVerificationResult getResult() { return currentResult; };
};

} // namespace lleq
