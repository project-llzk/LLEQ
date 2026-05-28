/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cvc5/cvc5.h>

#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/Value.h>

namespace lleq {

class WeakestPreconditionAnalysis {
  llzk::component::StructDefOp structDef;
  cvc5::TermManager mgr;

  cvc5::Term getExpression(mlir::Operation *op);

  cvc5::Term calculateWP(mlir::Operation *op, cvc5::Term postcondition);
  cvc5::Term calculateWP(mlir::Block *block, cvc5::Term postcondition);

  cvc5::Term getConstant(mlir::Value value);
  cvc5::Term getConstant(mlir::StringRef memberName, bool isWitness);

  // Making these function-local and static is problematic because it means the
  // contained cvc5::Term's get destroyed *after* the owned TermManager
  llvm::DenseMap<mlir::Value, cvc5::Term> constants;
  llvm::StringMap<cvc5::Term> witnessMembers, constraintMembers;

public:
  WeakestPreconditionAnalysis(llzk::component::StructDefOp structDef)
      : structDef{structDef} {}

  cvc5::Term generateVerificationConditions();
};

cvc5::Term getPostcondition(llzk::component::StructDefOp, cvc5::TermManager &);
} // namespace lleq
