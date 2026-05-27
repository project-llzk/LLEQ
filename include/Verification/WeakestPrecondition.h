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
  llvm::DenseMap<mlir::Value, cvc5::Term> constants;

  cvc5::Term getExpression(mlir::Operation *op);

  cvc5::Term calculateWP(mlir::Operation *op, cvc5::Term postcondition);
  cvc5::Term calculateWP(mlir::Block *block, cvc5::Term postcondition);

  cvc5::Term getConstant(mlir::Value value);
  cvc5::Term getConstant(mlir::StringRef memberName, bool isWitness);

public:
  WeakestPreconditionAnalysis(llzk::component::StructDefOp structDef)
      : structDef{structDef} {}

  cvc5::Term generateVerificationConditions();
};

cvc5::Term getPostcondition(llzk::component::StructDefOp, cvc5::TermManager &);
} // namespace lleq
