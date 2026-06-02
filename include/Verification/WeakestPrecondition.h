/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Verification/TermUtils.h"
#include <cvc5/cvc5.h>

#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/Field.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>

namespace lleq {

class WeakestPreconditionAnalysis {
  llzk::component::StructDefOp structDef;
  cvc5::TermManager mgr;
  llzk::Field field;

  TermBuilder builder;

  mlir::SymbolTableCollection tables;

  cvc5::Term getExpression(mlir::Operation *op);

  void calculateWP(mlir::scf::IfOp ifOp, ConjunctionTerm &postcondition);
  void calculateWP(mlir::Operation *op, ConjunctionTerm &postcondition);
  void calculateWP(mlir::Block *block, ConjunctionTerm &postcondition);

public:
  WeakestPreconditionAnalysis(llzk::component::StructDefOp structDef,
                              llzk::Field field)
      : structDef{structDef}, field{field}, builder{mgr, field} {}

  cvc5::Term generateVerificationConditions();
};

cvc5::Term getPostcondition(llzk::component::StructDefOp, cvc5::TermManager &);
} // namespace lleq
