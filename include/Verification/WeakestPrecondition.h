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

  void initSubcomponents() {
    for (auto memberDef : structDef.getMemberDefs()) {

      if (auto structType = mlir::dyn_cast<llzk::component::StructType>(
              memberDef.getType())) {
        auto definition = structType.getDefinition(
            tables, structDef->getParentOfType<mlir::ModuleOp>());
        llzk::ensure(mlir::succeeded(definition),
                     "could not find struct definition for " +
                         memberDef.getSymName());
        builder.populateSubcomponent(definition->get());
      }
    }
  }

public:
  WeakestPreconditionAnalysis(llzk::component::StructDefOp structDef,
                              llzk::Field field)
      : structDef{structDef}, field{field}, builder{mgr, field} {
    initSubcomponents();
  }

  cvc5::Term getPostcondition();
  void populateVerificationConditions();
  std::pair<cvc5::Term, TermBuilder::TermSet> generateVerificationConditions();

  // Generated VCs after doing weakest precondition analysis
  cvc5::Term verificationConditions;
  // Toplevel for free variables that appear in the VCs
  TermBuilder::TermSet extraDecls;
  // Mod-p bounds on the free variable declarations
  TermBuilder::TermSet declBounds;
};

} // namespace lleq
