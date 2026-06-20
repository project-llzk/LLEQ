/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Verification/TermUtils.h"
#include <cvc5/cvc5.h>

#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/Field.h>
#include <llzk/Util/TypeHelper.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <unordered_map>

namespace lleq {

inline llvm::FailureOr<llzk::component::StructType>
_sp_subcomponent_type(mlir::Type type) {
  if (auto structType = mlir::dyn_cast<llzk::component::StructType>(type)) {
    return structType;
  }
  if (auto arrType = mlir::dyn_cast<llzk::array::ArrayType>(type)) {
    return _sp_subcomponent_type(arrType.getElementType());
  }
  return {};
}

class StrongestPostconditionAnalysis {
  struct SPState {
    cvc5::Term formula;
    std::unordered_map<cvc5::Term, cvc5::Term, std::hash<cvc5::Term>> bindings;
  };

  llzk::component::StructDefOp structDef;
  cvc5::TermManager mgr;
  llzk::Field field;

  TermBuilder builder;
  mlir::SymbolTableCollection tables;

  cvc5::Term getExpression(mlir::Operation *op, const SPState &state);
  cvc5::Term getCurrentTerm(cvc5::Term term, const SPState &state);
  cvc5::Term getCurrentValue(mlir::Value value, const SPState &state);
  void bindValue(mlir::Value value, cvc5::Term term, SPState &state);
  void addConstraint(cvc5::Term term, SPState &state);

  void calculateSP(mlir::scf::IfOp ifOp, SPState &state);
  void calculateSP(mlir::Operation *op, SPState &state);
  void calculateSP(mlir::Block *block, SPState &state);

  void initSubcomponents() {
    for (auto memberDef : structDef.getMemberDefs()) {
      if (auto structType = _sp_subcomponent_type(memberDef.getType());
          llvm::succeeded(structType)) {
        auto definition = structType->getDefinition(
            tables, structDef->getParentOfType<mlir::ModuleOp>());
        llzk::ensure(mlir::succeeded(definition),
                     "could not find struct definition for " +
                         memberDef.getSymName());
        builder.populateSubcomponent(definition->get());
      }
    }
  }

public:
  StrongestPostconditionAnalysis(llzk::component::StructDefOp structDef,
                                 llzk::Field field)
      : structDef{structDef}, field{field}, builder{mgr, field} {
    initSubcomponents();
  }

  void populateVerificationConditions();
  cvc5::Term generateVerificationConditions();

  void emit(llvm::raw_ostream &os);

  // Generated formula after doing strongest postcondition analysis
  cvc5::Term verificationConditions;
  // Toplevel for free variables that appear in the formula
  TermBuilder::TermSet extraDecls;
  // Mod-p bounds on the free variable declarations
  TermBuilder::TermSet declBounds;
};

} // namespace lleq
