/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Verification/TermUtils.h"
#include <cvc5/cvc5.h>

#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/AttributeHelper.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/ErrorHelper.h>
#include <llzk/Util/Field.h>
#include <llzk/Util/TypeHelper.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>

namespace lleq {

inline llvm::FailureOr<llzk::component::StructType>
_subcomponent_type(mlir::Type type) {
  if (auto structType = mlir::dyn_cast<llzk::component::StructType>(type)) {
    return structType;
  }
  if (auto arrType = mlir::dyn_cast<llzk::array::ArrayType>(type)) {
    return _subcomponent_type(arrType.getElementType());
  }
  return {};
}

class WeakestPreconditionAnalysis {
  llzk::component::StructDefOp structDef;
  cvc5::TermManager mgr;
  llzk::Field field;

  TermBuilder builder;
  mlir::SymbolTableCollection tables;

  void calculateWP(mlir::scf::IfOp ifOp, ConjunctionTerm &postcondition);
  void calculateWP(mlir::Operation *op, ConjunctionTerm &postcondition);
  void calculateWP(mlir::Block *block, ConjunctionTerm &postcondition);

  mlir::DenseMap<mlir::Value, cvc5::Term> valueExpressions;

  void initSubcomponents() {
    for (auto memberDef : structDef.getMemberDefs()) {

      if (auto structType = _subcomponent_type(memberDef.getType());
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

  llvm::FailureOr<cvc5::Term>
  computeInvariant(mlir::scf::ForOp loop, const ConjunctionTerm &postcondition);

public:
  WeakestPreconditionAnalysis(llzk::component::StructDefOp structDef,
                              llzk::Field field)
      : structDef{structDef}, field{field}, builder{mgr, field} {
    initSubcomponents();
    // initExpressions();
  }

  ImplicationTerm getPostcondition();
  void populateVerificationConditions();
  cvc5::Term generateVerificationConditions();

  void addEquivalentMember(llzk::component::MemberDefOp memberDef);

  void emit(llvm::raw_ostream &os);

  // Generated VCs after doing weakest precondition analysis
  cvc5::Term verificationConditions;
  // Toplevel for free variables that appear in the VCs
  TermBuilder::TermSet extraDecls;
  // Mod-p bounds on the free variable declarations
  TermBuilder::TermSet declBounds;
};

} // namespace lleq
