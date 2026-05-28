/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cvc5/cvc5.h>

#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/Value.h>

namespace lleq {

struct ImplicationTerm {
  std::vector<cvc5::Term> antecedents;
  cvc5::Term consequent;

  static ImplicationTerm of(cvc5::Term term) {
    return ImplicationTerm{{}, term};
  }

  void addAntecedent(cvc5::Term term) { antecedents.push_back(term); }
  cvc5::Term buildTerm(cvc5::TermManager &mgr) {
    auto antecedent = mgr.mkTerm(cvc5::Kind::AND, antecedents);
    return mgr.mkTerm(cvc5::Kind::IMPLIES, {antecedent, consequent});
  }

  void substitute(cvc5::Term oldTerm, cvc5::Term newTerm) {
    for (auto &antecedent : antecedents) {
      antecedent = antecedent.substitute(oldTerm, newTerm);
    }
    consequent = consequent.substitute(oldTerm, newTerm);
  }
};

class WeakestPreconditionAnalysis {
  llzk::component::StructDefOp structDef;
  cvc5::TermManager mgr;

  cvc5::Term getExpression(mlir::Operation *op);

  void calculateWP(mlir::Operation *op, ImplicationTerm &postcondition);
  void calculateWP(mlir::Block *block, ImplicationTerm &postcondition);

  cvc5::Term getConstant(mlir::Value value);
  cvc5::Term getConstant(mlir::StringRef memberName, bool isWitness);

  llvm::DenseMap<mlir::Value, cvc5::Term> constants;
  llvm::StringMap<cvc5::Term> witnessMembers, constraintMembers;

public:
  WeakestPreconditionAnalysis(llzk::component::StructDefOp structDef)
      : structDef{structDef} {}

  cvc5::Term generateVerificationConditions();
};

cvc5::Term getPostcondition(llzk::component::StructDefOp, cvc5::TermManager &);
} // namespace lleq
