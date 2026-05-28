/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Verification/TermBuilderUtils.h"
#include <cvc5/cvc5.h>

#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/Field.h>
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
  llzk::Field field;

  TermBuilder builder;

  cvc5::Term getExpression(mlir::Operation *op);

  void calculateWP(mlir::Operation *op, ImplicationTerm &postcondition);
  void calculateWP(mlir::Block *block, ImplicationTerm &postcondition);

public:
  WeakestPreconditionAnalysis(llzk::component::StructDefOp structDef,
                              llzk::Field field)
      : structDef{structDef}, field{field}, builder{mgr, field} {}

  cvc5::Term generateVerificationConditions();
};

cvc5::Term getPostcondition(llzk::component::StructDefOp, cvc5::TermManager &);
} // namespace lleq
