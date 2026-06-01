/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cvc5/cvc5.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/StringMap.h>
#include <llzk/Util/Field.h>
#include <mlir/IR/Value.h>

namespace lleq {

struct TermBuilder {

  cvc5::TermManager &mgr;
  llzk::Field field;

  cvc5::Term getInteger(llvm::DynamicAPInt val);
  cvc5::Term getConstant(mlir::Value value);
  cvc5::Term getConstant(mlir::StringRef memberName, bool isWitness);

  llvm::DenseMap<mlir::Value, cvc5::Term> constants;
  llvm::StringMap<cvc5::Term> witnessMembers, constraintMembers;

  cvc5::Term reduceMod(mlir::Value val, llvm::DynamicAPInt mod);
  cvc5::Term assertEqual(mlir::Value a, mlir::Value b);
  cvc5::Term assertEqual(cvc5::Term a, mlir::Value b);
  cvc5::Term arrayRead(mlir::Value array, mlir::Value index);
  cvc5::Term arrayWrite(mlir::Value array, mlir::Value index,
                        mlir::Value value);

  TermBuilder(cvc5::TermManager &mgr, llzk::Field field)
      : mgr{mgr}, field{field} {}
};

struct ImplicationTerm {
  std::vector<cvc5::Term> antecedents;
  cvc5::Term consequent;

  static ImplicationTerm of(cvc5::Term term) {
    return ImplicationTerm{{}, term};
  }

  void addAntecedent(cvc5::Term term) { antecedents.push_back(term); }
  cvc5::Term buildTerm(cvc5::TermManager &mgr);
  void substitute(cvc5::Term oldTerm, cvc5::Term newTerm);
};

} // namespace lleq
