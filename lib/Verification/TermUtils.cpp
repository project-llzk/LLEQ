/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/TermUtils.h"

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace lleq {
cvc5::Term TermBuilder::getConstant(Value value) {
  if (auto it = constants.find(value); it != constants.end()) {
    return it->second;
  }

  std::optional<std::string> name;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    name.emplace("arg" + std::to_string(blockArg.getArgNumber()));
  }

  auto newConst = mgr.mkConst(mgr.getIntegerSort(), name);
  constants.insert({value, newConst});
  return newConst;
}
cvc5::Term TermBuilder::getConstant(StringRef memberName, bool isWitness) {
  auto &memberMap = isWitness ? witnessMembers : constraintMembers;

  if (auto it = memberMap.find(memberName); it != memberMap.end()) {
    return it->second;
  }
  auto newTerm = mgr.mkConst(mgr.getIntegerSort(),
                             (memberName + (isWitness ? "_w" : "_c")).str());
  memberMap.insert({memberName, newTerm});
  return newTerm;
}

llvm::DenseMap<Value, cvc5::Term> constants;
llvm::StringMap<cvc5::Term> witnessMembers, constraintMembers;

cvc5::Term TermBuilder::getInteger(llvm::DynamicAPInt val) {
  std::string str;
  llvm::raw_string_ostream os{str};
  val.print(os);
  return mgr.mkInteger(str);
}

cvc5::Term TermBuilder::reduceMod(Value val, llvm::DynamicAPInt mod) {
  return mgr.mkTerm(cvc5::Kind::INTS_MODULUS,
                    {getConstant(val), getInteger(mod)});
}

cvc5::Term TermBuilder::assertEqual(Value a, Value b) {
  return mgr.mkTerm(cvc5::Kind::EQUAL,
                    {reduceMod(a, field.prime()), reduceMod(b, field.prime())});
}

cvc5::Term TermBuilder::assertEqual(cvc5::Term a, mlir::Value b) {
  return mgr.mkTerm(cvc5::Kind::EQUAL, {a, reduceMod(b, field.prime())});
}

cvc5::Term TermBuilder::arrayRead(Value array, Value index) {
  return mgr.mkTerm(cvc5::Kind::SELECT,
                    {getConstant(array), getConstant(index)});
}
cvc5::Term TermBuilder::arrayWrite(Value array, Value index, Value value) {
  return mgr.mkTerm(cvc5::Kind::STORE, {getConstant(array), getConstant(index),
                                        reduceMod(value, field.prime())});
}

cvc5::Term ImplicationTerm::buildTerm(cvc5::TermManager &mgr) {
  auto antecedent = mgr.mkTerm(cvc5::Kind::AND, antecedents);
  return mgr.mkTerm(cvc5::Kind::IMPLIES, {antecedent, consequent});
}

void ImplicationTerm::substitute(cvc5::Term oldTerm, cvc5::Term newTerm) {
  for (auto &antecedent : antecedents) {
    antecedent = antecedent.substitute(oldTerm, newTerm);
  }
  consequent = consequent.substitute(oldTerm, newTerm);
}

} // namespace lleq
