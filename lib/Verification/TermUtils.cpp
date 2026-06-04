/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/TermUtils.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/DynamicAPIntHelper.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>

using namespace mlir;

namespace lleq {

static inline void traverse(cvc5::Term term, const TermBuilder::TermSet &vars,
                            TermBuilder::TermSet &acc) {
  if (vars.find(term) != vars.end()) {
    acc.insert(term);
  } else {
    for (auto child : term) {
      traverse(child, vars, acc);
    }
  }
}

TermBuilder::TermSet TermBuilder::getExtraDecls(cvc5::Term term) {
  TermSet vars;
  for (const auto &[_, c] : constants) {
    vars.insert(c);
  }
  for (const auto &[_, c] : witnessMembers) {
    vars.insert(c);
  }
  for (const auto &[_, c] : constraintMembers) {
    vars.insert(c);
  }

  TermSet decls;
  traverse(term, vars, decls);
  return decls;
}

static inline cvc5::Sort sortOfType(Type type, cvc5::TermManager &mgr) {
  if (type.isSignlessInteger() &&
      dyn_cast<IntegerType>(type).getIntOrFloatBitWidth() == 1) {
    return mgr.getBooleanSort();
  }
  if (auto arrType = dyn_cast<llzk::array::ArrayType>(type)) {
    return mgr.mkArraySort(mgr.getIntegerSort(),
                           sortOfType(arrType.getElementType(), mgr));
  }
  return mgr.getIntegerSort();
}

cvc5::Term TermBuilder::getConstant(Value value) {
  if (auto it = constants.find(value); it != constants.end()) {
    return it->second;
  }

  std::optional<std::string> name;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    name.emplace("arg" + std::to_string(blockArg.getArgNumber()));
  }

  auto newConst = mgr.mkConst(sortOfType(value.getType(), mgr), name);
  constants.insert({value, newConst});
  termTypes.insert({newConst, value.getType()});
  return newConst;
}
cvc5::Term TermBuilder::getConstant(llzk::component::MemberDefOp memberDef,
                                    bool isWitness) {
  auto &memberMap = isWitness ? witnessMembers : constraintMembers;

  auto memberName = memberDef.getSymName();
  if (auto it = memberMap.find(memberName); it != memberMap.end()) {
    return it->second;
  }

  auto newTerm = mgr.mkConst(sortOfType(memberDef.getType(), mgr),
                             (memberName + (isWitness ? "_w" : "_c")).str());
  memberMap.insert({memberName, newTerm});
  termTypes.insert({newTerm, memberDef.getType()});
  return newTerm;
}

cvc5::Term TermBuilder::getInteger(llvm::DynamicAPInt val) {
  std::string str;
  llvm::raw_string_ostream os{str};
  val.print(os);
  return mgr.mkInteger(str);
}

cvc5::Term TermBuilder::_reduce_mod_impl(cvc5::Term val,
                                         llvm::DynamicAPInt mod) {
  return mgr.mkTerm(cvc5::Kind::INTS_MODULUS, {val, getInteger(mod)});
}

// Returns the array length for an ArrayType, or nullopt
static inline std::optional<int64_t> sizeOfType(Type type) {
  if (auto arrType = dyn_cast<llzk::array::ArrayType>(type)) {
    auto shape = arrType.getShape();
    llzk::ensure(shape.size() == 1, "multidimensional arrays not supported");
    return shape.front();
  }
  return {};
}

// Returns the array length if either term is an array of known length, nullopt
// if neither, and asserts failure if the lengths differ
static inline std::optional<int64_t> getArraySize(
    cvc5::Term a, cvc5::Term b,
    std::unordered_map<cvc5::Term, Type, std::hash<cvc5::Term>> termTypes) {
  std::optional<int64_t> arraySize;
  if (auto ait = termTypes.find(a); ait != termTypes.end()) {
    arraySize = sizeOfType(ait->second);
  }
  if (auto bit = termTypes.find(b); bit != termTypes.end()) {
    auto size = sizeOfType(bit->second);
    llzk::ensure(!arraySize.has_value() || arraySize == size,
                 "incompatible array sizes");
    return size;
  }
  return arraySize;
}

cvc5::Term TermBuilder::_assert_equal_impl(cvc5::Term a, cvc5::Term b) {
  Value arrVal;
  if (a.getSort().isArray() && b.getSort().isArray()) {
    auto index = mgr.mkVar(mgr.getIntegerSort(), "i");
    // TODO: constrain `i` to the array bounds
    auto size = getArraySize(a, b, termTypes);
    auto eqTerm = _assert_equal_impl(_array_read_impl(a, index),
                                     _array_read_impl(b, index));
    if (size.has_value()) {
      auto bounds =
          mgr.mkTerm(cvc5::Kind::AND,
                     {mgr.mkTerm(cvc5::Kind::LEQ, {getInteger(0), index}),
                      mgr.mkTerm(cvc5::Kind::LT, {index, getInteger(*size)})});
      eqTerm = mgr.mkTerm(cvc5::Kind::IMPLIES, {bounds, eqTerm});
    }
    return mgr.mkTerm(cvc5::Kind::FORALL,
                      {
                          mgr.mkTerm(cvc5::Kind::VARIABLE_LIST, {index}),
                          eqTerm,
                      });
  }

  return mgr.mkTerm(cvc5::Kind::EQUAL, {_reduce_mod_impl(a, field.prime()),
                                        _reduce_mod_impl(b, field.prime())});
}

cvc5::Term TermBuilder::_array_read_impl(cvc5::Term arr, cvc5::Term index) {
  return mgr.mkTerm(cvc5::Kind::SELECT, {arr, index});
}

cvc5::Term TermBuilder::_array_write_impl(cvc5::Term arr, cvc5::Term index,
                                          cvc5::Term elem) {
  return mgr.mkTerm(cvc5::Kind::STORE, {arr, index, elem});
}

cvc5::Term ImplicationTerm::buildTerm(cvc5::TermManager &mgr) {
  if (antecedents.empty()) {
    return consequent;
  }

  auto antecedent =
      mgr.mkTerm(cvc5::Kind::AND, {antecedents.begin(), antecedents.end()});
  return mgr.mkTerm(cvc5::Kind::IMPLIES, {antecedent, consequent});
}

void ImplicationTerm::substitute(cvc5::Term oldTerm, cvc5::Term newTerm) {
  for (auto &antecedent : antecedents) {
    antecedent = antecedent.substitute(oldTerm, newTerm);
  }
  consequent = consequent.substitute(oldTerm, newTerm);
}

void ConjunctionTerm::addConjunct(const ImplicationTerm &term) {
  conjuncts.push_back(term);
}

void ConjunctionTerm::substitute(cvc5::Term oldTerm, cvc5::Term newTerm) {
  for (auto &conjunct : conjuncts) {
    conjunct.substitute(oldTerm, newTerm);
  }
}

cvc5::Term ConjunctionTerm::buildTerm(cvc5::TermManager &mgr) {
  llzk::ensure(!conjuncts.empty(), "cannot build term from empty conjunction");

  if (conjuncts.size() == 1) {
    return conjuncts.front().buildTerm(mgr);
  }
  auto builtConjuncts = llvm::map_to_vector(
      conjuncts, [&mgr](ImplicationTerm term) { return term.buildTerm(mgr); });
  return mgr.mkTerm(cvc5::Kind::AND,
                    {builtConjuncts.begin(), builtConjuncts.end()});
}

void ConjunctionTerm::addAntecedent(cvc5::Term term) {
  for (auto &conjunct : conjuncts) {
    conjunct.addAntecedent(term);
  }
}

} // namespace lleq
