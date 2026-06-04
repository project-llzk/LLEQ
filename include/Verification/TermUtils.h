/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cvc5/cvc5.h>
#include <functional>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/ADT/StringMap.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/DynamicAPIntHelper.h>
#include <llzk/Util/Field.h>
#include <mlir/IR/Value.h>

namespace lleq {

template <class T>
concept FormulaTerm =
    std::convertible_to<T, cvc5::Term> || std::convertible_to<T, mlir::Value>;

// A helper class for building common term shapes from MLIR SSA values
struct TermBuilder {

  using TermSet = std::unordered_set<cvc5::Term, std::hash<cvc5::Term>>;

  // Build an integer constant
  cvc5::Term getInteger(llvm::DynamicAPInt val);
  cvc5::Term getInteger(auto val) {
    return getInteger(llzk::toDynamicAPInt(val));
  }

  // Return a constant term of the appropriate sort for an SSA value
  cvc5::Term getConstant(mlir::Value value);

  // Return a constant of the appropriate sort for a struct member
  cvc5::Term getConstant(llzk::component::MemberDefOp memberDef,
                         bool isWitness);

  // Return all free variables in `term` that are tracked as constants
  TermSet getExtraDecls(cvc5::Term term);

  // Build terms bouding each variable in `decls` in the range `[0, p - 1]`
  TermSet getDeclBounds(TermSet decls, llvm::DynamicAPInt prime);

  cvc5::Term reduceMod(FormulaTerm auto val, llvm::DynamicAPInt mod) {
    return _reduce_mod_impl(_get_term(val), mod);
  }

  cvc5::Term assertEqual(FormulaTerm auto a, FormulaTerm auto b) {
    return _assert_equal_impl(_get_term(a), _get_term(b));
  }

  cvc5::Term arrayRead(FormulaTerm auto array, FormulaTerm auto index) {
    return _array_read_impl(_get_term(array), _get_term(index));
  }

  cvc5::Term arrayWrite(FormulaTerm auto array, FormulaTerm auto index,
                        FormulaTerm auto elem) {
    return _array_write_impl(_get_term(array), _get_term(index),
                             _get_term(elem));
  }

  TermBuilder(cvc5::TermManager &mgr, llzk::Field field)
      : mgr{mgr}, field{field} {}

private:
  llvm::DenseMap<mlir::Value, cvc5::Term> constants;
  llvm::StringMap<cvc5::Term> witnessMembers, constraintMembers;

  std::unordered_map<cvc5::Term, mlir::Type, std::hash<cvc5::Term>> termTypes;

  cvc5::Term _is_mod(cvc5::Term, llvm::DynamicAPInt);
  cvc5::Term _reduce_mod_impl(cvc5::Term, llvm::DynamicAPInt);
  cvc5::Term _assert_array_equal_impl(cvc5::Term, cvc5::Term,
                                      std::optional<llvm::DynamicAPInt>);
  cvc5::Term _assert_equal_impl(cvc5::Term, cvc5::Term);
  cvc5::Term _assert_equal_impl(cvc5::Term, mlir::Value);
  cvc5::Term _array_read_impl(cvc5::Term, cvc5::Term);
  cvc5::Term _array_write_impl(cvc5::Term, cvc5::Term, cvc5::Term);

  cvc5::Term _array_quantified_term(std::function<cvc5::Term(cvc5::Term)>,
                                    std::optional<int64_t>);

  cvc5::Term _get_term(FormulaTerm auto t) {
    using T = decltype(t);
    if constexpr (std::convertible_to<T, mlir::Value>) {
      return getConstant(t);
    } else {
      return t;
    }
  }

  cvc5::TermManager &mgr;
  llzk::Field field;
};

// `ImplicationTerm` and `ConjunctionTerm` are necessary because WP naturally
// produces terms that look like `(A1 -> ... -> An) /\ (B1 -> ... -> Bn) /\ ...`
// But its better to encode these as `((A1 /\ ... An-1) -> An) /\ ...`
// So we have to track our own implication and top-level conjunction terms

// A term of the shape (A1 /\ ... An-1) -> An
struct ImplicationTerm {
  llvm::SmallVector<cvc5::Term> antecedents;
  cvc5::Term consequent;

  static ImplicationTerm of(cvc5::Term term) {
    return ImplicationTerm{{}, term};
  }

  void addAntecedent(cvc5::Term term) { antecedents.push_back(term); }

  // Just forward to cvc5 substitution on each subterm
  void substitute(cvc5::Term oldTerm, cvc5::Term newTerm);

  // Materialize a cvc5 Term from the contained subterms
  cvc5::Term buildTerm(cvc5::TermManager &mgr);
};

// A term of the shape (A1 -> ... An) /\ ...
struct ConjunctionTerm {
  llvm::SmallVector<ImplicationTerm> conjuncts;

  static ConjunctionTerm of(const ImplicationTerm &term) {
    return ConjunctionTerm{{term}};
  }

  static ConjunctionTerm of(cvc5::Term term) {
    return of(ImplicationTerm::of(term));
  }

  void addConjunct(const ImplicationTerm &term);
  void addConjuncts(const ConjunctionTerm &term) {
    for (const auto &conj : term.conjuncts) {
      addConjunct(conj);
    }
  }

  // Just forward to cvc5 substitution on each subterm
  void substitute(cvc5::Term oldTerm, cvc5::Term newTerm);

  // Materialize a cvc5 Term from the contained subterms
  cvc5::Term buildTerm(cvc5::TermManager &mgr);

  // Add the antecedent to each conjunct
  void addAntecedent(cvc5::Term term);
};

} // namespace lleq
