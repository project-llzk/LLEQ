/**
 * Copyright 2026 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cvc5/cvc5.h>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/DynamicAPIntHelper.h>
#include <llzk/Util/Field.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>
#include <optional>

namespace lleq {

using ArrayShape = llvm::SmallVector<int64_t, 4>;

static inline std::optional<llzk::component::MemberWriteOp>
getArrayDestination(mlir::Value array) {
  std::optional<llzk::component::MemberWriteOp> destination;
  for (auto *user : array.getUsers()) {
    if (auto writeOp = mlir::dyn_cast<llzk::component::MemberWriteOp>(user)) {
      if (destination.has_value()) {
        mlir::emitWarning(array.getLoc())
            << "array value written to multiple members\n";
        return {};
      }
      destination.emplace(writeOp);
    }
  }
  return destination;
}

static inline cvc5::Term conjunctAll(llvm::ArrayRef<cvc5::Term> terms,
                                     cvc5::TermManager &mgr) {
  if (terms.size() == 0) {
    return mgr.mkBoolean(true);
  }
  if (terms.size() == 1) {
    return terms.front();
  }
  return mgr.mkTerm(cvc5::Kind::AND, {terms.begin(), terms.end()});
}

static inline cvc5::Term disjunctAll(llvm::ArrayRef<cvc5::Term> terms,
                                     cvc5::TermManager &mgr) {
  if (terms.size() == 0) {
    return mgr.mkBoolean(false);
  }
  if (terms.size() == 1) {
    return terms.front();
  }
  return mgr.mkTerm(cvc5::Kind::OR, {terms.begin(), terms.end()});
}

template <class T>
concept FormulaTerm =
    std::convertible_to<T, cvc5::Term> || std::convertible_to<T, mlir::Value>;

/// A helper class for building common term shapes from MLIR SSA values
struct TermBuilder {
  using TermSet = std::unordered_set<cvc5::Term, std::hash<cvc5::Term>>;

  cvc5::TermManager &manager() { return mgr; }

  // Build an integer constant
  cvc5::Term getInteger(llvm::DynamicAPInt val, bool asBoolean = false);
  cvc5::Term getInteger(auto val, bool asBoolean = false) {
    return getInteger(llzk::toDynamicAPInt(val), asBoolean);
  }

  void addEquivalentMember(llzk::component::MemberDefOp memberDef);

  // Return a constant term of the appropriate sort for an SSA value
  cvc5::Term getConstant(mlir::Value value);

  // Return a constant term for a polymorphic variable (assume integer-sort)
  cvc5::Term getConstant(llvm::StringRef symbolName);

  // Return a constant of the appropriate sort for a struct member
  cvc5::Term getConstant(llzk::component::MemberDefOp memberDef,
                         bool isWitness);
  cvc5::Term getConstant(llvm::StringRef symbolName, mlir::Type type,
                         bool isWitness);

  // Trace the use-def chain to build an expression for an SSA value
  cvc5::Term getExpression(mlir::Value value);

  // Return all free variables in `term` that are tracked as constants
  TermSet getExtraDecls(cvc5::Term term);

  // Build terms bounding each variable in `decls` in the range `[0, p - 1]`
  TermSet getDeclBounds(TermSet decls, llvm::DynamicAPInt prime);

  // Emit auxiliary declarations for subcomponents
  void emitSubcmpDeclarations(llvm::raw_ostream &os);

  // Returns a call to `(init-@subcmp args...)`
  cvc5::Term initSubcmp(llzk::component::StructDefOp subcmp,
                        llvm::ArrayRef<mlir::Value> args);

  // Returns a call to `(read-"subcmp"-"member" %subcmp)`
  cvc5::Term readSubcmpMember(mlir::Value subcmp,
                              llzk::component::MemberDefOp member);

  cvc5::Term reduceMod(FormulaTerm auto val, llvm::DynamicAPInt mod) {
    return _reduce_mod_impl(_get_term(val), mod);
  }

  cvc5::Term assertEqual(FormulaTerm auto a, FormulaTerm auto b) {
    return _assert_equal_impl(_get_term(a), _get_term(b));
  }

  cvc5::Term arrayRead(FormulaTerm auto array, FormulaTerm auto index) {
    cvc5::Term indexTerm = _get_term(index);
    return _array_read_impl(_get_term(array), {indexTerm});
  }

  cvc5::Term arrayRead(FormulaTerm auto array,
                       llvm::ArrayRef<mlir::Value> indices) {
    llvm::SmallVector<cvc5::Term> indexTerms = llvm::map_to_vector(
        indices, [this](mlir::Value index) { return _get_term(index); });
    return _array_read_impl(_get_term(array), indexTerms);
  }

  cvc5::Term arrayRead(FormulaTerm auto array,
                       llvm::ArrayRef<cvc5::Term> indices) {
    return _array_read_impl(_get_term(array), indices);
  }

  cvc5::Term arrayWrite(FormulaTerm auto array, FormulaTerm auto index,
                        FormulaTerm auto elem) {
    cvc5::Term indexTerm = _get_term(index);
    return _array_write_impl(_get_term(array), {indexTerm}, _get_term(elem));
  }

  cvc5::Term arrayWrite(FormulaTerm auto array,
                        llvm::ArrayRef<mlir::Value> indices,
                        FormulaTerm auto elem) {
    llvm::SmallVector<cvc5::Term> indexTerms = llvm::map_to_vector(
        indices, [this](mlir::Value index) { return _get_term(index); });
    return _array_write_impl(_get_term(array), indexTerms, _get_term(elem));
  }

  cvc5::Term arrayWrite(FormulaTerm auto array,
                        llvm::ArrayRef<cvc5::Term> indices,
                        FormulaTerm auto elem) {
    return _array_write_impl(_get_term(array), indices, _get_term(elem));
  }

  TermBuilder(cvc5::TermManager &mgr, llzk::Field field)
      : mgr{mgr}, field{field} {}

  // Generate the auxiliary definitions for a subcomponent
  void populateSubcomponent(llzk::component::StructDefOp subcmp);

private:
  cvc5::Sort _sort_of_type(mlir::Type);

  llvm::DenseMap<mlir::Value, cvc5::Term> constants;
  llvm::DenseMap<mlir::Value, cvc5::Term> expressions;
  llvm::StringMap<cvc5::Term> witnessMembers, constraintMembers, polyMembers;
  TermSet auxiliaryBounds;

  std::unordered_map<cvc5::Term, mlir::Type, std::hash<cvc5::Term>> termTypes;

  // Auxiliary declarations for subcomponents
  llvm::DenseMap<llzk::component::StructType, cvc5::Sort> subcmpSorts;
  llvm::DenseMap<llzk::component::StructType, cvc5::Term> subcmpInits;
  llvm::DenseMap<llzk::component::MemberDefOp, cvc5::Term> subcmpMembers;

  // Term builder implementations
  cvc5::Term _is_mod(cvc5::Term, llvm::DynamicAPInt);
  cvc5::Term _reduce_mod_impl(cvc5::Term, llvm::DynamicAPInt);
  cvc5::Term _assert_array_equal_impl(cvc5::Term, cvc5::Term,
                                      std::optional<llvm::DynamicAPInt>);
  cvc5::Term _assert_equal_impl(cvc5::Term, cvc5::Term);
  cvc5::Term _assert_equal_impl(cvc5::Term, mlir::Value);
  cvc5::Term _array_read_impl(cvc5::Term, llvm::ArrayRef<cvc5::Term>);
  cvc5::Term _array_write_impl(cvc5::Term, llvm::ArrayRef<cvc5::Term>,
                               cvc5::Term);

  cvc5::Term _array_quantified_term(
      std::function<cvc5::Term(llvm::ArrayRef<cvc5::Term>)>,
      llvm::ArrayRef<int64_t>);

  cvc5::Term _get_term(FormulaTerm auto t) {
    using T = decltype(t);
    if constexpr (std::convertible_to<T, mlir::Value>) {
      return getExpression(t);
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

struct Range {
  cvc5::Term lb, ub, step;
  static Range fromValues(mlir::Value lb, mlir::Value ub, mlir::Value step,
                          TermBuilder &builder) {
    return Range{builder.getExpression(lb), builder.getExpression(ub),
                 builder.getExpression(step)};
  }
};

struct Annotation {
  bool isArray;
  std::optional<llvm::SmallVector<Range>> arraySlice;
};

/// A term of the shape (A1 /\ ... /\ An) -> (B1 /\ ... /\ Bm)
struct ImplicationTerm {
  llvm::SmallVector<cvc5::Term> antecedents;
  llvm::SmallVector<cvc5::Term> consequents;

  // Optional annotations on each consequent term. The annotation is present if
  // the term is expressing equality between two signals, and the
  // `arraySlice` field carries one bound range per array dimension.
  llvm::SmallVector<std::optional<Annotation>> annotations;

  static ImplicationTerm of(cvc5::Term term) {
    return ImplicationTerm{{}, {term}};
  }

  void addAntecedent(cvc5::Term term) { antecedents.push_back(term); }

  // Just forward to cvc5 substitution on each subterm
  void substitute(cvc5::Term oldTerm, cvc5::Term newTerm);

  // Materialize a cvc5 Term from the contained subterms
  cvc5::Term buildTerm(cvc5::TermManager &mgr);
};

/// A term of the shape (A1 -> ... An) /\ ...
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
