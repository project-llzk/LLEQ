/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/SMTAPI.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/Field.h>
#include <mlir/IR/Value.h>
#include <optional>

namespace lleq {

/// When a signal is proven inequivalent, we have a model that shows possible
/// differing values for the witness and constraint
struct Counterexample {
  llvm::APInt witnessModel, constraintModel;
};

/// Represents the result of verifying a single struct member. If `equivalent`
/// is true, the witness/constraint versions of the member are equivalent.
/// Otherwise, `counterexample` is populated.
struct MemberEquivalenceResult {
  bool equivalent;
  std::optional<Counterexample> counterexample;

  MemberEquivalenceResult() : equivalent{true}, counterexample{std::nullopt} {}
  MemberEquivalenceResult(Counterexample cex)
      : equivalent{false}, counterexample{cex} {}
};

/// The result of verifying a struct, holding the set of all member signals
/// proven equivalent, models for all member signals proven inequivalent, and
/// the set of members for which we failed to prove equivalence/inequivalence
struct StructVerificationResult {
  llvm::SmallVector<llvm::StringRef> unknownMembers;
  llvm::SmallVector<llvm::StringRef> equivalentMembers;
  llvm::DenseMap<llvm::StringRef, Counterexample> inequivalentMembers;
};

/// The main driver class for the deductive verifier, this generates an SMT
/// query to prove equivalence of each struct member. A new instance of this
/// class should be constructed for each struct.
class DeductiveVerifier {
  llzk::component::StructDefOp structDef;
  std::optional<std::string> baseQuery;
  llzk::Field field;

public:
  DeductiveVerifier(llzk::component::StructDefOp structDef, llzk::Field field)
      : structDef{structDef}, field{field} {}

  /// Generate an SMT encoding of the struct body
  llvm::LogicalResult generateBaseQuery();

  /// Attempt to prove equivalence or inequivalence of the given struct member,
  /// or return failure if the solver could prove neither.
  llvm::FailureOr<MemberEquivalenceResult>
  proveEquivalence(llvm::StringRef memberName) const;

  /// Verify equivalence of all struct members, including non-signals (since
  /// equivalence/inequivalence of a non-signal could later help prove
  /// equivalence/inequivalence of a signal)
  StructVerificationResult
  verifyStruct(llvm::ArrayRef<llvm::StringRef> members = {});

  void addExtraAssertions(llvm::ArrayRef<std::string> assertions);

  void dump(llvm::raw_ostream &os) {
    if (baseQuery.has_value()) {
      os << *baseQuery << '\n';
    }
  }
};
} // namespace lleq
