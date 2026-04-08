/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/SMTAPI.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/Field.h>
#include <mlir/IR/Value.h>

namespace lleq {

enum class EquivalenceResult { Equivalent, Inequivalent, Unknown };

/// The main driver class for the deductive verifier, this holds an SMT solver
/// instance and queries it to prove equivalence/inequivalence of struct
/// signals. A new instance of this class should be constructed for each struct
/// being verified.
class DeductiveVerifier {
  llvm::SMTSolverRef solver;
  llvm::DenseMap<mlir::Value, llvm::SMTExprRef> smtExprs;
  llzk::Field field;

  llvm::LogicalResult _process_op(mlir::Operation *);

public:
  /// Set up the solver instance with all the assertions contained within the
  /// @product function of the provided struct
  llvm::LogicalResult prepareSolver(llzk::component::StructDefOp);

  /// Verify whether the witness and constraint version of provided signal are
  /// guaranteed to be equivalent
  EquivalenceResult verify(llzk::component::MemberDefOp);
};
} // namespace lleq
