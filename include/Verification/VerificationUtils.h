/**
 * Copyright 2026 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Verification/TermUtils.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/Field.h>
#include <mlir/IR/BuiltinOps.h>

#include <string>

namespace lleq {
enum class SolverResultKind {
  Unsat,
  Sat,
  Unknown,
  ExecutionFailure,
  ParseFailure,
};

struct SolverInvocationSpec {
  std::string name;
  std::string path;
  llvm::SmallVector<std::string> args;
  bool passQueryFileAsArg = false;
};

struct SolverRunResult {
  SolverResultKind kind;
  std::string stdoutText;
  std::string solverName;
};

/// Run product alignment on the module if no @product function is found
llvm::LogicalResult ensureProductFunc(mlir::ModuleOp module,
                                      llzk::component::StructDefOp structDef);

/// Build an SMTLIB2-style query based on the given SMT term, together with
/// appropriate declarations. Assumes that `query` is be a term constructed via
/// the provided TermBuilder `builder`
std::string buildSMTQuery(cvc5::Term query, TermBuilder &builder,
                          llzk::Field field);

/// Spawn a solver process, pass the SMTLIB2-style query string, and return the
/// output
llvm::FailureOr<std::string>
invokeSolverOnQuery(llvm::StringRef solverPath,
                    llvm::ArrayRef<llvm::StringRef> args, llvm::StringRef query,
                    bool passQueryFileAsArg);

/// Spawn a portfolio of solver processes and return the first known result
llvm::FailureOr<SolverRunResult>
invokeSolverPortfolio(llvm::ArrayRef<SolverInvocationSpec> solvers,
                      llvm::StringRef query);

/// Default z3 + cvc5 portfolio
llvm::SmallVector<SolverInvocationSpec> getWeakestPreconditionPortfolio();

/// Convenient utilities
bool checkUnsatWithZ3(llvm::StringRef query);
bool checkUnsatWithPortfolio(llvm::StringRef query);
} // namespace lleq
