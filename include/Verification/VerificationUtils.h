/**
 * Copyright 2026 Veridise Inc.
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

llvm::LogicalResult ensureProductFunc(mlir::ModuleOp module,
                                      llzk::component::StructDefOp structDef);

std::string buildSMTQuery(cvc5::Term query, TermBuilder &builder,
                          llzk::Field field);

llvm::FailureOr<std::string>
invokeSolverOnQuery(llvm::StringRef solverPath,
                    llvm::ArrayRef<llvm::StringRef> args,
                    llvm::StringRef query, bool passQueryFileAsArg);

llvm::FailureOr<SolverRunResult>
invokeSolverPortfolio(llvm::ArrayRef<SolverInvocationSpec> solvers,
                      llvm::StringRef query);

llvm::SmallVector<SolverInvocationSpec> getWeakestPreconditionPortfolio();

bool checkUnsatWithZ3(llvm::StringRef query);
bool checkUnsatWithPortfolio(llvm::StringRef query);
} // namespace lleq
