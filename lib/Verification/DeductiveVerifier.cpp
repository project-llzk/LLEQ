/**
 * Copyright 2026 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/DeductiveVerifier.h"
#include "Verification/SMTLIBEquivalenceEmitter.h"
#include "Verification/SolverUtils.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/SMT/IR/SMTOps.h>
#include <llzk/Util/ErrorHelper.h>

#include <llvm/Support/Program.h>
#include <optional>

using namespace mlir;
using namespace llzk::smt;

namespace lleq {

namespace {

// The StringRef is passed again by reference so that `consume_front(...)`
// actually mutates the string we're parsing outside the function
LogicalResult consumeValue(StringRef &model, StringRef var,
                           Counterexample &cex) {
  model = model.ltrim();
  if (!model.consume_front("(")) {
    return failure();
  }
  if (!model.consume_front(var)) {
    return failure();
  }
  if (model.consume_front("_c ")) {
    model.consumeInteger(10, cex.constraintModel);
  } else if (model.consume_front("_w ")) {
    model.consumeInteger(10, cex.witnessModel);
  } else {
    return failure();
  }
  if (!model.consume_front(")")) {
    return failure();
  }
  return success();
}

FailureOr<Counterexample> _parse_model(StringRef model, StringRef var) {
  Counterexample cex;

  if (!model.consume_front("(")) {
    return failure();
  }
  if (failed(consumeValue(model, var, cex))) {
    return failure();
  }
  if (failed(consumeValue(model, var, cex))) {
    return failure();
  }

  if (!model.consume_front(")")) {
    return failure();
  }
  return cex;
}

std::string resolveSolverPath() {
  if (std::optional<std::string> envPath =
          llvm::sys::Process::GetEnv("LLEQ_CVC5")) {
    if (llvm::sys::fs::can_execute(*envPath)) {
      return *envPath;
    }
    llvm::report_fatal_error(StringRef{"LLEQ_CVC5 is set to '"} + *envPath +
                             "', but that path is not executable");
  }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  // llvm::ErrorOr<std::string> appears to be deprecated, but silence the
  // warning for now
  auto solverPath = llvm::sys::findProgramByName("cvc5");
#pragma clang diagnostic pop
  if (!solverPath) {
    llvm::report_fatal_error(
        "could not find an executable cvc5 binary; install cvc5 or set "
        "LLEQ_CVC5 to the full path of the solver binary");
  }
  return *solverPath;
}

FailureOr<MemberEquivalenceResult> _invoke_solver(StringRef query,
                                                  StringRef var) {
  auto solverPath = resolveSolverPath();
  // Set a one second timeout for each check-sat query for now
  SmallVector<StringRef> args{solverPath, "--produce-models", "--tlimit-per",
                              "1000"};
  auto output = invokeSolverOnQuery(solverPath, args, query,
                                    /*passQueryFileAsArg=*/false);
  if (failed(output)) {
    return failure();
  }

  StringRef stdoutRef{*output};
  stdoutRef = stdoutRef.ltrim();
  auto lineEnd = stdoutRef.find_first_of("\r\n");
  StringRef result =
      lineEnd == StringRef::npos ? stdoutRef : stdoutRef.take_front(lineEnd);
  StringRef remainder =
      lineEnd == StringRef::npos ? StringRef{} : stdoutRef.drop_front(lineEnd);
  remainder = remainder.ltrim();

  if (result == "unsat") {
    return success(MemberEquivalenceResult{});
  } else if (result == "sat") {
    auto modelEnd = remainder.find_first_of("\r\n");
    StringRef model = modelEnd == StringRef::npos
                          ? remainder
                          : remainder.take_front(modelEnd);
    return _parse_model(model, var);
  }
  return failure();
}
} // namespace

LogicalResult DeductiveVerifier::generateBaseQuery() {
  if (baseQuery.has_value()) {
    baseQuery.reset();
  }
  baseQuery.emplace();
  llvm::raw_string_ostream os{*baseQuery};
  return emitSMTLIBEncoding(structDef, os, field.name());
}

FailureOr<MemberEquivalenceResult>
DeductiveVerifier::proveEquivalence(StringRef memberName) const {
  llzk::ensure(baseQuery.has_value(), "generateBaseQuery() not called");

  std::string query = *baseQuery;
  llvm::raw_string_ostream queryStream{query};
  queryStream << "(assert (not (= (mod " << memberName << "_w " << field.prime()
              << ") (mod " << memberName << "_c " << field.prime() << "))))\n";
  queryStream << "(check-sat)\n";
  queryStream << "(get-value (" << memberName << "_c " << memberName
              << "_w))\n";

  return _invoke_solver(query, memberName);
}

StructVerificationResult
DeductiveVerifier::verifyStruct(const DenseSet<StringRef> &members) {
  StructVerificationResult result;
  llzk::ensure(baseQuery.has_value() || succeeded(generateBaseQuery()),
               "failed to generate SMT query for struct @" +
                   structDef.getSymName());

  for (auto memberName : members) {
    auto memberResult = proveEquivalence(memberName);
    if (failed(memberResult)) {
      result.unknownMembers.insert(memberName);
    } else if (memberResult->equivalent) {
      // No counterexample, so they're equivalent
      result.equivalentMembers.insert(memberName);
    } else {
      // Map the inequivalent members to the counterexample
      result.inequivalentMembers.insert(
          {memberName, *memberResult->counterexample});
    }
  }
  return result;
}

void DeductiveVerifier::addExtraAssertions(ArrayRef<std::string> assertions) {
  if (!baseQuery.has_value()) {
    baseQuery.emplace();
  }
  for (auto assertion : assertions) {
    *baseQuery += (assertion + "\n");
  }
}

} // namespace lleq
