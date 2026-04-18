/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/DeductiveVerifier.h"
#include "Verification/SMTLIBEquivalenceEmitter.h"

#include <fstream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorOr.h>
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

FailureOr<std::string> resolveSolverPath(std::string &errorMessage) {
  if (std::optional<std::string> envPath =
          llvm::sys::Process::GetEnv("LLEQ_CVC5")) {
    if (llvm::sys::fs::can_execute(*envPath)) {
      return *envPath;
    }
    errorMessage = "LLEQ_CVC5 is set to '" + *envPath +
                   "', but that path is not executable";
    return failure();
  }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  // llvm::ErrorOr<std::string> appears to be deprecated, but silence the
  // warning for now
  auto solverPath = llvm::sys::findProgramByName("cvc5");
#pragma clang diagnostic pop
  if (!solverPath) {
    errorMessage =
        "could not find an executable cvc5 binary; install cvc5 or set "
        "LLEQ_CVC5 to the full path of the solver binary";
    return failure();
  }
  return *solverPath;
}

FailureOr<MemberEquivalenceResult> _invoke_solver(StringRef query,
                                                  StringRef var) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  // TODO: std::tmpnam is deprecated, because between generating the filename
  // and opening it via `ExecuteAndWait`, another process may claim the same
  // name and create a race condition. Modern variants that atomically open the
  // file don't work here because `ExecuteAndWait` exepcts a filename and not a
  // handle; we should implement a workaround at some point
  auto tempStdin = std::tmpnam(nullptr);
  auto tempStdout = std::tmpnam(nullptr);
#pragma clang diagnostic pop

  // Write the query to a temp file
  std::ofstream os{tempStdin};
  os << query.data();
  os.close();

  std::string solverError;
  FailureOr<std::string> solverPath = resolveSolverPath(solverError);
  if (failed(solverPath)) {
    llvm::report_fatal_error(solverError.c_str());
  }

  SmallVector<StringRef> args{"cvc5", "--produce-models"};

  std::string error;
  auto code = llvm::sys::ExecuteAndWait(
      *solverPath, args,
      /*Env=*/std::nullopt,
      /*Redirects=*/
      {std::string{tempStdin}, std::string{tempStdout}, ""}, 0, 0, &error);
  if (code) {
    llvm::report_fatal_error(error.c_str());
    return failure();
  }

  std::ifstream is{tempStdout};
  std::string result;
  is >> result;

  if (result == "unsat") {
    return success(MemberEquivalenceResult{});
  } else if (result == "sat") {
    char model[256];
    // Skip the newline after the unsat/sat result
    is.getline(model, 256);
    is.getline(model, 256);
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

StructVerificationResult DeductiveVerifier::verifyStruct() {
  StructVerificationResult result;
  // I'm pretty sure llzk::ensure isn't compiled out in release builds
  llzk::ensure(succeeded(generateBaseQuery()),
               "failed to generate SMT query for struct @" +
                   structDef.getSymName());

  for (auto memberDef : structDef.getMemberDefs()) {
    SmallVector<char> _memberName;
    StringRef memberName = memberDef.getSymName();
    auto memberResult = proveEquivalence(memberName);
    llzk::ensure(succeeded(memberResult),
                 "failed to prove equivalence/inequivalence for member @" +
                     structDef.getSymName() + "::" + memberName);

    if (memberResult->equivalent) {
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

} // namespace lleq
