/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolicStore.h"
#include "Verification/DeductiveVerifier.h"
#include "Verification/SMTLIBEquivalenceEmitter.h"
#include "Verification/SymbolicVerification.h"
#include "lleq/CliOptions.h"
#include <cstdlib>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Analysis/AnalysisUtil.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/InitDialects.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/SymbolHelper.h>
#include <llzk/Util/SymbolLookup.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LogicalResult.h>

#define BUG_REPORT_URL "https://github.com/Veridise/LLEQ/issues"

using namespace lleq;

static inline void dumpStore(llzk::component::StructDefOp structDef) {
  llvm::outs() << "-- " << structDef.getSymName() << " --\n";
  SymbolicStore store;
  if (mlir::failed(store.buildStore(structDef))) {
    llvm::report_fatal_error("symbolic store construction failed");
  }
  store.dump(llvm::outs());
}

static inline void printDiag(mlir::Diagnostic &d) {
  switch (d.getSeverity()) {
  case mlir::DiagnosticSeverity::Error:
    llvm::WithColor::error() << d.getLocation() << ':' << d.str() << '\n';
    break;
  case mlir::DiagnosticSeverity::Warning:
    llvm::WithColor::warning() << d.getLocation() << ':' << d.str() << '\n';
    break;
  case mlir::DiagnosticSeverity::Remark:
    llvm::WithColor::remark() << d.getLocation() << ':' << d.str() << '\n';
    break;
  case mlir::DiagnosticSeverity::Note:
    llvm::WithColor::note() << d.getLocation() << ':' << d.str() << '\n';
    break;
  default:
    break;
  }

  for (auto &note : d.getNotes()) {
    printDiag(note);
  }
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  llvm::setBugReportMsg("PLEASE submit a bug report to " BUG_REPORT_URL
                        " and include the crash backtrace, relevant LLZK "
                        "files, and associated run script(s).\n");

  llvm::InitLLVM initLLVM(argc, argv);
  llvm::setBugReportMsg(
      "LLEQ has crashed! Please report the bug to contact@veridise.com\n");

  llvm::cl::ParseCommandLineOptions(argc, argv, "LLZK Equivalence Verifier\n");

  mlir::DialectRegistry registry;
  llzk::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    printDiag(diag);
    return llvm::success();
  });

  auto parserConfig = mlir::ParserConfig(&context);
  auto mod =
      mlir::parseSourceFile<mlir::ModuleOp>(cli::inputFile(), parserConfig);
  if (!mod) {
    llvm::errs() << "Failed to parse " << cli::inputFile() << '\n';
    return EXIT_FAILURE;
  }

  // TODO: split on :: and parse out FQN to use LLZK symbol lookup
  llzk::component::StructDefOp structDef;
  mod->walk([&structDef](llzk::component::StructDefOp s) {
    if (s.getSymName() == cli::smtStruct()) {
      if (structDef) {
        llvm::errs() << "found multiple structs named @" << cli::smtStruct()
                     << '\n';
        exit(EXIT_FAILURE);
      }
      structDef = s;
    }
  });

  if (!structDef) {
    llvm::errs() << "could not find struct @" << cli::smtStruct() << '\n';
    return EXIT_FAILURE;
  }

  switch (cli::subCmd()) {
  case cli::SubCmd::DumpStore:
    dumpStore(structDef);
    return EXIT_SUCCESS;
  case cli::SubCmd::Verify:
  case cli::SubCmd::DumpSmt: {
    auto field = llzk::Field::getField(cli::fieldName());

    llvm::SmallVector<std::string> extraAssertions;
    if (cli::enableStore()) {
      SymbolicVerifier symbolicVerifier{structDef};
      if (failed(symbolicVerifier.buildStore())) {
        return EXIT_FAILURE;
      }
      extraAssertions = symbolicVerifier.generateAssertions(field);
    }

    DeductiveVerifier deductiveVerifier{structDef, field};
    if (failed(deductiveVerifier.generateBaseQuery())) {
      return EXIT_FAILURE;
    }

    deductiveVerifier.addExtraAssertions(extraAssertions);

    if (cli::subCmd() == cli::SubCmd::DumpSmt) {
      deductiveVerifier.dump(llvm::outs());
      return EXIT_SUCCESS;
    }

    StructVerificationResult result = deductiveVerifier.verifyStruct();
    llvm::outs() << "The following members were proven equivalent:\n";
    for (auto member : result.equivalentMembers) {
      llvm::outs() << "+ @" << structDef.getSymName() << "::" << member << "\n";
    }
    llvm::outs() << "The following members were proven inequivalent:\n";
    for (auto [member, counterexample] : result.inequivalentMembers) {
      auto [w, c] = counterexample;
      llvm::outs() << "- @" << structDef.getSymName() << "::" << member << "\n";
      llvm::outs() << "\twitness: " << w << "\n";
      llvm::outs() << "\tconstraint: " << c << "\n";
    }
    return EXIT_SUCCESS;
  }
  }

  // if (cli::emitSMTLIB() || cli::disableStore()) {
  //   if (cli::dumpStore()) {
  //     llvm::errs() << "--dump-store cannot be combined with this option\n";
  //     return EXIT_FAILURE;
  //   }

  //   if (cli::smtStruct().empty()) {
  //     llvm::errs() << "--struct is required with this option\n";
  //     return EXIT_FAILURE;
  //   }

  //   if (cli::smtField().empty()) {
  //     llvm::errs() << "--field is required with this option\n";
  //     return EXIT_FAILURE;
  //   }

  //   auto field = llzk::Field::getField(cli::smtField());

  //   if (cli::emitSMTLIB()) {
  //     if (failed(emitSMTLIBEncoding(structDef, llvm::outs(),
  //                                         field.name()))) {
  //       llvm::errs() << "failed to emit SMTLIB for struct @"
  //                    << cli::smtStruct() << '\n';
  //       return EXIT_FAILURE;
  //     }
  //     return EXIT_SUCCESS;
  //   }

  //   SymbolicVerifier symbolicVerifier{structDef};
  //   if (failed(symbolicVerifier.buildStore())) {
  //     return EXIT_FAILURE;
  //   }

  //   DeductiveVerifier deductiveVerifier{structDef, field};
  //   deductiveVerifier.addExtraAssertions(
  //       symbolicVerifier.generateAssertions(field));

  //   return EXIT_SUCCESS;
  // }

  // if (cli::dumpStore()) {
  //   mod->walk(
  //       [](llzk::component::StructDefOp structDef) { dumpStore(structDef);
  //       });
  // }

  return EXIT_SUCCESS;
}
