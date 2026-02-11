/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolicStore.h"
#include "Transforms/LLEQWhileToFor.h"
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
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LogicalResult.h>

#define BUG_REPORT_URL "https://github.com/Veridise/LLEQ/issues"

static inline void dumpStore(llzk::component::StructDefOp structDef) {
  lleq::SymbolicStore store;
  if (mlir::failed(store.build_store(structDef))) {
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

  mlir::OpBuilder builder(&context);
  auto mod = builder.create<mlir::ModuleOp>(
      mlir::NameLoc::get(builder.getStringAttr("LLEQ")));

  auto parserConfig = mlir::ParserConfig(&context);
  if (llvm::failed(mlir::parseSourceFile(lleq::cli::inputFile(), mod.getBody(),
                                         parserConfig))) {
    llvm::errs() << "Failed to parse " << lleq::cli::inputFile() << '\n';
    return EXIT_FAILURE;
  }

  if (lleq::cli::dumpStore()) {
    mod.walk(
        [](llzk::component::StructDefOp structDef) { dumpStore(structDef); });
  }

  return EXIT_SUCCESS;
}
