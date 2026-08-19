/**
 * Copyright 2025 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolicStore.h"
#include "Verification/FixpointVerifier.h"
#include "Verification/StructContracts.h"
#include "Verification/SymbolicVerifier.h"
#include "Verification/WeakestPrecondition.h"
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
#include <llzk/Dialect/Array/Transforms/TransformationPasses.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/InitDialects.h>
#include <llzk/Dialect/POD/Transforms/TransformationPasses.h>
#include <llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Transforms/LLZKTransformationPasses.h>
#include <llzk/Util/ErrorHelper.h>
#include <llzk/Util/Field.h>
#include <llzk/Util/SymbolHelper.h>
#include <llzk/Util/SymbolLookup.h>
#include <llzk/Util/TypeHelper.h>
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
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#define BUG_REPORT_URL "https://github.com/Veridise/LLEQ/issues"

using namespace lleq;
using namespace mlir;

// Copied this from LLZK
FailureOr<llzk::FieldRef> resolveSelectedField(ModuleOp mod,
                                               StringRef fieldName) {
  llzk::FieldSet fields;
  if (!fieldName.empty()) {
    auto fieldLookupResult = llzk::Field::tryGetField(fieldName);
    if (failed(fieldLookupResult)) {
      mod.emitError() << "unknown field \"" << fieldName << "\"";
      return failure();
    }
    fields.insert(fieldLookupResult.value());
  }

  (void)collectFields(mod, fields);

  if (fields.empty()) {
    mod.emitError() << "no prime field specified; could not deduce";
    return failure();
  }

  if (fields.size() > 1) {
    mod.emitError() << "multiple fields unsupported";
    return failure();
  }

  return *(fields.begin());
}

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

  // Start by lowering the struct to eliminate scf.for and arrays
  mlir::PassManager pm(mod->getContext(),
                       mlir::PassManager::getAnyOpAnchorName(),
                       mlir::PassManager::Nesting::Implicit);
  pm.enableVerifier(false);
  pm.addPass(llzk::pod::createPodToScalarPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(llzk::createWhileToForPass());
  if (cli::flattenStruct()) {
    pm.addPass(std::move(llzk::polymorphic::createFlatteningPass()));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(std::move(llzk::array::createArrayToScalarPass()));
  }

  llzk::ensure(llvm::succeeded(pm.run(*mod)),
               "failed to prepare module for verification");

  // Apply all contracts?
  mod->walk([](llzk::verif::ContractOp contract) {
    applyContractToStruct(contract);
  });

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
  case cli::SubCmd::WeakestPrecondition: {
    auto field = resolveSelectedField(*mod, cli::fieldName());
    if (failed(field)) {
      // already emits an error
      return EXIT_FAILURE;
    }

    WeakestPreconditionAnalysis analysis{structDef, *field};
    SymbolicVerifier symbolicStore{structDef};
    if (failed(symbolicStore.buildStore())) {
      llvm::errs() << "Failed to build symbolic store\n";
      return EXIT_FAILURE;
    }

    for (auto memberDef : structDef.getMemberDefs()) {
      if (symbolicStore.areEquivalent(memberDef.getSymName())) {
        analysis.addEquivalentMember(memberDef);
      }
    }

    analysis.emit(llvm::outs());

    return EXIT_SUCCESS;
  }
  case cli::SubCmd::Verify:
  case cli::SubCmd::DumpSmt: {
    auto field = resolveSelectedField(*mod, cli::fieldName());
    if (failed(field)) {
      // already emits an error
      return EXIT_FAILURE;
    }
    FixpointVerifier verifier{structDef, *field};
    llzk::ensure(succeeded(verifier.init(cli::enableStore())),
                 "failed to generate SMT encoding");

    if (cli::subCmd() == cli::SubCmd::DumpSmt) {
      verifier.dumpSmt(llvm::outs());
      return EXIT_SUCCESS;
    }

    while (verifier.runIteration() == mlir::ChangeResult::Change)
      ;
    verifier.report(llvm::outs());
    return EXIT_SUCCESS;
  }
  }

  return EXIT_SUCCESS;
}
