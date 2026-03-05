/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolicStore.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Analysis/Store.h"
#include "Analysis/SymbolExpr.h"
#include "Analysis/SymbolicStoreAnalysis.h"
#include "Transforms/LLEQIfToIfElse.h"
#include "Transforms/LLEQWhileToFor.h"

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Analysis/AnalysisUtil.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Transforms/LLZKComputeConstrainToProductPass.h>
#include <llzk/Util/ErrorHelper.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LLVM.h>

#define DEBUG_TYPE "symbolic-store"

using namespace lleq;

void SymbolicStore::dump(llvm::raw_ostream &os) const {
  if (!signalStore) {
    os << "(null)\n";
    return;
  }

  for (auto [signal, symbol] : *signalStore) {
    os << signal << ": " << symbol << '\n';
  }
}

mlir::LogicalResult
SymbolicStore::build_store(llzk::component::StructDefOp structDef) {
  component = structDef;

  if (component.getComputeFuncOp() != nullptr) {
    // Make sure we work over a product program
    mlir::SymbolTableCollection tables;
    llzk::LightweightSignalEquivalenceAnalysis equivalence{component};

    if (mlir::failed(llzk::alignStartingAt(component, tables, equivalence))) {
      return mlir::failure();
    }
  }

  auto productFunc = component.getProductFuncOp();
  llzk::ensure(productFunc, "alignment failed");

  if (llvm::failed(transform::transformWhileToFor(productFunc))) {
    llvm::report_fatal_error("while->for conversion failed");
  }
  if (llvm::failed(transform::transformIfToIfElse(productFunc))) {
    llvm::report_fatal_error("default else conversion failed");
  }

  // Pre-populate the liveness analysis so our custom analyses traverse region
  // bodies as they are encountered rather than waiting for the liveness
  // analysis to traverse them.
  if (mlir::failed(
          llzk::dataflow::loadAndRunRequiredAnalyses(solver, productFunc))) {
    return mlir::failure();
  }

  solver.load<lleq::ScalarSymbolAnalysis>(*pool);
  solver.load<lleq::SymbolicStoreAnalysis>(*pool);

  if (mlir::failed(solver.initializeAndRun(productFunc))) {
    return mlir::failure();
  }
  mlir::ProgramPoint *terminator =
      solver.getProgramPointAfter(&*std::prev(productFunc.getBlocks().end()));
  std::tie(signalStore, valueStore) =
      solver.lookupState<StoreLattice>(terminator)->getStores();
  return mlir::success();
}

Symbol SymbolicStore::lookup(mlir::Value value) {
  auto *latticeElem = solver.lookupState<ScalarLattice>(value);
  if (!latticeElem) {
    return pool->uninitialized();
  }
  return latticeElem->getValue();
}
