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
using namespace llvm;

void SymbolicStore::dump(raw_ostream &os) const {
  if (!signalStore) {
    os << "(null)\n";
    return;
  }

  for (auto [signal, symbol] : *signalStore) {
    os << signal << ": " << symbol << '\n';
  }
}

mlir::LogicalResult
SymbolicStore::buildStore(llzk::component::StructDefOp structDef) {
  component = structDef;

  if (component.getProductFuncOp() == nullptr) {
    // Make sure we work over a product program
    mlir::SymbolTableCollection tables;
    llzk::LightweightSignalEquivalenceAnalysis equivalence{component};

    llzk::ProductAligner aligner{tables, equivalence};
    if (!aligner.alignFuncs(component, component.getComputeFuncOp(),
                            component.getConstrainFuncOp())) {
      return failure();
    }
  }

  auto productFunc = component.getProductFuncOp();
  llzk::ensure(productFunc, "alignment failed");

  if (failed(transform::transformWhileToFor(productFunc))) {
    report_fatal_error("while->for conversion failed");
  }
  if (failed(transform::transformIfToIfElse(productFunc))) {
    report_fatal_error("default else conversion failed");
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

Symbol SymbolicStore::lookup(Signal signal, ArrayRef<Symbol> indices) {
  llzk::ensure(signalStore != nullptr, "lookup() called before buildStore()");

  // It's so silly that IndexedLocation<T> has to own the vector of indices so I
  // have to allocate a new vector here
  auto indexedSignal = IndexedSignal{signal, SmallVector<Symbol>{indices}};
  if (auto it = signalStore->find(indexedSignal); it != signalStore->end()) {
    return it->second;
  }
  return pool->fresh_unknown();
}

DenseSet<StoreIndex> SymbolicStore::getWrittenIndices(Signal signal) {
  llzk::ensure(signalStore != nullptr,
               "getWrittenIndices() called before buildStore()");

  DenseSet<StoreIndex> indices;
  for (auto [ref, value] : *signalStore) {
    if (ref.name == signal) {
      indices.insert(ref.index);
    }
  }
  return indices;
}
