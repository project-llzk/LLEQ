/**
 * Copyright 2025 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/SymbolExpr.h"
#include "Store.h"

#include <cassert>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/PointerUnion.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace lleq {
/// @brief Represents a mapping between circuit signals and symbolic
/// expressions. Each entry in the store is keyed by both the signal, which is
/// assumed to be a (possibly multidimensional) array, and a vector of symbolic
/// indices into the array, one per dimension. This is later used to statically
/// prove equivalence between pairs of witness/constraint signals.
class SymbolicStore {
  std::unique_ptr<SymbolPool> pool = std::make_unique<SymbolPool>();
  mlir::DataFlowSolver solver{mlir::DataFlowConfig{}.setInterprocedural(false)};

  llzk::component::StructDefOp component;
  SignalStore *signalStore;
  ValueStore *valueStore;

public:
  SymbolicStore() {}

  /// @brief Build a store from a given circuit component (struct)
  /// @param structDef
  mlir::LogicalResult buildStore(llzk::component::StructDefOp structDef);

  /// @brief Generate a symbolic expression corresponding to an MLIR SSA value,
  /// possibly looking up values in the store to do so
  /// @param value
  Symbol lookup(mlir::Value value);

  /// @brief Look up the symbolic expression corresponding to a signal in the
  /// store at a given index, or return Unknown if the signal is untracked at
  /// that index
  /// @param signal
  /// @param indices
  Symbol lookup(Signal signal, llvm::ArrayRef<Symbol> indices = {});

  /// @brief Return a vector containing all symbolic indices written to for the
  /// given signal. If the signal is untracked, returns an empty vector.
  /// @param signal
  llvm::DenseSet<StoreIndex> getWrittenIndices(Signal signal);

  /// @brief Pretty-print the contents of the store
  /// @param os
  void dump(llvm::raw_ostream &os) const;
};

} // namespace lleq
