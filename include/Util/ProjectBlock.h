/**
 * Copyright 2026 Project LLZK.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Operation.h>

namespace lleq::util {

/// Separate a basic block into sequences of ops ("projections") required to
/// compute one of several specified result operations ("seeds").
///
/// This class traverses the use/def chains backwards starting at each seed and
/// tracks which projection each op should belong in, while avoiding recomputing
/// upwards-closed sets for already-visited operations.
class BlockProjector {
  // Used as a tag to identify projections
  using Projection = int;

  // For each operation in the worklist, which projections its being considered
  // a part of
  llvm::DenseMap<mlir::Operation *, llvm::DenseSet<Projection>> worklist;

  // The set of operations in each projection
  llvm::DenseMap<Projection, llvm::DenseSet<mlir::Operation *>> projections;

public:
  // "seed" a projection with the given set of ops
  void addSeed(llvm::ArrayRef<mlir::Operation *> seed, Projection current);
  // Perform traversals until fixpoint, populating `projections`
  void run();

  void dumpProjections() {
    for (auto [_, ops] : projections) {
      llvm::dbgs() << "---\n";
      for (auto op : ops) {
        op->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
    }
  }
};

} // namespace lleq::util
