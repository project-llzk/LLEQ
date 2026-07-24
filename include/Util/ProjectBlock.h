#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Operation.h>

namespace lleq::util {

class BlockProjector {
  using Projection = int;

  // For each operation in the worklist, which projections its being considered
  // a part of
  llvm::DenseMap<mlir::Operation *, llvm::DenseSet<Projection>> worklist;

  // The set of operations in each projection
  llvm::DenseMap<Projection, llvm::DenseSet<mlir::Operation *>> projections;

public:
  void addSeed(llvm::ArrayRef<mlir::Operation *> seed, Projection current);
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
