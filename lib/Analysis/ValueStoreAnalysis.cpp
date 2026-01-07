#include "Analysis/ValueStoreAnalysis.h"
#include "Analysis/SignalValueAnalysis.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>

namespace lleq {

using namespace llzk::component;

mlir::LogicalResult
ValueStoreAnalysis::visitOperation(mlir::Operation *op,
                                   const ValueStoreLattice &before,
                                   ValueStoreLattice *after) {
  llvm::dbgs() << "VSA visiting " << *op << "\n";
  mlir::ChangeResult result = after->join(before);
  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case<FieldWriteOp>([this, after, &result](FieldWriteOp write) {
        mlir::Value src = write.getVal();
        SVALattice *symbol = getOrCreate<SVALattice>(write.getVal());
        result |= after->write(write.getFieldName(), symbol->getValue());
      })
      .Case<FieldReadOp>([this, before](FieldReadOp read) {
        llvm::dbgs() << "Hit a read: " << read << "\n";
        SVALattice *lat = getOrCreate<SVALattice>(read.getVal());
        Symbol newSym = before.lookupOrNull(read.getFieldName());
        if (newSym) {
          llvm::dbgs() << "Propagating " << lat->getValue() << " to " << newSym
                       << "\n";
          auto changed = lat->join(newSym);
          llvm::dbgs() << "new lat: " << lat->getValue()
                       << ", changed: " << static_cast<int>(changed) << "\n";
          propagateIfChanged(lat, lat->join(newSym));
        }
      })

      ;
  return mlir::success();
}

} // namespace lleq
