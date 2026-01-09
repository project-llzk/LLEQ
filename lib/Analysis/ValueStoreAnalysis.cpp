#include "Analysis/ValueStoreAnalysis.h"
#include "Analysis/SignalValueAnalysis.h"
#include "Analysis/SymbolicStore.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>

#define DEBUG_TYPE "value-store-analysis"

namespace lleq {

using namespace llzk::component;

template <class T>
mlir::ChangeResult ValueStoreLattice::_write_impl(Ref<T> ref, Symbol sym) {
  initialized = true;
  auto &st = store<T>();
  if (st.contains(ref) && *st.at(ref) == *sym) {
    return mlir::ChangeResult::NoChange;
  } else {
    st.write(ref, sym, WriteMode::AntiUnify);
  }
  return mlir::ChangeResult::Change;
}

template mlir::ChangeResult
    ValueStoreLattice::_write_impl<mlir::Value>(ValueRef, Symbol);
template mlir::ChangeResult
    ValueStoreLattice::_write_impl<llvm::StringRef>(SignalRef, Symbol);

mlir::ChangeResult
ValueStoreLattice::join(const mlir::dataflow::AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const ValueStoreLattice *>(&other);
  if (!rhs) {
    llvm::report_fatal_error("cannot join incomparable lattices");
  }
  if (!rhs->initialized) {
    return mlir::ChangeResult::NoChange;
  }
  if (!initialized) {
    *valueStore = *rhs->valueStore;
    *signalStore = *rhs->signalStore;
    initialized = true;
    return mlir::ChangeResult::Change;
  }

  if (*this == *rhs) {
    return mlir::ChangeResult::NoChange;
  }

  valueStore->join_with(*rhs->valueStore);
  signalStore->join_with(*rhs->signalStore);

  return mlir::ChangeResult::Change;
}

mlir::LogicalResult
ValueStoreAnalysis::visitOperation(mlir::Operation *op,
                                   const ValueStoreLattice &before,
                                   ValueStoreLattice *after) {
  after->setPool(&pool);
  LLVM_DEBUG({
    llvm::dbgs() << "Operation: " << *op << "\n";
    llvm::dbgs() << "Before:\n";
    before.print(llvm::dbgs());
    llvm::dbgs() << "Start:\n";
    after->print(llvm::dbgs());
  });
  mlir::ChangeResult result = after->join(before);
  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case<FieldWriteOp>([this, after, &result](FieldWriteOp write) {
        SVALattice *symbol = getOrCreate<SVALattice>(write.getVal());
        symbol->useDefSubscribe(this);
        result |= after->write(write.getFieldName(), symbol->getValue());
      })
      .Case<FieldReadOp>([this, &before](FieldReadOp read) {
        SVALattice *lat = getOrCreate<SVALattice>(read.getVal());
        Symbol newSym = before.lookupOrNull(read.getFieldName());
        if (newSym) {
          propagateIfChanged(lat, lat->join(newSym));
        }
      });
  LLVM_DEBUG({
    llvm::dbgs() << "After:\n";
    after->print(llvm::dbgs());
  });
  propagateIfChanged(after, result);
  return mlir::success();
}

} // namespace lleq
