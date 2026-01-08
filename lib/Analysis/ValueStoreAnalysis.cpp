#include "Analysis/ValueStoreAnalysis.h"
#include "Analysis/SignalValueAnalysis.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>

namespace lleq {

using namespace llzk::component;

mlir::ChangeResult ValueStoreLattice::write(llvm::StringRef ref, Symbol sym) {
  initialized = true;
  if (valueStore.contains(ref) && *valueStore.at(ref) == *sym) {
    return mlir::ChangeResult::NoChange;
  } else if (valueStore.contains(ref)) {
    valueStore[ref] = anti_unify(valueStore[ref], sym);
  } else {
    valueStore[ref] = sym;
  }
  return mlir::ChangeResult::Change;
}

mlir::ChangeResult
ValueStoreLattice::join(const mlir::dataflow::AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const ValueStoreLattice *>(&other);
  pool = rhs->pool;
  if (!rhs) {
    llvm::report_fatal_error("cannot join incomparable lattices");
  }
  if (!rhs->initialized) {
    return mlir::ChangeResult::NoChange;
  }
  if (!initialized) {
    valueStore = rhs->valueStore;
    initialized = true;
    return mlir::ChangeResult::Change;
  }

  if (*this == *rhs) {
    return mlir::ChangeResult::NoChange;
  }

  for (auto [key, val] : valueStore) {
    if (!rhs->valueStore.contains(key) &&
        val->kind != impl::SymbolBase::SymbolKind::SK_Uninitialized) {
      valueStore[key] = pool->fresh_unknown();
    } else if (rhs->valueStore.contains(key)) {
      valueStore[key] = anti_unify(val, rhs->valueStore.at(key));
    }
  }

  return mlir::ChangeResult::Change;
}

mlir::LogicalResult
ValueStoreAnalysis::visitOperation(mlir::Operation *op,
                                   const ValueStoreLattice &before,
                                   ValueStoreLattice *after) {
  after->setPool(&pool);
  mlir::ChangeResult result = after->join(before);
  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case<FieldWriteOp>([this, after, &result](FieldWriteOp write) {
        SVALattice *symbol = getOrCreate<SVALattice>(write.getVal());
        symbol->useDefSubscribe(this);
        result |= after->write(write.getFieldName(), symbol->getValue());
      })
      .Case<FieldReadOp>([this, before](FieldReadOp read) {
        SVALattice *lat = getOrCreate<SVALattice>(read.getVal());
        Symbol newSym = before.lookupOrNull(read.getFieldName());
        if (newSym) {
          propagateIfChanged(lat, lat->join(newSym));
        }
      });

  propagateIfChanged(after, result);
  return mlir::success();
}

} // namespace lleq
