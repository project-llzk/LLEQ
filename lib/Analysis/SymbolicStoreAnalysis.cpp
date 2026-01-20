#include "Analysis/SymbolicStoreAnalysis.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/ErrorHelper.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/IR/Value.h>

#define DEBUG_TYPE "symbolic-store-analysis"

namespace lleq {

using namespace llzk::component;
using namespace llzk::array;

void StoreLattice::print(llvm::raw_ostream &os) const {
  LLVM_DEBUG({
    if (!initialized) {
      os << "(uninit)\n";
      return;
    }
    if (valueStore == nullptr || signalStore == nullptr) {
      os << "(null)\n";
      return;
    }
    os << "--\n";
    if (valueStore->size() == 0) {
      os << "(empty)\n";
    }
    for (auto [key, val] : *valueStore) {
      os << key << ": " << val << "\n";
    }
    os << "--\n";
    if (signalStore->size() == 0) {
      os << "(empty)\n";
    }
  });
  for (auto [key, val] : *signalStore) {
    os << key << ": " << static_cast<Symbol>(val) << "\n";
  }
  // os << "--\n";
}

Symbol SymbolicStoreAnalysis::getBoundSymbol(mlir::Value value) {
  ScalarLattice *lattice = getOrCreate<ScalarLattice>(value);
  lattice->useDefSubscribe(this);
  return lattice->getValue();
}

template <class T>
mlir::ChangeResult StoreLattice::_write_impl(IndexedLocation<T> ref,
                                             Symbol sym) {
  initialized = true;
  auto &st = store<T>();
  if (st.contains(ref) && *st.at(ref) == *sym) {
    return mlir::ChangeResult::NoChange;
  }
  // TODO: is `Clobber` the correct mode to use here?
  st.write(ref, sym, WriteMode::HavocAliases);
  return mlir::ChangeResult::Change;
}

template mlir::ChangeResult StoreLattice::_write_impl<mlir::Value>(ValueRef,
                                                                   Symbol);
template mlir::ChangeResult
    StoreLattice::_write_impl<llvm::StringRef>(SignalRef, Symbol);

mlir::ChangeResult
StoreLattice::join(const mlir::dataflow::AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const StoreLattice *>(&other);
  if (!rhs) {
    llvm::report_fatal_error("cannot join incomparable lattices");
  }
  if (!rhs->initialized) {
    return mlir::ChangeResult::NoChange;
  }
  if (!initialized) {
    setPool(rhs->pool);
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

mlir::LogicalResult SymbolicStoreAnalysis::visitOperation(
    mlir::Operation *op, const StoreLattice &before, StoreLattice *after) {
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
      .Case<FieldWriteOp>([this, after, &result, &before](FieldWriteOp write) {
        if (llvm::dyn_cast<llzk::array::ArrayType>(
                write.getOperandTypes()[1])) {
          // Its an array so copy from valueStore to signalStore
          if (!before.initialized) {
            // This is weird but there's nothing to copy
            // Hopefully we'll visit this state again when there is something
            return;
          }
          for (auto [ref, sym] : *before.valueStore) {
            if (ref.name == write.getVal()) {
              // `after->write` will correctly clobber any entries signalStore
              // already has for this signal
              result |= after->write(
                  SignalRef{write.getFieldName(), ref.indices}, sym);
            }
          }
          return;
        }
        // Otherwise, its a scalar, so lookup the symbol from
        // SignalValueAnalysis and write it to the store
        Symbol written = getBoundSymbol(write.getVal());
        result |= after->write(write.getFieldName(), written);
      })
      .Case<FieldReadOp>([this, &before, &result, after](FieldReadOp read) {
        if (llvm::dyn_cast<llzk::array::ArrayType>(read.getType())) {
          // Its an array so copy from signalStore to valueStore
          for (auto [ref, sym] : *before.signalStore) {
            if (ref.name == read.getFieldName()) {
              // Technically, `after->write` attempts to clobber here, but since
              // `read.getVal()` should be a fresh SSA value, it doesn't matter
              result |= after->write(ValueRef{read.getVal(), ref.indices}, sym);
            }
          }
          return;
        }
        // Otherwise, its a scalar, so inject into SignalValueAnalysis
        ScalarLattice *lat = getOrCreate<ScalarLattice>(read.getVal());
        Symbol newSym = before.lookupOrNull(read.getFieldName());
        if (newSym) {
          propagateIfChanged(lat, lat->join(newSym));
        }
      })
      .Case<WriteArrayOp>([this, after, &result](WriteArrayOp write) {
        Symbol rval = getBoundSymbol(write.getRvalue());
        llvm::SmallVector<Symbol> indices;
        for (auto idx : write.getIndices()) {
          indices.push_back(getBoundSymbol(idx));
        }
        // `after->write` will automatically clobber
        result |= after->write(ValueRef{write.getArrRef(), indices}, rval);
      })
      .Case<ReadArrayOp>([this, &before, after](ReadArrayOp read) {
        // If the array is a block arg, there's nothing to propagate
        if (llvm::isa<mlir::BlockArgument>(read.getArrRef())) {
          return;
        }
        ScalarLattice *lat = getOrCreate<ScalarLattice>(read.getResult());
        llvm::SmallVector<Symbol> indices;
        for (auto idx : read.getIndices()) {
          indices.push_back(getBoundSymbol(idx));
        }
        Symbol newSym = after->lookup(ValueRef{read.getArrRef(), indices});
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
