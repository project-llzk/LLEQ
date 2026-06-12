/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolicStoreAnalysis.h"
#include "Analysis/ScalarSymbolAnalysis.h"
#include "Analysis/Store.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Constrain/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Util/ErrorHelper.h>
#include <llzk/Util/TypeHelper.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

#define DEBUG_TYPE "symbolic-store-analysis"

namespace lleq {

using namespace llzk::component;
using namespace llzk::array;
using namespace llzk::constrain;

void StoreLattice::print(llvm::raw_ostream &os) const {
  if (!initialized) {
    os << "(uninit)\n";
    return;
  }
  if (valueStore == nullptr || signalStore == nullptr) {
    os << "(null)\n";
    return;
  }
  if (valueStore->size() == 0) {
    os << "(empty)\n";
  }
  for (auto [key, val] : *valueStore) {
    os << key << ": " << val << '\n';
  }
  if (signalStore->size() == 0) {
    os << "(empty)\n";
  }
  for (auto [key, val] : *signalStore) {
    os << key << ": " << static_cast<Symbol>(val) << "\n";
  }
}

Symbol SymbolicStoreAnalysis::getBoundSymbol(mlir::Value value) {
  ScalarLattice *lattice = getOrCreate<ScalarLattice>(value);
  lattice->useDefSubscribe(this);
  auto latticeElem = lattice->getValue();
  latticeElem.initPool(&pool);
  return latticeElem;
}

template <class T>
mlir::ChangeResult StoreLattice::_write_impl(IndexedLocation<T> ref,
                                             Symbol sym) {
  initialized = true;
  auto &st = store<T>();
  if (st.contains(ref) && *st.at(ref) == *sym) {
    return mlir::ChangeResult::NoChange;
  }
  st.write(ref, sym, WriteMode::HavocAliases);
  return mlir::ChangeResult::Change;
}

template mlir::ChangeResult StoreLattice::_write_impl<mlir::Value>(IndexedValue,
                                                                   Symbol);
template mlir::ChangeResult StoreLattice::_write_impl<Signal>(IndexedSignal,
                                                              Symbol);

mlir::ChangeResult
StoreLattice::copy(const mlir::dataflow::AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const StoreLattice *>(&other);
  if (!rhs) {
    llvm::report_fatal_error("cannot copy incomparable lattices");
  }
  if (!rhs->initialized || *this == *rhs) {
    return mlir::ChangeResult::NoChange;
  }

  initPool(rhs->pool);
  llzk::ensure(rhs->valueStore && rhs->signalStore, "stores not initialized");
  *valueStore = *rhs->valueStore;
  *signalStore = *rhs->signalStore;
  initialized = true;
  return mlir::ChangeResult::Change;
}

mlir::ChangeResult
StoreLattice::join(const mlir::dataflow::AbstractDenseLattice &other) {

  const auto *rhs = dynamic_cast<const StoreLattice *>(&other);
  if (!rhs) {
    llvm::report_fatal_error("cannot join incomparable lattices");
  }
  if (!rhs->initialized) {
    if (initialized) {
      valueStore->clear();
      signalStore->clear();
      return mlir::ChangeResult::Change;
    }
    return mlir::ChangeResult::NoChange;
  }
  if (!initialized) {
    initPool(rhs->pool);
    llzk::ensure(rhs->valueStore && rhs->signalStore, "stores not initialized");
    *valueStore = *rhs->valueStore;
    *signalStore = *rhs->signalStore;
    initialized = true;
    return mlir::ChangeResult::Change;
  }

  if (*this == *rhs) {
    return mlir::ChangeResult::NoChange;
  }

  auto anchor = getAnchor().dyn_cast<mlir::ProgramPoint *>();
  AUTag tag = anchor;
  if (!anchor->isBlockStart()) {
    tag = anchor->getPrevOp();
  }

  valueStore->join_with(*rhs->valueStore, tag);
  signalStore->join_with(*rhs->signalStore, tag);

  return mlir::ChangeResult::Change;
}

mlir::LogicalResult SymbolicStoreAnalysis::visitOperation(
    mlir::Operation *op, const StoreLattice &_before, StoreLattice *after) {
  after->initPool(&pool);

  // This is kind of a hack, but if `op` is the first op in a basic block whose
  // parent op has region control flow, try to manually inherit the lattice from
  // the parent (since at initialization time it won't be present)
  const auto &before =
      (op->getPrevNode() == nullptr &&
       llvm::isa<mlir::RegionBranchOpInterface>(op->getParentOp()))
          ? *getOrCreate<StoreLattice>(getProgramPointBefore(op->getParentOp()))
          : _before;

  LLVM_DEBUG({
    llvm::dbgs() << '\n';
    before.print(llvm::dbgs());
    llvm::dbgs() << "Operation: " << *op << '\n';
  });

  mlir::ChangeResult result = after->copy(before);
  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case<mlir::scf::YieldOp>([this, after](mlir::scf::YieldOp yieldOp) {
        auto afterState = getOrCreate<StoreLattice>(
            getProgramPointAfter(yieldOp->getParentOp()));
        propagateIfChanged(afterState, afterState->join(*after));
      })
      .Case<MemberWriteOp>([this, after, &result,
                            &before](MemberWriteOp write) {
        if (llvm::isa<llzk::array::ArrayType>(write.getVal().getType())) {
          // It's an array so copy from valueStore to signalStore
          if (!before.initialized) {
            // This is weird but there's nothing to copy
            // Hopefully we'll visit this state again when there is
            // something
            return;
          }
          for (auto [ref, sym] : *before.valueStore) {
            if (ref.name == write.getVal()) {
              // `after->write` will correctly clobber any entries
              // signalStore already has for this signal
              result |=
                  after->write(IndexedSignal{Signal{Signal::Source::Witness,
                                                    write.getMemberName()},
                                             ref.index},
                               sym);
            }
          }
          return;
        }
        // Otherwise, it's a scalar, so lookup the symbol from
        // SignalValueAnalysis and write it to the store
        Symbol written = getBoundSymbol(write.getVal());
        result |= after->write(
            Signal{Signal::Source::Witness, write.getMemberName()}, written);
      })
      .Case<MemberReadOp>([this, &before, &result, after](MemberReadOp read) {
        if (llvm::isa<llzk::array::ArrayType>(read.getType())) {
          // Its an array so copy from signalStore to valueStore
          for (auto [ref, sym] : *before.signalStore) {
            if (ref.name.name == read.getMemberName() &&
                // Make sure we don't accidentally read a value @compute wrote
                // to this field while in @constrain
                sourceMatchesOp(read, ref.name.source)) {
              // Technically, `after->write` attempts to clobber here, but since
              // `read.getVal()` should be a fresh SSA value, it doesn't matter
              result |=
                  after->write(IndexedValue{read.getVal(), ref.index}, sym);
            }
          }
          return;
        }
        // Otherwise, its a scalar, so inject into SignalValueAnalysis
        ScalarLattice *lat = getOrCreate<ScalarLattice>(read.getVal());
        Symbol newSym = before.lookupOrNull(
            Signal{isWitnessOp(read) ? Signal::Source::Witness
                                     : Signal::Source::Constraint,
                   read.getMemberName()});
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
        result |= after->write(IndexedValue{write.getArrRef(), indices}, rval);
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

        // If reading from an array during a constraint op, don't
        // default-initialize with unknowns because a constrain may later
        // initialize this value
        Symbol newSym =
            isWitnessOp(read)
                ? after->lookup(IndexedValue{read.getArrRef(), indices})
                : after->lookupOrNull(IndexedValue{read.getArrRef(), indices});
        if (newSym) {
          propagateIfChanged(lat, lat->join(newSym));
        }
      })
      .Case<EmitEqualityOp>([this, after, &result](EmitEqualityOp eq) {
        auto getIndexedLoc =
            [this](mlir::Value val) -> llvm::FailureOr<IndexedSignal> {
          if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
          }
          // If the value immediately comes from a constraint `struct.readm`
          if (auto read =
                  llvm::dyn_cast_or_null<MemberReadOp>(val.getDefiningOp())) {
            // No sense constraining a member of a subcomponent
            if (isSubcmpRead(read)) {
              return {};
            }
            return {{{Signal::Source::Constraint, read.getMemberName()}, {}}};
          }
          if (auto arrRead =
                  llvm::dyn_cast_or_null<ReadArrayOp>(val.getDefiningOp())) {
            if (auto arr = llvm::dyn_cast_or_null<MemberReadOp>(
                    arrRead.getArrRef().getDefiningOp())) {
              llvm::SmallVector<Symbol> indices;
              for (auto idx : arrRead.getIndices()) {
                indices.push_back(getBoundSymbol(idx));
              }
              return {{{Signal::Source::Constraint, arr.getMemberName()},
                       std::move(indices)}};
            }
          }
          return {};
        };

        auto leftLoc = getIndexedLoc(eq.getLhs());
        auto rightLoc = getIndexedLoc(eq.getRhs());
        if (llvm::succeeded(leftLoc)) {
          auto rightSym = getBoundSymbol(eq.getRhs());
          result |= after->write(*leftLoc, rightSym);
          // Update the ScalarSymbolAnalysis with the value for leftLoc
          auto lat = getOrCreate<ScalarLattice>(eq.getLhs());
          propagateIfChanged(lat, lat->join(rightSym));
        } else if (llvm::succeeded(rightLoc)) {
          auto leftSym = getBoundSymbol(eq.getLhs());
          result |= after->write(*rightLoc, leftSym);
          // Update the ScalarSymbolAnalysis with the value for rightLoc
          auto lat = getOrCreate<ScalarLattice>(eq.getRhs());
          propagateIfChanged(lat, lat->join(leftSym));
        }
      })
      .Case<llzk::function::CallOp>([this](llzk::function::CallOp call) {
        if (!(call.calleeIsCompute() || call.calleeIsConstrain())) {
          return;
        }
        auto subcmp = (call.calleeIsCompute() ? call.getResult(0)
                                              : call.getArgOperands().front());
        llvm::dbgs() << "In call: " << call << "\n\t";
        llvm::dbgs() << "Subcomponent symbol is: "
                     << getOrCreate<ScalarLattice>(subcmp);
      });
  LLVM_DEBUG({
    after->print(llvm::dbgs());
    llvm::dbgs() << '\n';
  });
  propagateIfChanged(after, result);
  return mlir::success();
}

} // namespace lleq
