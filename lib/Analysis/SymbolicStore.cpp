/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Analysis/SymbolicStore.h"
#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"

#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#define DEBUG_TYPE "symbolic-store"

using namespace lleq;

void SymbolicStore::dump(llvm::raw_ostream &os) const {
  os << "Signal store: (" << signalStore.size() << " entries)\n";
  for (const auto &[sig, sym] : signalStore) {
    os << "* " << sig.name;
    for (auto idx : sig.indices) {
      os << "[" << idx << "]";
    }
    os << " -> " << sym << "\n";
  }

  os << "Value store: (" << valueStore.size() << " entries)\n";
  for (const auto &[sig, sym] : valueStore) {
    os << "+ " << pool->getNameForValue(sig.name);
    for (auto idx : sig.indices) {
      os << "[" << idx << "]";
    }
    os << " -> " << sym << "\n";
  }
}

void SymbolicStore::build_store(llzk::component::StructDefOp structDef) {
  auto computeFunc = structDef.getComputeFuncOp();
  // TODO: this doesn't work if there's any control flow in the blocks lol
  for (auto &block : computeFunc.getFunctionBody().getBlocks()) {
    process_block(&block);
  }
}

Symbol SymbolicStore::lookup(mlir::Value value) {
  if (llvm::isa<mlir::TypedValue<llzk::array::ArrayType>>(value)) {
    llvm::report_fatal_error("cannot generate a symbol for a non-scalar value");
  }

  // Input signal
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    return pool->index(value, {});
  }

  mlir::Operation *definingOp = value.getDefiningOp();
  return llvm::TypeSwitch<mlir::Operation *, Symbol>(definingOp)
      // A template parameter
      .Case<llzk::polymorphic::ConstReadOp>(
          [this](llzk::polymorphic::ConstReadOp templRead) {
            return pool->templ_param(templRead.getConstName());
          })
      // Read from a value array into a value scalar
      .Case<llzk::array::ReadArrayOp>([this](llzk::array::ReadArrayOp arrRead) {
        auto array = arrRead.getArrRef();
        // Get the indices as symbols
        llvm::SmallVector<Symbol> indices;
        for (auto index : arrRead.getIndices()) {
          indices.push_back(lookup(index));
        }
        auto ref = ValueRef{array, indices};

        // Is the array an input?
        if (llvm::dyn_cast<mlir::BlockArgument>(array)) {
          return pool->index(array, indices);
        }

        // If the index hasn't been written to before, mark it
        if (!valueStore.contains(ref)) {
          valueStore[ref] = pool->fresh_unknown();
        }
        return valueStore[ref];
      })
      .Case<llzk::felt::FeltBinaryOpInterface>(
          [this](llzk::felt::FeltBinaryOpInterface binop) {
            return pool->func_call(
                binop->getName().getStringRef(),
                {lookup(binop.getLhs()), lookup(binop.getRhs())});
          })
      .Case<llzk::felt::FeltConstantOp>([this](llzk::felt::FeltConstantOp op) {
        return pool->constant(op.getValue());
      })
      .Case<mlir::arith::ConstantOp>([this](mlir::arith::ConstantOp op) {
        return pool->constant(
            llvm::dyn_cast<mlir::IntegerAttr>(op.getValue()).getValue());
      })
      .Case<llzk::component::FieldReadOp>(
          [this](llzk::component::FieldReadOp read) {
            // Reading from a scalar field should copy from the symbol store
            SignalRef ref{.name = read.getFieldName(), .indices = {}};
            if (!signalStore.contains(ref)) {
              signalStore[ref] = pool->fresh_unknown();
            }
            return signalStore[ref];
          })
      .Case<llzk::function::CallOp>([this](llzk::function::CallOp call) {
        llvm::SmallVector<Symbol> args;
        for (auto arg : call.getArgOperands()) {
          args.push_back(lookup(arg));
        }
        return pool->func_call(call.getCallee().getLeafReference(), args);
      })
      .Default([this](mlir::Operation *op) {
        if (op->getNumResults() != 1) {
          return pool->fresh_unknown();
        }
        // Treat it as an uninterpreted function
        llvm::SmallVector<Symbol> args;
        for (auto arg : op->getOperands()) {
          args.push_back(lookup(arg));
        }
        return pool->func_call(op->getName().getStringRef(), args);
      });
}

void SymbolicStore::process_operation(mlir::Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Processing op: " << *op << "\n");
  // TODO: handle control flow ops
  // TODO: handle constraint ops

  llvm::TypeSwitch<mlir::Operation *>(op)
      .Case<llzk::component::FieldReadOp>(
          [this](llzk::component::FieldReadOp read) {
            // Copy every written index from signalStore to valueStore
            for (const auto &[signalRef, value] : signalStore) {
              if (signalRef.name == read.getFieldName()) {
                valueStore[ValueRef{.name = read.getResult(),
                                    .indices = signalRef.indices}] = value;
              }
            }
          })
      .Case<llzk::component::FieldWriteOp>(
          [this](llzk::component::FieldWriteOp write) {
            // Copy every written index from valueStore to signalStore
            for (const auto &[valueRef, value] : valueStore) {
              if (valueRef.name == write.getVal()) {
                signalStore[SignalRef{.name = write.getFieldName(),
                                      .indices = valueRef.indices}] = value;
              }
            }
          })
      .Case<llzk::array::WriteArrayOp>([this](llzk::array::WriteArrayOp write) {
        // Havoc every possible array index that could be clobbered
        auto array = write.getArrRef();
        for (const auto &[valueRef, value] : valueStore) {
          // TODO: actually check if the index could be clobbered
          if (valueRef.name == array) {
            valueStore[valueRef] = pool->fresh_unknown();
          }
        }

        // Write the new symbol
        llvm::SmallVector<Symbol> indices;
        for (auto index : write.getIndices()) {
          indices.push_back(lookup(index));
        }

        valueStore[ValueRef{array, indices}] = lookup(write.getRvalue());
      })
      .Default([](auto) {});
}

void SymbolicStore::process_block(mlir::Block *block) {
  for (auto &op : block->getOperations()) {
    process_operation(&op);
  }
}

template <class T>
auto _group_idx(const llvm::DenseMap<Ref<T>, Symbol> &store) {
  llvm::DenseMap<T, llvm::SmallVector<llvm::SmallVector<Symbol>>> grouped;

  for (auto [k, _] : store) {
    grouped[k.name].push_back(k.indices);
  }

  return grouped;
}

template <class T>
auto _join_stores(const llvm::DenseMap<Ref<T>, Symbol> &a,
                  const llvm::DenseMap<Ref<T>, Symbol> &b) {
  // For every pair of entries (arr, phi_1) :- x; (arr, phi_2) :- y from `a` and
  // `b`, add an entry (arr, AU(phi_1, phi_2)) :- AU(x, y) to the result, where
  // AU represents antiunification
  llvm::DenseMap<Ref<T>, Symbol> result;
  auto groupedA = _group_idx(a);
  auto groupedB = _group_idx(b);

  for (auto [arrA, idxAs] : groupedA) {
    if (!groupedB.contains(arrA)) {
      // If `arrA` is only written to in `a`, then just copy all the entries
      for (auto idxA : idxAs) {
        Ref<T> ref{.name = arrA, .indices = idxA};
        result[ref] = a.at(ref);
      }
      continue;
    }

    // If `arrA` is written to in both `a` and `b`, pairwise anti-unify the
    // written indices
    auto idxBs = groupedB[arrA];
    for (auto idxA : idxAs) {
      for (auto idxB : idxBs) {
        anti_unify_all_inplace(idxA, idxB);
        Ref<T> ref{arrA, idxA};
        // and anti-unify all values written to these anti-unified indices
        if (result.contains(ref)) {
          anti_unify_inplace(result[ref], a.at({arrA, idxA}));
          anti_unify_inplace(result[ref], b.at({arrA, idxB}));
        } else {
          result[ref] = anti_unify(a.at({arrA, idxA}), b.at({arrA, idxB}));
        }
      }
    }
  }

  // Finally, just copy over all entries only written to in `b`
  for (auto [arrB, idxBs] : groupedB) {
    if (!groupedA.contains(arrB)) {
      for (auto idxB : idxBs) {
        Ref<T> ref{arrB, idxB};
        result[ref] = b.at(ref);
      }
    }
  }

  return result;
}

SymbolicStore SymbolicStore::join(const SymbolicStore &a,
                                  const SymbolicStore &b) {
  SymbolicStore result;
  result.signalStore = _join_stores(a.signalStore, b.signalStore);
  result.valueStore = _join_stores(a.valueStore, b.valueStore);
  return result;
}
