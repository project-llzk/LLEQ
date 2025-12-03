#pragma once

#include "Analysis/SymbolExpr.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace lleq {

struct SignalRef {
  mlir::Value value;
  std::vector<Symbol> indices;
};

class SymbolicStore {
  SymbolPool pool;
  mlir::DenseMap<SignalRef, Symbol> store;

  Symbol _build_sym(mlir::Value value);

public:
  void process_block(mlir::Block *block);
  void process(mlir::Operation *op);

  Symbol lookup(SignalRef);
  SignalRef getSignal(mlir::Value value);
};
} // namespace lleq
