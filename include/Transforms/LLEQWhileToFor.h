/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>

namespace lleq::transform {

struct ForOpInfo {
  std::optional<mlir::Value> lb, ub, step, ivar;
  size_t ivar_index = -1;
  bool success() const {
    return lb.has_value() && ub.has_value() && step.has_value() &&
           ivar.has_value() && ivar_index != static_cast<size_t>(-1);
  }

  llvm::raw_ostream &print(llvm::raw_ostream &os) const;
};

ForOpInfo parseInfo(mlir::scf::WhileOp op);

llvm::LogicalResult transformWhileToFor(llzk::function::FuncDefOp);
llvm::FailureOr<mlir::scf::ForOp>
transformWhileToFor(mlir::scf::WhileOp op, mlir::RewriterBase &rewriter);
} // namespace lleq::transform
