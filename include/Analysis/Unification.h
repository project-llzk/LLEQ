/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analysis/SymbolExpr.h"
#include <llvm/ADT/DenseMap.h>
#include <mlir/Support/LLVM.h>

namespace lleq {

using Substitutions = llvm::SmallVector<std::pair<unsigned, Symbol>>;

// Attempt to unify a single pair of symbols, updating `s` with the
// substitutions. Returns `failure()` if the symbols cannot be unified
mlir::LogicalResult unify(Symbol a, Symbol b, Substitutions &s);

// Attempt to simultaneously unify multiple pairs of symbols, updating `s` with
// the substitutions. Returns `failure()` if any pair cannot be unified, or if
// the substitutions disagree
mlir::LogicalResult unify_all(llvm::ArrayRef<Symbol> as,
                              llvm::ArrayRef<Symbol> bs, Substitutions &s);

// Substitute a set of unknowns in a symbol
Symbol substitute(Symbol original, const Substitutions &m);

} // namespace lleq
