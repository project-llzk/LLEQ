/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llzk/Dialect/Struct/IR/Ops.h>

#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

namespace lleq {

/// Lower the selected LLZK struct to SMT, build a single-member inequivalence
/// query, and print the resulting SMTLIB script.
llvm::LogicalResult emitSMTLIBEquivalence(
    llzk::component::StructDefOp structDef, llvm::raw_ostream &os,
    const std::optional<std::string> &fieldName = std::nullopt);

} // namespace lleq
