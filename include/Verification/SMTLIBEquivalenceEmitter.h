/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llzk/Dialect/Struct/IR/Ops.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

namespace lleq {

/// Lower the selected LLZK struct to SMT and print its SMTLIB encoding.
llvm::LogicalResult emitSMTLIBEncoding(llzk::component::StructDefOp structDef,
                                       llvm::raw_ostream &os,
                                       llvm::StringRef fieldName);

} // namespace lleq
