/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/Support/LogicalResult.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>

namespace lleq {
llvm::LogicalResult ensureProductFunc(mlir::ModuleOp module,
                                      llzk::component::StructDefOp structDef);
} // namespace lleq
