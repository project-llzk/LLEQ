/**
 * Copyright 2026 Project LLZK.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/Support/LogicalResult.h>
#include <llzk/Dialect/Struct/IR/Ops.h>

namespace lleq::util {
/// Run product alignment on the module if no @product function is found; errors
/// if product alignment failed
void ensureProductFunc(mlir::ModuleOp module,
                       llzk::component::StructDefOp structDef);
} // namespace lleq::util
