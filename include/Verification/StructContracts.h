/**
 * Copyright 2026 Project LLZK.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Dialect/Verif/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>

namespace lleq {

/// Lower require/ensure clauses in the `contract` to assertions in its
/// target
void applyContractToStruct(llzk::verif::ContractOp contract);

/// Apply all contracts found in `module` to their targets
void applyAllContracts(mlir::ModuleOp module);

} // namespace lleq
