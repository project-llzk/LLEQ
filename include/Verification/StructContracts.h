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

void applyContractToStruct(llzk::verif::ContractOp contract);
void applyAllContracts(mlir::ModuleOp module);

void projectContracts(llzk::verif::ContractOp contract);

} // namespace lleq
