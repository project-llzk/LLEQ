/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/Support/LogicalResult.h>
#include <llzk/Dialect/Function/IR/Ops.h>

namespace lleq::transform {

llvm::LogicalResult transformIfToIfElse(llzk::function::FuncDefOp);

}
