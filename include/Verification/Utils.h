/**
 * Copyright 2026 Project LLZK.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/Support/LogicalResult.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Transforms/LocationSnapshot.h>
#include <stdexcept>

namespace lleq::util {

class UnsupportedConstruct : public std::runtime_error {
  mlir::Location loc;

public:
  UnsupportedConstruct(mlir::Location errorLoc, const std::string &message)
      : std::runtime_error{message}, loc{errorLoc} {}
  mlir::Location getLoc() const { return loc; }
};

/// Run product alignment on the module if no @product function is found; errors
/// if product alignment failed
void ensureProductFunc(mlir::ModuleOp module,
                       llzk::component::StructDefOp structDef);
} // namespace lleq::util
