#pragma once

#include <llvm/Support/LogicalResult.h>
#include <llzk/Dialect/Struct/IR/Ops.h>

namespace lleq {
/// Run product alignment on the module if no @product function is found; errors
/// if product alignment failed
void ensureProductFunc(mlir::ModuleOp module,
                       llzk::component::StructDefOp structDef);
} // namespace lleq
