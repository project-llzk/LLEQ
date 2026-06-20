#pragma once

#include <mlir/Dialect/SCF/IR/SCF.h>

namespace lleq {

void conjecturePredicates(mlir::scf::ForOp loop);

}
