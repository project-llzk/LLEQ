#include <iostream>
#include <mlir/Support/LogicalResult.h>

#include "Analysis/SymbolExpr.h"

mlir::LogicalResult test() { return mlir::failure(); }

int main(int argc, char **argv) {
  lleq::SymbolPool pool;
  lleq::Symbol expr = pool.arith(
      pool.constant(mlir::APInt{8, 250}),
      pool.arith(pool.fresh_unknown(), pool.templ_param("N"), '+'), '*');
  lleq::Symbol expr2 = pool.index(mlir::Value::getFromOpaquePointer(nullptr),
                                  {expr, pool.templ_param("M")});

  if (mlir::failed(test())) {
    std::cout << "Good job: " << expr2 << "\n";
  }
  return 0;
}
