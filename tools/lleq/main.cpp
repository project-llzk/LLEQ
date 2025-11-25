#include <iostream>
#include <mlir/Support/LogicalResult.h>

#include "Analysis/SymbolExpr.h"

mlir::LogicalResult test() { return mlir::failure(); }

int main(int argc, char **argv) {

  lleq::SymbolPool pool;
  lleq::Symbol expr = pool.arith(
      pool.fresh_unknown(),
      pool.arith(pool.fresh_unknown(), pool.templ_param("N"), '+'), '*');

  if (mlir::failed(test())) {
    std::cout << "Good job: " << expr << "\n";
  }
  return 0;
}
