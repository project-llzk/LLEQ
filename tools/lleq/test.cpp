#include "Analysis/SymbolExpr.h"
#include <iostream>

int main() {
  lleq::SymbolPool pool;
  auto x = pool.fresh_unknown();
  auto y = pool.fresh_unknown();
  auto sum1 = pool.func_call("add", {x, y});
  auto sum2 = pool.func_call("add", {x, y});

  if (*sum1 == *sum2) {
    std::cout << "Yay!\n";
  } else {
    std::cout << "Oh no :(\n";
  }
}
