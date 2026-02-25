#include "Analysis/SymbolExpr.h"
#include "Analysis/Unification.h"
#include <llvm/Support/Debug.h>

using namespace lleq;

int main() {
  SymbolPool pool;
  Symbol u = pool.fresh_unknown();
  Symbol x = pool.templ_param("x");
  Symbol y = pool.templ_param("y");
  Symbol a = pool.func_call("f", {x, u});
  Symbol b = pool.func_call("f", {u, y});
  Substitutions s;
  if (llvm::failed(unify(a, b, s))) {
    llvm::dbgs() << "Failed to unify " << a << " with " << b << "\n";
    return 0;
  }
  for (auto [k, v] : s) {
    llvm::dbgs() << "?" << k << " -> " << v << "\n";
  }
}
