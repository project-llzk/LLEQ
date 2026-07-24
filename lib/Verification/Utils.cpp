#include "Verification/Utils.h"

#include <llzk/Analysis/LightweightSignalEquivalenceAnalysis.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Transforms/LLZKComputeConstrainToProductPass.h>
#include <llzk/Transforms/LLZKTransformationPasses.h>
#include <mlir/Pass/PassManager.h>

using namespace mlir;
using namespace llzk;

namespace lleq {
void ensureProductFunc(ModuleOp module, component::StructDefOp structDef) {
  if (structDef.getProductFuncOp()) {
    return;
  }

  auto computeFunc = structDef.getComputeFuncOp();
  auto constrainFunc = structDef.getConstrainFuncOp();
  if (!computeFunc || !constrainFunc) {
    structDef.emitError()
        << "expected the selected struct to define either @product or both "
           "@compute and @constrain";
    return;
  }

  SymbolTableCollection tables;
  LightweightSignalEquivalenceAnalysis equivalence(module);
  ProductAligner aligner(tables, equivalence);
  auto productFunc = aligner.alignFuncs(structDef, computeFunc, constrainFunc);
  if (!productFunc) {
    structDef.emitError()
        << "failed to align @compute/@constrain into @product";
    return;
  }

  ensure(succeeded(aligner.alignCalls(productFunc)),
         "failed to align subcomponent calls");

  // Now, try fusing loops
  PassManager pm{module->getContext()};
  pm.addPass(llzk::createFuseProductLoopsPass());
  ensure(succeeded(pm.run(module)), "failed to align loops");
}
} // namespace lleq
