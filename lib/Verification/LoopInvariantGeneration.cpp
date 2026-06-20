#include "Verification/LoopInvariantGeneration.h"

#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>

using namespace llzk;
using namespace mlir;

namespace lleq {

static inline Block *nestedLoopBody(scf::ForOp loop,
                                    SmallVector<Value> &counters) {
  counters.push_back(loop.getInductionVar());
  if (auto first = dyn_cast<scf::ForOp>(loop.getBody()->front())) {
    return nestedLoopBody(first, counters);
  }
  return loop.getBody();
}

void conjecturePredicates(scf::ForOp loop) {
  SmallVector<Value> counters;
  auto *body = nestedLoopBody(loop, counters);

  DenseSet<Value> signalArrays;
  SmallVector<array::WriteArrayOp> arrayWrites;
  body->walk([&signalArrays, &arrayWrites](array::WriteArrayOp write) {
    if (llvm::any_of(write.getArrRef().getUsers(), [](auto *user) {
          return llvm::isa<component::MemberWriteOp>(user);
        })) {
      signalArrays.insert(write.getArrRef());
      arrayWrites.push_back(write);
    }
  });

  SmallVector<component::MemberWriteOp> scalarWrites;
  body->walk([&scalarWrites, &signalArrays](component::MemberWriteOp write) {
    // Skip array-typed member writes (note: if for some reason the loop
    // writes a full array to a struct member each iteration this may miss
    // it, but there's no indices for us to do anything about in that case
    // anyway)
    auto val = write.getVal();
    if (!isa<array::ArrayType>(val.getType())) {
      scalarWrites.push_back(write);
    }
  });
}

} // namespace lleq
