#include "Util/ProjectBlock.h"

using namespace mlir;

namespace lleq::util {

template <class KeyT, class ValueT>
void insertOrDefault(DenseMap<KeyT, DenseSet<ValueT>> &map, KeyT key,
                     ValueT val) {
  if (auto it = map.find(key); it != map.end()) {
    it->second.insert(val);
  } else {
    map.insert({key, {}});
    map.find(key)->second.insert(val);
  }
}

void BlockProjector::addSeed(ArrayRef<Operation *> seed, Projection current) {
  for (auto *op : seed) {
    insertOrDefault(worklist, op, current);
  }
}

void BlockProjector::run() {
  while (!worklist.empty()) {
    // Pop one item off the worklist
    auto [op, inProjections] = *worklist.begin();
    worklist.erase(worklist.begin());

    // Add `op` to each projection it should belong to
    for (auto proj : inProjections) {
      insertOrDefault(projections, proj, op);
    }

    // Look at all producers of values that `op` consumes
    // TODO: add logic to also track struct read/writes
    for (auto operand : op->getOperands()) {
      if (auto producer = operand.getDefiningOp()) {
        // TODO: actually enqueue the containing op of `producer` in the current
        // block
        for (auto proj : inProjections) {
          insertOrDefault(worklist, producer, proj);
        }
      }
    }
  }
}

} // namespace lleq::util
