/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/SymbolicVerifier.h"
#include <llvm/Support/raw_ostream.h>

namespace lleq {
using namespace llzk;
using namespace mlir;

LogicalResult SymbolicVerifier::buildStore() {
  return store.buildStore(structDef);
}

bool SymbolicVerifier::areEquivalent(llvm::StringRef memberName) {
  Signal witness{Signal::Source::Witness, memberName};
  Signal constraint{Signal::Source::Constraint, memberName};

  // If the two signals are written to at different indices, they can't be
  // equivalent
  auto witnessWritten = store.getWrittenIndices(witness);
  auto constraintWritten = store.getWrittenIndices(constraint);
  if (witnessWritten != constraintWritten) {
    return false;
  }

  // Check if the written value at each index matches
  for (auto index : witnessWritten) {
    if (*store.lookup(witness, index) != *store.lookup(constraint, index)) {
      return false;
    }
  }

  return true;
}

static std::string _get_assertion(StringRef memberName, llzk::Field field) {
  std::string assertion;
  llvm::raw_string_ostream os{assertion};
  os << "(assert (= (mod " << memberName << "_w " << field.prime() << ") (mod "
     << memberName << "_c " << field.prime() << ")))";
  return assertion;
}

SmallVector<std::string>
SymbolicVerifier::generateAssertions(llzk::Field field) {
  SmallVector<std::string> assertions;
  for (auto memberDef : structDef.getMemberDefs()) {
    auto memberName = memberDef.getSymName();
    if (areEquivalent(memberName)) {
      assertions.push_back(_get_assertion(memberName, field));
    }
  }
  return assertions;
}

} // namespace lleq
