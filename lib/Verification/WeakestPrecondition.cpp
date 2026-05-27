/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Verification/WeakestPrecondition.h"

#include <llzk/Dialect/Function/IR/Ops.h>
#include <vector>

using namespace llzk;

namespace lleq {

cvc5::Term getPostcondition(component::StructDefOp structDef,
                            cvc5::TermManager &mgr) {

  auto members = structDef.getMemberDefs();
  std::vector<cvc5::Term> memberEquivs;

  for (auto memberDef : members) {
    auto memberName = memberDef.getSymName();
    auto witnessSym =
        mgr.mkConst(mgr.getIntegerSort(), (memberName + "_w").str());
    auto constraintSym =
        mgr.mkConst(mgr.getIntegerSort(), (memberName + "_c").str());
    memberEquivs.push_back(
        mgr.mkTerm(cvc5::Kind::EQUAL, {witnessSym, constraintSym}));
  }

  return mgr.mkTerm(cvc5::Kind::AND, memberEquivs);
}

} // namespace lleq
