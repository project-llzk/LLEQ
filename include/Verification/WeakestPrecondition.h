/**
 * Copyright 2026 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cvc5/cvc5.h>

#include <llzk/Dialect/Struct/IR/Ops.h>

namespace lleq {
cvc5::Term getPostcondition(llzk::component::StructDefOp, cvc5::TermManager &);
}
