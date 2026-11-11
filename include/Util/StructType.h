/**
 * Copyright 2026 Project LLZK.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llzk/Dialect/Struct/IR/Types.h>

namespace lleq::util {

/// Canonicalize an explicitly empty parameter list without changing nonempty
/// parameters or the fully qualified struct name.
inline llzk::component::StructType
normalizeStructType(llzk::component::StructType type) {
  if (type.getParams() && type.getParams().empty()) {
    return llzk::component::StructType::get(type.getNameRef());
  }
  return type;
}

} // namespace lleq::util
