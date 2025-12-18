/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/Support/CommandLine.h>

namespace lleq::cli {
extern llvm::cl::OptionCategory llCat;

// Whether to stop after constructing the symbolic store and dump it
[[nodiscard]] bool dumpStore();

// Whether to disable the symbolic store construction phase
[[nodiscard]] bool disableStore();

// Whether to disable the deductive verification phase
[[nodiscard]] bool disableVerifier();

// The input file
[[nodiscard]] std::string &inputFile();

} // namespace lleq::cli
