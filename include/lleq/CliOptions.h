/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/Support/CommandLine.h>

namespace lleq::cli {
// extern llvm::cl::OptionCategory lleqCat;
enum class SubCmd { Verify, DumpSmt, DumpStore, WeakestPrecondition };

// The mode in which to run LLEQ (verify, dump-smt, or dump-store)
[[nodiscard]] SubCmd subCmd();

// The name of the struct to use for verification/debugging
[[nodiscard]] std::string &smtStruct();

// The name of the prime field to use for SMT lowering
[[nodiscard]] std::string &fieldName();

// Whether to enable symbolic store construction
[[nodiscard]] bool enableStore();

// The input file
[[nodiscard]] std::string &inputFile();

} // namespace lleq::cli
