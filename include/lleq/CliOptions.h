/**
 * Copyright 2025 Veridise Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/Support/CommandLine.h>

namespace lleq::cli {
extern llvm::cl::OptionCategory lleqCat;

// Whether to stop after constructing the symbolic store and dump it
[[nodiscard]] bool dumpStore();

// Whether to disable the symbolic store construction phase
[[nodiscard]] bool disableStore();

// Whether to disable the deductive verification phase
[[nodiscard]] bool disableVerifier();

// Whether to emit an SMTLIB encoding instead of running analysis
[[nodiscard]] bool emitSMTLIB();

// The selected struct for SMTLIB emission
[[nodiscard]] std::string &smtStruct();

// Prime field name forwarded to LLZK SMT lowering
[[nodiscard]] std::string &smtField();

// The input file
[[nodiscard]] std::string &inputFile();

} // namespace lleq::cli
