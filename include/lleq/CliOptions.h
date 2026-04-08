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

// Whether to emit an SMTLIB equivalence query instead of running analysis
[[nodiscard]] bool emitSMTLIBEquiv();

// The selected member for SMTLIB equivalence emission
[[nodiscard]] std::string &equivMember();

// The selected root struct for SMTLIB equivalence emission
[[nodiscard]] std::string &equivRootStruct();

// Optional prime field name forwarded to LLZK SMT lowering
[[nodiscard]] std::string &equivField();

// The input file
[[nodiscard]] std::string &inputFile();

} // namespace lleq::cli
