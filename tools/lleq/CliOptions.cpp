#include "lleq/CliOptions.h"
#include <llvm/Support/CommandLine.h>

namespace lleq::cli {

llvm::cl::OptionCategory llCat("LLEQ", "Options to configure LLEQ");

static llvm::cl::opt<bool> dumpStoreOpt(
    "dump-store",
    llvm::cl::desc("Stop after constructing the symbolic store and print it"),
    llvm::cl::cat(llCat));

static llvm::cl::opt<bool> disableStoreOpt(
    "disable-store",
    llvm::cl::desc(
        "Disable the symbolic store construction/lightweight static analysis"),
    llvm::cl::cat(llCat));

static llvm::cl::opt<bool>
    disableVerifierOpt("disable-verifier",
                       llvm::cl::desc("Disable the deductive verifier"),
                       llvm::cl::cat(llCat));

static llvm::cl::opt<std::string> inputFileOpt(llvm::cl::Positional,
                                               llvm::cl::desc("[.llzk file]"),
                                               llvm::cl::cat(llCat));

bool dumpStore() { return dumpStoreOpt; }

bool disableStore() { return disableStoreOpt; }

bool disableVerifier() { return disableVerifierOpt; }

std::string &inputFile() { return inputFileOpt; }

} // namespace lleq::cli
