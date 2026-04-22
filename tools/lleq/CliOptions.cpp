#include "lleq/CliOptions.h"
#include <llvm/Support/CommandLine.h>

namespace lleq::cli {

llvm::cl::OptionCategory lleqCat("LLEQ", "Options to configure LLEQ");

static llvm::cl::opt<bool> dumpStoreOpt(
    "dump-store",
    llvm::cl::desc("Stop after constructing the symbolic store and print it"),
    llvm::cl::cat(lleqCat));

static llvm::cl::opt<bool> disableStoreOpt(
    "disable-store",
    llvm::cl::desc(
        "Skip symbolic-store construction and run only the deductive verifier"),
    llvm::cl::cat(lleqCat));

static llvm::cl::opt<bool>
    disableVerifierOpt("disable-verifier",
                       llvm::cl::desc("Disable the deductive verifier"),
                       llvm::cl::cat(lleqCat));

static llvm::cl::opt<bool> emitSMTLIBOpt(
    "emit-smtlib",
    llvm::cl::desc("Print the SMTLIB encoding of the selected struct and stop"),
    llvm::cl::cat(lleqCat));

static llvm::cl::opt<std::string>
    smtStructOpt("struct",
                 llvm::cl::desc("The struct to lower before emitting SMTLIB"),
                 llvm::cl::init(""), llvm::cl::cat(lleqCat));

static llvm::cl::opt<std::string> smtFieldOpt(
    "field", llvm::cl::desc("Prime field name forwarded to LLZK SMT lowering"),
    llvm::cl::init(""), llvm::cl::cat(lleqCat));

static llvm::cl::opt<std::string> inputFileOpt(llvm::cl::Positional,
                                               llvm::cl::desc("[.llzk file]"),
                                               llvm::cl::cat(lleqCat));

bool dumpStore() { return dumpStoreOpt; }

bool disableStore() { return disableStoreOpt; }

bool disableVerifier() { return disableVerifierOpt; }

bool emitSMTLIB() { return emitSMTLIBOpt; }

std::string &smtStruct() { return smtStructOpt; }

std::string &smtField() { return smtFieldOpt; }

std::string &inputFile() { return inputFileOpt; }

} // namespace lleq::cli
