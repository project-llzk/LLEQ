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
        "Disable the symbolic store construction/lightweight static analysis"),
    llvm::cl::cat(lleqCat));

static llvm::cl::opt<bool>
    disableVerifierOpt("disable-verifier",
                       llvm::cl::desc("Disable the deductive verifier"),
                       llvm::cl::cat(lleqCat));

static llvm::cl::opt<bool> emitSMTLIBEquivOpt(
    "emit-smtlib-equiv",
    llvm::cl::desc(
        "Lower the selected struct to SMT and print a single-member "
        "inequivalence query as SMTLIB"),
    llvm::cl::cat(lleqCat));

static llvm::cl::opt<std::string> equivMemberOpt(
    "member",
    llvm::cl::desc("The struct member to compare as <member>_w != <member>_c"),
    llvm::cl::init(""), llvm::cl::cat(lleqCat));

static llvm::cl::opt<std::string> equivRootStructOpt(
    "root-struct",
    llvm::cl::desc("The root struct to lower before emitting SMTLIB"),
    llvm::cl::init("Main"), llvm::cl::cat(lleqCat));

static llvm::cl::opt<std::string> equivFieldOpt(
    "field",
    llvm::cl::desc(
        "Optional prime field name forwarded to LLZK SMT lowering"),
    llvm::cl::init(""), llvm::cl::cat(lleqCat));

static llvm::cl::opt<std::string> inputFileOpt(llvm::cl::Positional,
                                               llvm::cl::desc("[.llzk file]"),
                                               llvm::cl::cat(lleqCat));

bool dumpStore() { return dumpStoreOpt; }

bool disableStore() { return disableStoreOpt; }

bool disableVerifier() { return disableVerifierOpt; }

bool emitSMTLIBEquiv() { return emitSMTLIBEquivOpt; }

std::string &equivMember() { return equivMemberOpt; }

std::string &equivRootStruct() { return equivRootStructOpt; }

std::string &equivField() { return equivFieldOpt; }

std::string &inputFile() { return inputFileOpt; }

} // namespace lleq::cli
