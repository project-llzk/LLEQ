#include "lleq/CliOptions.h"
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>

namespace lleq::cli {

llvm::cl::OptionCategory lleqCat("LLEQ", "Options to configure LLEQ");

// lleq verify --struct ... --field ... [--enable-store]
// lleq dump-smt --struct ... --field ... [--enable-store]
// lleq dump-store --struct ...

llvm::cl::SubCommand verifyCmd("verify", "Verify struct equivalence");

llvm::cl::SubCommand
    dumpSmtCmd("dump-smt", "Print the SMTLIB encoding of the selected struct");
llvm::cl::SubCommand dumpStoreCmd(
    "dump-store",
    "Print the symbolic store constructed for the selected struct");

llvm::cl::SubCommand wpCmd("wp", "Compute and dump weakest precondition");

// --struct ...
static llvm::cl::opt<std::string> structOpt(
    "struct", llvm::cl::desc("The struct to use for verification/debugging"),
    llvm::cl::sub(llvm::cl::SubCommand::getAll()), llvm::cl::Required);

// --field ...
static llvm::cl::opt<std::string> smtVerifyFieldNameOpt(
    "field",
    llvm::cl::desc("The prime field to use for SMT lowering, if not specified "
                   "in the LLZK file"),
    llvm::cl::sub(verifyCmd));
static llvm::cl::opt<std::string> smtDumpFieldNameOpt(
    "field",
    llvm::cl::desc("The prime field to use for SMT lowering, if not specified"),
    llvm::cl::sub(dumpSmtCmd));

// [--enable-store]
static llvm::cl::opt<bool>
    verifyEnableStoreOpt("enable-store",
                         llvm::cl::desc("Enable symbolic store construction"),
                         llvm::cl::sub(verifyCmd));
static llvm::cl::opt<bool>
    dumpEnableStoreOpt("enable-store",
                       llvm::cl::desc("Enable symbolic store construction"),
                       llvm::cl::sub(dumpSmtCmd));

static llvm::cl::opt<std::string>
    inputFileOpt(llvm::cl::Positional, llvm::cl::desc("[.llzk file]"),
                 llvm::cl::sub(llvm::cl::SubCommand::getAll()));

SubCmd subCmd() {
  if (verifyCmd) {
    return SubCmd::Verify;
  }
  if (dumpSmtCmd) {
    return SubCmd::DumpSmt;
  }
  if (wpCmd) {
    return SubCmd::WeakestPrecondition;
  }
  return SubCmd::DumpStore;
}

std::string &smtStruct() { return structOpt; }
std::string &fieldName() {
  return verifyCmd ? smtVerifyFieldNameOpt : smtDumpFieldNameOpt;
}
bool enableStore() {
  return verifyCmd ? verifyEnableStoreOpt : dumpEnableStoreOpt;
}

std::string &inputFile() { return inputFileOpt; }

} // namespace lleq::cli
