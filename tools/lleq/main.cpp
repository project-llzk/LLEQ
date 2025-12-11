#include <cstdlib>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/InitDialects.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LogicalResult.h>

#define BUG_REPORT_URL "https://github.com/Veridise/LLEQ/issues"

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  llvm::setBugReportMsg("PLEASE submit a bug report to " BUG_REPORT_URL
                        " and include the crash backtrace, relevant LLZK "
                        "files, and associated run script(s).\n");

  if (argc < 2) {
    llvm::errs() << "Usage: " << argv[0] << " [.llzk file]\n";
    return EXIT_FAILURE;
  }

  auto inputFilename = argv[1];

  mlir::DialectRegistry registry;
  llzk::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    (void)diag; // TODO
    return mlir::success();
  });

  mlir::OpBuilder builder(&context);
  auto mod = builder.create<mlir::ModuleOp>(
      mlir::NameLoc::get(builder.getStringAttr("LLEQ")));

  auto parserConfig = mlir::ParserConfig(&context);
  if (mlir::failed(
          mlir::parseSourceFile(inputFilename, mod.getBody(), parserConfig))) {
    llvm::errs() << "Failed to parse " << inputFilename << '\n';
    return EXIT_FAILURE;
  }
  mod->dumpPretty();
  return EXIT_SUCCESS;
}
