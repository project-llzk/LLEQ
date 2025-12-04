#include <cstdlib>
#include <iostream>
#include <llzk/Dialect/InitDialects.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LogicalResult.h>

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [.llzk file]\n";
    return EXIT_FAILURE;
  }

  auto inputFilename = argv[1];

  mlir::DialectRegistry registry;
  llzk::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();
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
    std::cerr << "Failed to parse " << inputFilename << "\n";
    return EXIT_FAILURE;
  }
  mod->dumpPretty();
  return EXIT_SUCCESS;
}
