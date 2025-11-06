#include <iostream>
#include <mlir/Support/LogicalResult.h>

mlir::LogicalResult test() { return mlir::failure(); }

int main(int argc, char **argv) {
  if (mlir::failed(test())) {
    std::cout << "Good job\n";
  }
  return 0;
}
