add_executable(lleq main.cpp CliOptions.cpp)
add_executable(test test.cpp)
find_package(cvc5)
target_link_libraries(
  lleq
  PRIVATE
  cvc5
  LLEQHeaders
  LLEQ
  ${LLEQ_LLZK_DIALECT_REGISTRATION_TARGET}
  LLZK::AllDialects
  LLZK::LLZKAnalysis
  LLZK::LLZKTransforms
  LLZK::SMTConversions
  MLIRParser
  MLIRIR
  LLVMSupport)

target_link_libraries(test cvc5)
