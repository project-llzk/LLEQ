add_executable(lleq main.cpp CliOptions.cpp)
add_executable(test test.cpp)
find_package(cvc5)
target_link_libraries(
  lleq
  cvc5
  LLEQHeaders
  LLEQ
  LLVMSupport
  ${LLEQ_LLZK_DIALECT_REGISTRATION_TARGET}
  MLIRParser)

target_link_libraries(test cvc5)
