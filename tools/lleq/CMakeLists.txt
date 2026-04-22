add_executable(lleq main.cpp CliOptions.cpp)
target_link_libraries(
  lleq
  LLEQHeaders
  LLEQ
  LLVMSupport
  ${LLEQ_LLZK_DIALECT_REGISTRATION_TARGET}
  MLIRParser)

