add_executable(lleq main.cpp CliOptions.cpp)
target_link_libraries(
  lleq
  PRIVATE
  LLEQHeaders
  LLEQ
  ${LLEQ_LLZK_DIALECT_REGISTRATION_TARGET}
  LLZK::LLZKTransforms
  LLZK::LLZKAnalysis
  LLZK::AllDialects
  LLZK::SMTConversions
  MLIRParser
  MLIRIR
  LLVMSupport)
