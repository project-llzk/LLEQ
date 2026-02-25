add_executable(lleq main.cpp CliOptions.cpp)
target_link_libraries(lleq LLEQHeaders LLEQ LLVMSupport LLZK::LLZKDialectRegistration MLIRParser)


