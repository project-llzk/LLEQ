add_library(LLEQ)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(Analysis_SRC 
    Analysis/SymbolExpr.cpp
    Analysis/Unification.cpp
    Analysis/SymbolicStore.cpp
    Analysis/ScalarSymbolAnalysis.cpp
    Analysis/SymbolicStoreAnalysis.cpp)

set(Transforms_SRC
    Transforms/LLEQWhileToFor.cpp
    Transforms/LLEQIfToIfElse.cpp)

set(Verification_SRC
    Verification/SMTLIBEquivalenceEmitter.cpp)
if(LLEQ_HAS_LLZK_SMT_IR_HEADERS)
  list(APPEND Verification_SRC Verification/DeductiveVerifier.cpp)
endif()

target_sources(LLEQ PRIVATE
    ${Analysis_SRC}
    ${Transforms_SRC}
    ${Verification_SRC}
    )

get_target_property(
  LLEQ_LLZK_INCLUDE_DIRS
  ${LLEQ_LLZK_DIALECT_REGISTRATION_TARGET}
  INTERFACE_INCLUDE_DIRECTORIES)
if(LLEQ_LLZK_INCLUDE_DIRS)
  target_include_directories(LLEQ SYSTEM PRIVATE ${LLEQ_LLZK_INCLUDE_DIRS})
endif()
get_target_property(
  LLEQ_LLZK_DIALECT_INCLUDE_DIRS
  LLZK::LLZKDialectHeaders
  INTERFACE_INCLUDE_DIRECTORIES)
if(LLEQ_LLZK_DIALECT_INCLUDE_DIRS)
  target_include_directories(
    LLEQ SYSTEM PRIVATE ${LLEQ_LLZK_DIALECT_INCLUDE_DIRS})
endif()
target_include_directories(LLEQ SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

target_link_libraries(
  LLEQ
  PUBLIC
  LLEQHeaders
  LLZK::LLZKTransforms
  LLZK::LLZKAnalysis
  LLZK::LLZKDialect
  MLIRIR)
if(LLEQ_HAS_SMT_BACKEND)
  target_link_libraries(LLEQ PUBLIC LLZK::SMTConversions)
endif()
target_compile_definitions(LLEQ PRIVATE LLEQ_HAS_SMT_BACKEND=${LLEQ_HAS_SMT_BACKEND})
install(TARGETS LLEQ)
