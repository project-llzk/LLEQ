add_library(LLEQ)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(Analysis_SRC
    Analysis/SymbolExpr.cpp
    Analysis/Unification.cpp
    Analysis/SymbolicStore.cpp
    Analysis/ScalarSymbolAnalysis.cpp
    Analysis/SymbolicStoreAnalysis.cpp)

set(Transforms_SRC
    Transforms/LLEQIfToIfElse.cpp)

set(Verification_SRC
    Verification/SolverUtils.cpp
    Verification/SMTLIBEquivalenceEmitter.cpp
    Verification/DeductiveVerifier.cpp
    Verification/SymbolicVerifier.cpp
    Verification/FixpointVerifier.cpp
    Verification/TermUtils.cpp
    Verification/WeakestPrecondition.cpp
    Verification/StructContracts.cpp
    Verification/Utils.cpp)

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
  LLZK::AllDialects
  LLZK::SMTConversions
  MLIRIR)
install(TARGETS LLEQ)
