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

target_sources(LLEQ PRIVATE
    ${Analysis_SRC}
    ${Transforms_SRC}
    )
target_link_libraries(LLEQ PUBLIC LLEQHeaders LLZKTransforms LLZKAnalysis LLZKDialect LLZKTransforms MLIRIR)
install(TARGETS LLEQ)
