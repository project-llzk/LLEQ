add_library(LLEQ)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

target_sources(LLEQ PRIVATE
    Analysis/SymbolExpr.cpp
    Analysis/Unification.cpp
    Analysis/SymbolicStore.cpp
    Analysis/ScalarSymbolAnalysis.cpp
    Analysis/SymbolicStoreAnalysis.cpp)
target_link_libraries(LLEQ PUBLIC LLEQHeaders)
install(TARGETS LLEQ)
