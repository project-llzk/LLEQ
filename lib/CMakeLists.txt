add_library(LLEQ)

target_sources(LLEQ PRIVATE
    Analysis/SymbolExpr.cpp
    Analysis/SymbolicStore.cpp)
target_link_libraries(LLEQ PUBLIC LLEQHeaders)
install(TARGETS LLEQ)
