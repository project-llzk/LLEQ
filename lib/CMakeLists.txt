add_library(LLEQ)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

target_sources(LLEQ PRIVATE
    Analysis/SymbolExpr.cpp
    Analysis/SymbolicStore.cpp)
target_link_libraries(LLEQ PUBLIC LLEQHeaders)
install(TARGETS LLEQ)
