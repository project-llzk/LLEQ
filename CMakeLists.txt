cmake_minimum_required(VERSION 3.23)

set(ZKVERIFIER_VERSION "0.1.0")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_BUILD_TYPE Debug)

project(
    ZKVerifier
    DESCRIPTION "Port of Zequal to LLZK"
    VERSION "${ZKVERIFIER_VERSION}"
    HOMEPAGE_URL https://github.com/Veridise/LLZK-Verifier
)


# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
message(STATUS "Flags: ${CMAKE_CXX_FLAGS_DEBUG}")

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVM in: ${LLVM_DIR}")
  
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR in: ${MLIR_DIR}")
  
find_package(LLZK REQUIRED CONFIG)
message(STATUS "Using LLZK in: ${LLZK_DIR}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
