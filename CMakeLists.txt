cmake_minimum_required(VERSION 3.23)

set(ZKVERIFIER_VERSION "0.1.0")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

project(
    ZKVerifier
    DESCRIPTION "Port of Zequal to LLZK"
    VERSION "${ZKVERIFIER_VERSION}"
    HOMEPAGE_URL https://github.com/Veridise/LLZK-Verifier
)


set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard")

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DebWithSans")
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    message(STATUS "Adding coverage flags")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
  endif()
  add_compile_options(-O0 -g)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "DebWithSans")
  # NOTE: undefined behavior sanitizer ("undefined") has been removed due to its
  # poor interaction with gtest fixtures and other standard MLIR/LLVM operations.
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVM in: ${LLVM_DIR}")
  
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR in: ${MLIR_DIR}")
  
find_package(LLZK REQUIRED CONFIG)
message(STATUS "Using LLZK in: ${LLZK_DIR}")

if(TARGET LLZK::SMTConversions)
  set(LLEQ_HAS_SMT_BACKEND 1)
else()
  set(LLEQ_HAS_SMT_BACKEND 0)
endif()

set(LLEQ_HAS_LLZK_SMT_IR_HEADERS 0)
if(TARGET LLZK::LLZKDialectHeaders)
  get_target_property(
    LLEQ_LLZK_DIALECT_INCLUDE_DIRS
    LLZK::LLZKDialectHeaders
    INTERFACE_INCLUDE_DIRECTORIES)
  foreach(LLEQ_LLZK_INCLUDE_DIR IN LISTS LLEQ_LLZK_DIALECT_INCLUDE_DIRS)
    if(EXISTS "${LLEQ_LLZK_INCLUDE_DIR}/llzk/Dialect/SMT/IR/SMTOps.h")
      set(LLEQ_HAS_LLZK_SMT_IR_HEADERS 1)
      break()
    endif()
  endforeach()
endif()

if(TARGET LLZK::LLZKDialectRegistration)
  set(LLEQ_LLZK_DIALECT_REGISTRATION_TARGET LLZK::LLZKDialectRegistration)
elseif(TARGET LLZK::DialectRegistration)
  set(LLEQ_LLZK_DIALECT_REGISTRATION_TARGET LLZK::DialectRegistration)
else()
  message(FATAL_ERROR "Could not find an LLZK dialect registration target")
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

add_custom_target(check DEPENDS $<$<BOOL:BUILD_TESTING>:check-lit>)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)
