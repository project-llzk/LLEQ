find_program(LLVM_EXTERNAL_LIT NAMES lit llvm-lit ${LLVM_EXTERNAL_LIT} REQUIRED)
message(STATUS "Using lit: ${LLVM_EXTERNAL_LIT}")
find_program(LLEQ_CVC5_EXECUTABLE NAMES cvc5 DOC "Path to the cvc5 executable")
if(NOT LLEQ_CVC5_EXECUTABLE)
  set(LLEQ_CVC5_EXECUTABLE "" CACHE FILEPATH "Path to the cvc5 executable" FORCE)
endif()
message(STATUS "Using cvc5: ${LLEQ_CVC5_EXECUTABLE}")

set(LLEQ_TEST_DEPENDS lleq)
set(LIT_ARGS "--xunit-xml-output report.xml")

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

add_lit_testsuite(
    check-lit
    "Running the lit-based regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    ${LLEQ_TEST_DEPENDS}
    ARGS
    ${LIT_ARGS})
