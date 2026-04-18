#!/usr/bin/env python3

# Modified version of
# https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/examples/standalone/test/lit.cfg.py
# from LLVM, which is licensed under Apache 2.0 with LLVM Exceptions.

# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'LLEQ'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.llzk']
config.suffixes.extend(config.extra_suffixes)

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lleq_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%input_dir', config.test_source_root))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['CMakeLists.txt']

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lleq_obj_root, 'test')
config.lleq_tools_dir = os.path.join(config.lleq_obj_root, 'tools/lleq')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.lleq_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
if config.cvc5_executable:
    llvm_config.with_environment('LLEQ_CVC5', config.cvc5_executable)

tool_dirs = [config.lleq_tools_dir, config.llvm_tools_dir]
tools = [
    "lleq"
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.available_features.add('smt-equiv')

# Limit testing time in the case of non-converging analyses
config.maxIndividualTestTime = 60
