# LLEQ

LLEQ is an equivalence verifier for zero-knowledge circuits in LLZK IR, based on [Zequal](https://veridise.com/wp-content/uploads/2025/08/zequal.pdf). LLEQ consumes **LLZK IR**, not high-level circuit languages directly. If your circuit is written in another language, first lower it to LLZK using an appropriate frontend (see [Frontends](https://github.com/project-llzk/.github/blob/main/profile/README.md)), then run LLEQ on the resulting `.llzk` file

## Quick Links

- [Build](#build)
- [Usage](#usage)
- [Examples](#examples)
- [Capabilities and Limitations](#capabilities-and-limitations)

## Build

LLEQ can be built with Nix, or manually with CMake.

### Nix (Recommended)

This repository is configured with a Nix flakes environment. The flake was built using Nix version 2.31; older/newer versions may or may not.
Once you have a Nix installation (See [Installation](https://nix.dev/manual/nix/2.31/installation/index.html)), you can run `nix build .#lleq` from the repository root, or `nix develop` to enter a developer shell. Inside the dev shell, the built `lleq` binary is added to `PATH` for
convenience.

### Manual

For a manual build, you will need the following:

- LLVM
- MLIR
- LLZK
- a C++23-capable compiler
- `cvc5`
- `z3`
- Python 3

Both LLZK and MLIR must be built and installed in a way that CMake can discover with `find_package(LLZK)` (See [project-llzk/llzk-lib](https://github.com/project-llzk/llzk-lib/blob/main/doc/doxygen/01_setup.md#manual-build-setup) for instructions on how to build LLVM/MLIR/LLZK).

At runtime, LLEQ expects both `cvc5` and `z3` to be available on `PATH`. Solver
discovery can also be overridden by setting the following environment variables:

```
LLEQ_CVC5=/path/to/cvc5
LLEQ_Z3=/path/to/z3
```

#### Build with CMake

Configure:

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
```

If LLVM, MLIR, or LLZK are installed in nonstandard locations, point CMake at
their package configuration directories with `-DLLVM_DIR=...`,
`-DMLIR_DIR=...`, and `-DLLZK_DIR=...`.

Build:

```bash
cmake --build build
```

### Run the Test Suite

To run the LIT regression tests after a manual build:

```bash
cmake --build build --target check
```

The Nix package build also runs the test suite.

## Usage

LLEQ operates on an input `.llzk` file and a selected struct:

```bash
lleq <subcommand> --struct <StructName> [options] input.llzk
```

Available subcommands:

- `dump-store`: print the symbolic store constructed for the selected struct
- `verify`: prove equivalence of signal members
- `dump-smt`: print the SMT-LIB query generated for deductive verification
- `wp`: infer loop invariants and emit weakest-precondition-style verification conditions as SMT-LIB

Common options:

- `--struct <name>`: required; selects the struct to analyze
- `--field <field-name>`: selects the prime field when it cannot be inferred
- `--flatten`: run LLZK flattening and array-to-scalar lowering before analysis
- `--enable-store`: available for `verify` and `dump-smt`; adds extra symbolic-store facts

### Verify Output

`lleq verify` reports one line per signal member:

- `+ @Struct::member`: proven equivalent
- `- @Struct::member`: proven inequivalent
- `* @Struct::member`: unknown

For inequivalent members, LLEQ also prints a witness/constraint counterexample model.

Note that `lleq verify` does not perform loop invariant inference. To verify a struct containing loops or arrays, pass the `--flatten` argument to first unroll and scalarize the struct.

### Weakest-Precondition Queries

`lleq wp` emits an SMT-LIB query that can be sent directly to an SMT solver to
check the verification condition and, when it fails, obtain a counterexample.

For example:

```bash
lleq wp --struct DecomposeProduct_1 --field babybear examples/circom-examples/DecomposeProduct.llzk | z3 -in
```

## Examples

See [Examples](examples/README.md)

## Capabilities and Limitations

Analysis of structs with multidimensional arrays or subcomponent calls is not currently well-supported. Verification may also fail for structs where LLZK's product alignment produces poor alignments.
