# LLEQ

LLEQ is an equivalence verifier for zero-knowledge circuits in LLZK IR. LLEQ consumes **LLZK IR**, not high-level circuit languages directly. If your circuit is written in another language, first lower it to LLZK using an appropriate frontend, then run LLEQ on the resulting `.llzk` file. For example:

- [project-llzk/circom](https://github.com/project-llzk/circom), a Circom frontend with LLZK support
- [project-llzk/haloumi](https://github.com/project-llzk/haloumi), a Halo2/PLONKish frontend

## Quick Links

- [Build](#build)
- [Usage](#usage)
- [Examples](#examples)
- [Capabilities and Limitations](#capabilities-and-limitations)

## Build

The recommended way to build LLEQ is with Nix, but manual CMake builds are also supported.

### Dependencies

LLEQ depends on:

- LLVM
- MLIR
- LLZK
- a C++23-capable compiler
- `cvc5`
- `z3`
- Python 3

For manual builds, you must already have LLZK built and installed in a way that
CMake can discover with `find_package(LLZK)`. See:

- [project-llzk/llzk-lib](https://github.com/project-llzk/llzk-lib)

At runtime, LLEQ expects both `cvc5` and `z3` to be available on `PATH`. Solver
discovery can also be overridden with:

- `LLEQ_CVC5=/path/to/cvc5`
- `LLEQ_Z3=/path/to/z3`

### Build with Nix

Build the default package:

```bash
nix build .#lleq
```

Enter a development shell:

```bash
nix develop
```

Inside the dev shell, the built `lleq` binary is added to `PATH` for
convenience.

### Build with CMake

Configure:

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build build
```

If LLVM, MLIR, or LLZK are installed in nonstandard locations, point CMake at
their package configuration directories with variables such as `LLVM_DIR`,
`MLIR_DIR`, and `LLZK_DIR`.

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
lleq wp --struct DecomposeProduct_1 --field babybear examples/circom-demo/DecomposeProduct.llzk | z3 -in
```

## Examples

See [Examples](examples/README.md)

## Capabilities and Limitations

Analysis of structs with multidimensional arrays or subcomponent calls is not currently well-supported. Verification may also fail for structs where LLZK's product alignment produces poor alignments.
