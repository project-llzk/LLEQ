# Examples

This directory collects example circuits and benchmark artifacts used for LLEQ
development, demos, and evaluation.

## `circom-examples`

The [`circom-examples`](./circom-examples) subdirectory contains LLZK IR derived
from Circom benchmarks maintained in the
[`project-llzk/circom-benchmarks`](https://github.com/project-llzk/circom-benchmarks)
repository.

Each imported benchmark is lowered to LLZK and then normalized with the
following pass pipeline before being checked into this repository:

```text
--llzk-while-to-for
--llzk-compute-constrain-to-product=root-struct=<RootStruct>
--llzk-fuse-product-loops
--canonicalize
```

Here, `<RootStruct>` is the benchmark's `llzk.main` root struct. The resulting
normalized `.llzk` files are stored flat in [`circom-examples`](./circom-examples).

The subdirectory also includes `verification_results.csv`, which records LLEQ
verification outcomes for the imported set.

## Collecting Verification Data

Run:

```bash
python3 scripts/collect_circom_demo_results.py
```

For each imported benchmark, the script runs:

```text
lleq verify --flatten --struct <RootStruct>
lleq wp --struct <RootStruct> | z3 -in
```

Each mode uses a 120-second timeout and reports one of:

- `verified`
- `counterexample`
- `partial`
- `timeout`
- `error`

`verify` is classified as `partial` when any signal remains marked with `*`.
`wp` is classified as `partial` when z3 returns `unknown`.
