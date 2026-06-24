# Circom Demo Examples

This directory holds Circom benchmarks lowered to LLZK and then normalized for
LLEQ demos.

## Regenerating The Example Set

Recompile the Circom benchmarks with the concrete backend in
`/Users/raghav/Veridise/circom-benchmarks`, then import them here:

```bash
python3 scripts/import_circom_demo_examples.py
```

The import script scans `~/Veridise/circom-benchmarks/llzk-outputs`, extracts
each benchmark's `llzk.main` root struct, and runs:

```text
--llzk-while-to-for
--llzk-compute-constrain-to-product=root-struct=<MainStruct>
--llzk-fuse-product-loops
--canonicalize
```

Successful outputs are copied into this directory as flat `.llzk` files. The
script also writes `import_manifest.csv` with preprocessing successes and
failures.

## Collecting Verification Data

Run:

```bash
python3 scripts/collect_circom_demo_results.py
```

For each imported benchmark, the script runs:

```text
lleq verify --flatten --struct <RootStruct>
lleq wp --struct <RootStruct> | cvc5 --produce-models
```

Each mode uses a 120-second timeout and reports one of:

- `verified`
- `counterexample`
- `partial`
- `timeout`
- `error`

`verify` is classified as `partial` when any signal remains marked with `*`.
`wp` is classified as `partial` when cvc5 returns `unknown`.
