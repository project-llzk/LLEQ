import argparse
import collections
import csv
import pathlib
import re
import subprocess
import time

ENTRYPOINT_RE = re.compile(
    r"llzk\.main\s*=\s*!struct\.type<@([A-Za-z0-9_]+)(?:::@[A-Za-z0-9_]+)?<",
    re.ASCII,
)


def find_benchmarks(source_dir: pathlib.Path) -> list[pathlib.Path]:
    print(f"Finding benchmarks in {source_dir}")
    return sorted(source_dir.glob("**/*.llzk"))


def get_root_struct(llzk_file: pathlib.Path) -> str | None:
    with llzk_file.open(encoding="utf-8") as handle:
        for line in handle:
            if match := ENTRYPOINT_RE.search(line):
                return match.group(1)
    return None


def preprocess_benchmark(
    llzk_file: pathlib.Path, llzk_opt_bin: str, output_file: pathlib.Path
) -> tuple[str, float, str, str]:
    root_struct = get_root_struct(llzk_file)
    if root_struct is None:
        return ("error", 0.0, "", "failed to find llzk.main root struct")

    args = [
        llzk_opt_bin,
        "--llzk-while-to-for",
        "--canonicalize",
        f"--llzk-compute-constrain-to-product=root-struct={root_struct}",
        "--llzk-fuse-product-loops",
        str(llzk_file),
        "-o",
        str(output_file),
    ]

    start = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode == 0:
        return ("success", elapsed, root_struct, "")

    output_file.unlink(missing_ok=True)
    message = (proc.stderr or proc.stdout).strip()[:500]
    return ("error", elapsed, root_struct, message)


def clear_old_outputs(output_dir: pathlib.Path) -> None:
    for llzk_file in output_dir.glob("*.llzk"):
        llzk_file.unlink()
    for manifest in output_dir.glob("*.csv"):
        if manifest.name != "README.md":
            manifest.unlink()


def lower_circom_benchmarks(
    benchmark_dir: pathlib.Path, circom_frontend: str, nthreads: int
) -> None:
    args = [
        "python3",
        str(benchmark_dir / "scripts" / "circom_to_llzk_eval.py"),
        "--benchmark_dir",
        str(benchmark_dir),
        "--circom-bin",
        circom_frontend,
        "--nthreads",
        str(nthreads),
    ]
    subprocess.run(args, check=True)


def get_root() -> pathlib.Path:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.PIPE
        )
        return pathlib.Path(result.decode().strip())
    except subprocess.CalledProcessError:
        raise FileNotFoundError("Not a git repository")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Circom benchmarks into examples/circom-examples."
    )

    parser.add_argument(
        "--circom-frontend",
        required=True,
        help="Path to the Circom frontend binary used to lower benchmarks to LLZK.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=10,
        help="Number of lowering jobs to run at once.",
    )
    parser.add_argument(
        "--llzk",
        required=True,
        help="Path to llzk-opt.",
    )
    args = parser.parse_args()

    root = get_root()
    benchmark_dir = (root / "circom-benchmarks").resolve()
    source_dir = (root / "llzk-outputs").resolve()
    output_dir = (root / "examples" / "circom-examples").resolve()

    lower_circom_benchmarks(benchmark_dir, args.circom_frontend, args.nthreads)

    output_dir.mkdir(parents=True, exist_ok=True)
    clear_old_outputs(output_dir)

    rows: list[tuple[str, str, str, str, str, str, str]] = []
    imported_count = 0

    for llzk_file in find_benchmarks(source_dir):
        benchmark = llzk_file.parent.name.removesuffix("_llzk")
        root_struct = get_root_struct(llzk_file)
        output_file = output_dir / f"{benchmark}_{root_struct}.llzk"
        result, elapsed, root_struct, message = preprocess_benchmark(
            llzk_file, args.llzk, output_file
        )
        if result == "success":
            imported_count += 1
        rows.append(
            (
                benchmark,
                str(llzk_file),
                str(output_file if result == "success" else ""),
                root_struct,
                result,
                f"{elapsed:.6f}",
                message,
            )
        )

    manifest_path = output_dir / "import_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Benchmark",
                "Source File",
                "Optimized File",
                "Root Struct",
                "Preprocess Result",
                "Time Seconds",
                "Message",
            ]
        )
        writer.writerows(rows)

    print(f"imported: {imported_count}, failed: {len(rows) - imported_count}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
