import argparse
import csv
import multiprocessing
import pathlib
import re
import subprocess
import time

ENTRYPOINT_RE = re.compile(
    r"llzk\.main\s*=\s*!struct\.type<@([A-Za-z0-9_]+)(?:::@[A-Za-z0-9_]+)?<",
    re.ASCII,
)
VERIFY_COUNTEREXAMPLE_RE = re.compile(r"^- @", re.MULTILINE)
VERIFY_UNKNOWN_RE = re.compile(r"^\* @", re.MULTILINE)
SAT_RE = re.compile(r"^\s*sat\s*$", re.MULTILINE)
UNSAT_RE = re.compile(r"^\s*unsat\s*$", re.MULTILINE)
UNKNOWN_RE = re.compile(r"^\s*unknown\s*$", re.MULTILINE)


def get_benchmarks(benchmark_dir: pathlib.Path) -> list[tuple[str, pathlib.Path, str]]:
    benchmarks: list[tuple[str, pathlib.Path, str]] = []
    for llzk_file in sorted(benchmark_dir.glob("*.llzk")):
        root_struct = get_root_struct(llzk_file)
        if root_struct is None:
            continue
        benchmarks.append((llzk_file.stem, llzk_file, root_struct))
    return benchmarks


def get_root_struct(llzk_file: pathlib.Path) -> str | None:
    with llzk_file.open(encoding="utf-8") as handle:
        for line in handle:
            if match := ENTRYPOINT_RE.search(line):
                return match.group(1)
    return None


def classify_verify(stdout: str) -> tuple[str, str]:
    if VERIFY_UNKNOWN_RE.search(stdout):
        return ("partial", "verification finished with unknown members")
    if VERIFY_COUNTEREXAMPLE_RE.search(stdout):
        return ("counterexample", "")
    return ("verified", "")


def run_verify(
    benchmark: str,
    llzk_file: pathlib.Path,
    root_struct: str,
    lleq_bin: str,
    timeout: float,
) -> tuple[str, str, str, str, str, str]:
    args = [lleq_bin, "verify", "--flatten", "--struct", root_struct, str(llzk_file)]
    start = time.perf_counter()
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return (benchmark, root_struct, "verify", "timeout", f"{elapsed:.6f}", "timeout")

    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip()[:500]
        return (benchmark, root_struct, "verify", "error", f"{elapsed:.6f}", message)

    result, message = classify_verify(proc.stdout)
    message = message or proc.stdout.strip()[:500]
    return (benchmark, root_struct, "verify", result, f"{elapsed:.6f}", message)


def run_wp(
    benchmark: str,
    llzk_file: pathlib.Path,
    root_struct: str,
    lleq_bin: str,
    cvc5_bin: str,
    timeout: float,
) -> tuple[str, str, str, str, str, str]:
    lleq_args = [lleq_bin, "wp", "--struct", root_struct, str(llzk_file)]
    cvc5_args = [cvc5_bin, "--produce-models"]
    start = time.perf_counter()

    try:
        lleq_proc = subprocess.Popen(
            lleq_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cvc5_proc = subprocess.Popen(
            cvc5_args,
            stdin=lleq_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert lleq_proc.stdout is not None
        lleq_proc.stdout.close()
        cvc5_stdout, cvc5_stderr = cvc5_proc.communicate(timeout=timeout)
        lleq_stdout, lleq_stderr = lleq_proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        lleq_proc.kill()
        cvc5_proc.kill()
        lleq_proc.communicate()
        cvc5_proc.communicate()
        return (benchmark, root_struct, "wp", "timeout", f"{elapsed:.6f}", "timeout")

    elapsed = time.perf_counter() - start
    if lleq_proc.returncode != 0:
        message = (lleq_stderr or lleq_stdout).strip()[:500]
        return (benchmark, root_struct, "wp", "error", f"{elapsed:.6f}", message)
    if cvc5_proc.returncode != 0:
        message = (cvc5_stderr or cvc5_stdout).strip()[:500]
        return (benchmark, root_struct, "wp", "error", f"{elapsed:.6f}", message)
    if UNSAT_RE.search(cvc5_stdout):
        return (benchmark, root_struct, "wp", "verified", f"{elapsed:.6f}", "unsat")
    if SAT_RE.search(cvc5_stdout):
        return (benchmark, root_struct, "wp", "counterexample", f"{elapsed:.6f}", "sat")
    if UNKNOWN_RE.search(cvc5_stdout):
        return (benchmark, root_struct, "wp", "partial", f"{elapsed:.6f}", "unknown")

    message = (cvc5_stdout or cvc5_stderr).strip()[:500]
    return (benchmark, root_struct, "wp", "error", f"{elapsed:.6f}", message)


def run_task(
    benchmark: str,
    llzk_file: pathlib.Path,
    root_struct: str,
    mode: str,
    lleq_bin: str,
    cvc5_bin: str,
    timeout: float,
) -> tuple[str, str, str, str, str, str]:
    if mode == "verify":
        return run_verify(benchmark, llzk_file, root_struct, lleq_bin, timeout)
    return run_wp(benchmark, llzk_file, root_struct, lleq_bin, cvc5_bin, timeout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect LLEQ verification data for examples/circom-demo."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="examples/circom-demo",
        help="Directory containing optimized LLZK demo examples.",
    )
    parser.add_argument(
        "--lleq-bin",
        default="./build/tools/lleq/lleq",
        help="Path to the lleq binary.",
    )
    parser.add_argument(
        "--cvc5-bin",
        default="/nix/store/hxfws6z4z0c3d8l87pr4lfz672vxp32d-cvc5-1.3.1/bin/cvc5",
        help="Path to the cvc5 binary.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-benchmark timeout in seconds.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of jobs to run at once.",
    )
    parser.add_argument(
        "--output",
        default="examples/circom-demo/verification_results.csv",
        help="CSV path for collected results.",
    )
    args = parser.parse_args()

    benchmark_dir = pathlib.Path(args.benchmark_dir).resolve()
    benchmarks = get_benchmarks(benchmark_dir)
    tasks: list[tuple[str, pathlib.Path, str, str, str, str, float]] = []
    for benchmark, llzk_file, root_struct in benchmarks:
        tasks.append(
            (benchmark, llzk_file, root_struct, "verify", args.lleq_bin, args.cvc5_bin, args.timeout)
        )
        tasks.append(
            (benchmark, llzk_file, root_struct, "wp", args.lleq_bin, args.cvc5_bin, args.timeout)
        )

    with multiprocessing.Pool(args.nthreads) as pool:
        results = pool.starmap(run_task, tasks)

    results.sort()
    output_path = pathlib.Path(args.output).resolve()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Benchmark", "Root Struct", "Mode", "Result", "Time Seconds", "Message"]
        )
        writer.writerows(results)

    counts = {
        "verified": 0,
        "counterexample": 0,
        "partial": 0,
        "timeout": 0,
        "error": 0,
    }
    for _, _, _, result, _, _ in results:
        counts[result] += 1

    print(
        "verified: {verified}, counterexample: {counterexample}, partial: {partial}, timeout: {timeout}, error: {error}".format(
            **counts
        )
    )
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
