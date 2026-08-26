"""
 Copyright 2026 Project LLZK.
 SPDX-License-Identifier: Apache-2.0
"""

import argparse
import csv
import multiprocessing
import os
import pathlib
import re
import signal
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
UNSUPPORTED_PREFIX = "Unsupported:"


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
        return (
            benchmark,
            root_struct,
            "verify",
            "timeout",
            f"{elapsed:.6f}",
            "timeout",
        )

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
    z3_bin: str,
    timeout: float,
) -> tuple[str, str, str, str, str, str]:
    lleq_args = [lleq_bin, "wp", "--struct", root_struct, str(llzk_file)]
    z3_args = [z3_bin, "-in"]
    start = time.perf_counter()
    lleq_proc: subprocess.Popen[str] | None = None
    z3_proc: subprocess.Popen[str] | None = None

    try:
        lleq_proc = subprocess.Popen(
            lleq_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        z3_proc = subprocess.Popen(
            z3_args,
            stdin=lleq_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert lleq_proc.stdout is not None
        lleq_proc.stdout.close()
        z3_stdout, z3_stderr = z3_proc.communicate(timeout=timeout)
        lleq_stdout, lleq_stderr = lleq_proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        for proc in (z3_proc, lleq_proc):
            if proc is None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # On Darwin, process-group delivery can fail with EPERM if the
                # direct child has already exited or the group is otherwise no
                # longer signalable. Fall back to killing the direct child.
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass

        for proc in (z3_proc, lleq_proc):
            if proc is None:
                continue
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is None or pipe.closed:
                    continue
                pipe.close()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

        return (benchmark, root_struct, "wp", "timeout", f"{elapsed:.6f}", "timeout")

    elapsed = time.perf_counter() - start
    if not lleq_stdout.strip() and lleq_stderr.startswith(UNSUPPORTED_PREFIX):
        message = lleq_stderr.strip()[:500]
        return (
            benchmark,
            root_struct,
            "wp",
            "unsupported",
            f"{elapsed:.6f}",
            message,
        )
    if lleq_proc.returncode != 0:
        message = (lleq_stderr or lleq_stdout).strip()[:500]
        return (benchmark, root_struct, "wp", "error", f"{elapsed:.6f}", message)
    if UNSAT_RE.search(z3_stdout):
        return (benchmark, root_struct, "wp", "verified", f"{elapsed:.6f}", "unsat")
    if SAT_RE.search(z3_stdout):
        return (benchmark, root_struct, "wp", "counterexample", f"{elapsed:.6f}", "sat")
    if UNKNOWN_RE.search(z3_stdout):
        return (benchmark, root_struct, "wp", "partial", f"{elapsed:.6f}", "unknown")
    if z3_proc.returncode != 0:
        message = (z3_stderr or z3_stdout).strip()[:500]
        return (benchmark, root_struct, "wp", "error", f"{elapsed:.6f}", message)

    message = (z3_stdout or z3_stderr).strip()[:500]
    return (benchmark, root_struct, "wp", "error", f"{elapsed:.6f}", message)


def run_task(
    benchmark: str,
    llzk_file: pathlib.Path,
    root_struct: str,
    mode: str,
    lleq_bin: str,
    z3_bin: str,
    timeout: float,
) -> tuple[str, str, str, str, str, str]:
    if mode == "verify":
        return run_verify(benchmark, llzk_file, root_struct, lleq_bin, timeout)
    return run_wp(benchmark, llzk_file, root_struct, lleq_bin, z3_bin, timeout)


def run_task_unpack(
    args: tuple[str, pathlib.Path, str, str, str, str, float],
) -> tuple[str, str, str, str, str, str]:
    return run_task(*args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect LLEQ verification data for examples/circom-examples."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="examples/circom-examples",
        help="Directory containing optimized LLZK demo examples.",
    )
    parser.add_argument(
        "--lleq-bin",
        default="./build/tools/lleq/lleq",
        help="Path to the lleq binary.",
    )
    parser.add_argument(
        "--z3-bin",
        default="z3",
        help="Path to the z3 binary.",
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
        default="examples/circom-examples/verification_results.csv",
        help="CSV path for collected results.",
    )
    args = parser.parse_args()

    benchmark_dir = pathlib.Path(args.benchmark_dir).resolve()
    benchmarks = get_benchmarks(benchmark_dir)
    tasks: list[tuple[str, pathlib.Path, str, str, str, str, float]] = []
    for benchmark, llzk_file, root_struct in benchmarks:
        tasks.append(
            (
                benchmark,
                llzk_file,
                root_struct,
                "wp",
                args.lleq_bin,
                args.z3_bin,
                args.timeout,
            )
        )

    results: list[tuple[str, str, str, str, str, str]] = []
    if args.nthreads == 1:
        for i, (
            benchmark,
            llzk_file,
            root_struct,
            mode,
            lleq_bin,
            z3_bin,
            timeout,
        ) in enumerate(tasks):
            print(f"Running {benchmark} ({mode}, {i + 1}/{len(tasks)})")
            result = run_task(
                benchmark, llzk_file, root_struct, mode, lleq_bin, z3_bin, timeout
            )
            results.append(result)
            print(f"Exit condition: {result[3]}, time: {result[4]}")
    else:
        total = len(tasks)
        print(f"Launching {total} verification tasks.")
        next_milestone = 1
        finished = set()
        all_benchmarks = set(next(zip(*benchmarks)))
        print(all_benchmarks)
        with multiprocessing.Pool(args.nthreads) as pool:
            for i, result in enumerate(
                pool.imap_unordered(run_task_unpack, tasks), start=1
            ):
                results.append(result)
                finished.add(result[0])
                pct = i * 100 // total
                if pct >= next_milestone:
                    print(f"Progress: {i}/{total} ({pct}%) complete")
                    print(f"Remaining: {all_benchmarks - finished}")
                    next_milestone += 1

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
        "unsupported": 0,
        "error": 0,
    }
    for _, _, _, result, _, _ in results:
        counts[result] += 1

    print(
        "verified: {verified}, counterexample: {counterexample}, partial: {partial}, timeout: {timeout}, unsupported: {unsupported}, error: {error}".format(
            **counts
        )
    )
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
