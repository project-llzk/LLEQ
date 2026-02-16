import argparse
import csv
import multiprocessing
import os
import pathlib
import re
import subprocess
import time

ENTRYPOINT_RE = re.compile(r'module attributes {.*llzk\.main\s+=\s+\!struct.type<@([A-Za-z0-9_]+).*>}', re.ASCII)
def get_llzk_files(benchmark_dir: str) -> list[tuple[str, str, str]]:
    root = pathlib.Path(benchmark_dir)

    benchmark_files: list[pathlib.Path] = []
    benchmarks = []

    for benchmark in os.listdir(root):
        name = benchmark.removesuffix('_llzk')
        file = root / benchmark / pathlib.Path(name + '.llzk')
        if file.exists: benchmark_files.append(file)

    for filename in benchmark_files:
        for line in filename.open():
            if groups := ENTRYPOINT_RE.match(line):
                benchmarks.append((str(filename), os.path.basename(filename), groups.group(1)))

    return benchmarks

def run_task(benchmark_name: str, args: list[str], timeout: int) -> tuple[str, str, str, str]:
    start = time.perf_counter()

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        elapsed = time.perf_counter() - start
        if proc.returncode == 0:
            return (benchmark_name, 'success', f'{elapsed:.6f}', '')
        error_msg = proc.stderr.strip()[:500]
        return (benchmark_name, 'error', f'{elapsed:.6f}', error_msg)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return (benchmark_name, 'timeout', f'{elapsed:.6f}', 'timeout')
    

def build_tasks(benchmarks: list[tuple[str, str, str]], llzk_bin: str, lleq_bin: str, timeout: float) -> list[tuple[str, list[str], float]]:
    align_tasks: list[tuple[str, list[str], float]] = []
    verify_tasks: list[tuple[str, list[str], float]] = []
    for file, name, main in benchmarks:
        align_args = [llzk_bin, file, f'--llzk-compute-constrain-to-product=root-struct={main}']
        verify_args = [lleq_bin, file, '--dump-store']
        align_tasks.append((f'{name}:align', align_args, timeout))
        verify_tasks.append((f'{name}:verify', verify_args, timeout))
    return align_tasks + verify_tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Try verifying circom benchmarks and collect timing results')
    parser.add_argument('--benchmark-dir')
    parser.add_argument('--timeout', type=float, default=2)
    parser.add_argument('--llzk-bin')
    parser.add_argument('--lleq-bin', default='./build/tools/lleq/lleq')
    parser.add_argument('--nthreads', type=int, default=multiprocessing.cpu_count())

    args = parser.parse_args()

    benchmarks = get_llzk_files(args.benchmark_dir)
    tasks = build_tasks(benchmarks, args.llzk_bin, args.lleq_bin, args.timeout)
    with multiprocessing.Pool(args.nthreads) as p:
        results = p.starmap(run_task, tasks)


    success_cnt = 0
    error_cnt = 0
    timeout_cnt = 0
    results.sort()
    for _, cause, _, _ in results:
        success_cnt += 1 if cause == "success" else 0
        error_cnt += 1 if cause == "error" else 0
        timeout_cnt += 1 if cause == "timeout" else 0

    output_path = f"circom_benchmarks_results.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Benchmark", "Result", "Time Seconds", "Error Message"])
        writer.writerows(results)
    print(f"success: {success_cnt}, errored: {error_cnt}, timeout: {timeout_cnt}")


