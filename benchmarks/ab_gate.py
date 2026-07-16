#!/usr/bin/env python
"""Run the Lorenz performance gate against a base Git ref."""
import argparse
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
BENCHMARK = Path(__file__).resolve().parent / "lorenz_mean_runtime.py"
EXPECTED_CONFIGS = {"fixed", "adaptive"}
RESULT_RE = re.compile(
    r"^RESULT\s+(\S+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)$"
)
Z_THRESHOLD = 3.0
MAX_RELATIVE_STD = 0.05
BACKENDS = {
    "numba-cuda": ("numba.cuda", "numba-cuda"),
    "mlir": ("numba_cuda_mlir", "mlir"),
}


def installed_backends():
    """Return the installed CUDA backends."""
    return {
        label
        for label, (module, _) in BACKENDS.items()
        if importlib.util.find_spec(module) is not None
    }


def resolve_backends(requested):
    """Validate and return the requested backends."""
    installed = installed_backends()
    if requested is None:
        selected = list(BACKENDS)
    else:
        selected = [label.strip() for label in requested.split(",")]
    unknown = set(selected) - set(BACKENDS)
    missing = set(selected) - installed
    if unknown:
        raise SystemExit(
            f"unknown backend: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise SystemExit(
            f"backend not installed: {', '.join(sorted(missing))}"
        )
    if not selected:
        raise SystemExit("no CUDA backend selected")
    return selected


def git_output(*args):
    """Run Git and return stripped stdout."""
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def add_base_worktree(base_ref):
    """Create a detached worktree for the base ref."""
    tree = Path(tempfile.mkdtemp(prefix="cubie_base_"))
    subprocess.run(
        [
            "git", "-C", str(REPO), "worktree", "add", "--detach",
            str(tree), base_ref,
        ],
        check=True,
    )
    return tree


def remove_worktree(tree):
    """Remove a temporary worktree."""
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "remove", "--force",
         str(tree)],
        check=False,
    )


def run_side(tree, backend, cache_dir, grid_dir, args, references=None):
    """Run one benchmark side and return its mean and standard deviation."""
    command = [
        sys.executable,
        str(BENCHMARK),
        "--repeats", str(args.repeats),
        "--grid-cache", str(grid_dir),
        "--no-clear-cache",
    ]
    if args.n_runs is not None:
        command.append(str(args.n_runs))
    if references is not None:
        for key in sorted(EXPECTED_CONFIGS):
            mean, std = references[key]
            command.extend(
                [f"--ref-{key}", str(mean), str(std)]
            )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(tree) / "src")
    env["CUBIE_CUDA_BACKEND"] = BACKENDS[backend][1]
    env["CUBIE_CACHE_DIR"] = str(cache_dir)
    process = subprocess.run(
        command,
        env=env,
        cwd=str(tree),
        capture_output=True,
        text=True,
    )
    if process.returncode:
        sys.stderr.write(process.stdout)
        sys.stderr.write(process.stderr)
        raise SystemExit(
            f"{backend} benchmark failed in {tree} "
            f"with exit code {process.returncode}"
        )

    results = {}
    for line in process.stdout.splitlines():
        match = RESULT_RE.match(line.strip())
        if not match:
            continue
        key = match.group(1)
        if key in results:
            raise SystemExit(f"duplicate RESULT for {backend}/{key}")
        mean = float(match.group(2))
        std = float(match.group(3))
        if not math.isfinite(mean) or mean <= 0.0:
            raise SystemExit(f"invalid mean for {backend}/{key}: {mean}")
        if not math.isfinite(std) or std < 0.0:
            raise SystemExit(f"invalid std for {backend}/{key}: {std}")
        results[key] = (mean, std)

    if set(results) != EXPECTED_CONFIGS:
        missing = EXPECTED_CONFIGS - set(results)
        extra = set(results) - EXPECTED_CONFIGS
        raise SystemExit(
            f"invalid RESULT set for {backend}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return results


def compare(backend, base, branch, repeats):
    """Compare one backend and return rows plus the gate result."""
    rows = []
    failed = False
    for key in sorted(EXPECTED_CONFIGS):
        base_mean, base_std = base[key]
        branch_mean, branch_std = branch[key]
        contaminated = (
            base_std > MAX_RELATIVE_STD * base_mean
            or branch_std > MAX_RELATIVE_STD * branch_mean
        )
        denominator = math.sqrt(
            (base_std**2 + branch_std**2) / repeats
        )
        if denominator == 0.0:
            z_value = 0.0 if branch_mean == base_mean else math.inf
        else:
            z_value = (branch_mean - base_mean) / denominator
        if contaminated:
            verdict = "CONTAMINATED"
            failed = True
        elif z_value >= Z_THRESHOLD:
            verdict = "REGRESSION"
            failed = True
        elif z_value <= -Z_THRESHOLD:
            verdict = "improvement"
        else:
            verdict = "ok"
        rows.append(
            (backend, key, base_mean, base_std, branch_mean,
             branch_std, z_value, verdict)
        )
    return rows, failed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="origin/main")
    parser.add_argument("--backends")
    parser.add_argument("--n-runs", type=int)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.n_runs is not None and args.n_runs < 1:
        parser.error("--n-runs must be positive")
    return args


def main():
    """Run the A/B performance gate."""
    args = parse_args()
    backends = resolve_backends(args.backends)
    base_sha = git_output("rev-parse", args.main)
    branch_sha = git_output("rev-parse", "HEAD")
    print(f"A {args.main} {base_sha}")
    print(f"B HEAD {branch_sha}")

    base_tree = add_base_worktree(base_sha)
    temporary_root = Path(tempfile.mkdtemp(prefix="cubie_ab_gate_"))
    failed = False
    try:
        for backend in backends:
            grid_dir = temporary_root / "grid"
            base = run_side(
                base_tree, backend, temporary_root / f"A_{backend}",
                grid_dir, args,
            )
            branch = run_side(
                REPO, backend, temporary_root / f"B_{backend}",
                grid_dir, args, references=base,
            )
            rows, backend_failed = compare(
                backend, base, branch, args.repeats
            )
            failed = failed or backend_failed
            for row in rows:
                label, key, a_mean, a_std, b_mean, b_std, z, verdict = row
                print(
                    f"{label:<11}{key:<10}"
                    f"A {a_mean:9.3f} +/- {a_std:7.3f}  "
                    f"B {b_mean:9.3f} +/- {b_std:7.3f}  "
                    f"z {z:+7.2f}  {verdict}"
                )
    finally:
        if args.keep:
            print(f"Kept base worktree: {base_tree}")
            print(f"Kept caches: {temporary_root}")
        else:
            remove_worktree(base_tree)
            shutil.rmtree(temporary_root, ignore_errors=True)

    print(f"GATE: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
