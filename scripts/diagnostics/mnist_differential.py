"""Compare the pre-RGB MNIST implementation with another Cirkit revision.

The baseline is exported to a temporary directory with ``git archive``.  Each revision is evaluated
by a fresh worker process, preventing Python module caches or shared GPU state from contaminating
the result.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator


WORKTREE = "WORKTREE"
DEFAULT_BASELINE = "83fc3c5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE)
    parser.add_argument(
        "--candidate-ref",
        default=WORKTREE,
        help="Git ref to test, or WORKTREE for the current checkout (default).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("mnist-differential-results"))
    parser.add_argument("--dataset", choices=("mnist", "synthetic"), default="mnist")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--synthetic-samples", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--structure-samples", type=int, default=2048)
    parser.add_argument("--valid-split", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--noise-scale", type=float, default=2.0)
    parser.add_argument("--rg", default="quad-graph")
    parser.add_argument("--inner-layer", default="cp")
    parser.add_argument("--activation", default="clamp")
    parser.add_argument("--weights-init", default="uniform")
    parser.add_argument("--data-dtype", choices=("uint8", "long"), default="uint8")
    parser.add_argument(
        "--estimated-mode",
        choices=("off", "on", "both"),
        default="both",
        help="Compare random initialization, estimated initialization, or both.",
    )
    parser.add_argument(
        "--adaptive-alpha", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-mixing-weights", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--bpd-atol",
        type=float,
        default=0.01,
        help="Maximum absolute BPD difference allowed at every checkpoint.",
    )
    parser.add_argument(
        "--no-fail-on-diff",
        action="store_true",
        help="Always return exit code 0 while still reporting differences.",
    )
    return parser.parse_args()


def repository_root() -> Path:
    # Resolve from this script before invoking Git.  This also works in environments where Git's
    # dubious-ownership protection requires ``safe.directory`` for the checkout.
    candidate = Path(__file__).resolve().parents[2]
    if not (candidate / ".git").exists():
        script = Path(__file__).resolve()
        raise RuntimeError(f"Could not find the Cirkit Git checkout above {script}")
    return candidate


@contextmanager
def source_tree(repo: Path, revision: str, label: str) -> Iterator[Path]:
    if revision.upper() == WORKTREE:
        yield repo
        return

    with tempfile.TemporaryDirectory(prefix="cirkit-mnist-diff-") as temporary_root:
        path = Path(temporary_root) / label
        path.mkdir()
        print(f"Exporting temporary source tree for {label}: {revision}", flush=True)
        safe_repo = repo.as_posix()
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={safe_repo}",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                revision,
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not export Git revision {revision}:\n"
                f"{result.stderr.decode(errors='replace')}"
            )
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            destination = path.resolve()
            for member in archive.getmembers():
                member_path = (path / member.name).resolve()
                try:
                    member_path.relative_to(destination)
                except ValueError as error:
                    raise RuntimeError(f"Unsafe path in Git archive: {member.name}") from error
            if sys.version_info >= (3, 12):
                archive.extractall(path, filter="data")
            else:
                archive.extractall(path)
        yield path


def worker_arguments(
    args: argparse.Namespace,
    source_root: Path,
    label: str,
    revision_label: str,
    output: Path,
    use_estimated_weights: bool,
) -> list[str]:
    worker = Path(__file__).with_name("_mnist_differential_probe.py").resolve()
    command = [
        sys.executable,
        str(worker),
        "--source-root",
        str(source_root),
        "--label",
        label,
        "--revision-label",
        revision_label,
        "--output",
        str(output),
        "--dataset",
        args.dataset,
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--synthetic-samples",
        str(args.synthetic_samples),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--k",
        str(args.k),
        "--steps",
        str(args.steps),
        "--eval-every",
        str(args.eval_every),
        "--batch-size",
        str(args.batch_size),
        "--eval-samples",
        str(args.eval_samples),
        "--structure-samples",
        str(args.structure_samples),
        "--valid-split",
        str(args.valid_split),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--alpha",
        str(args.alpha),
        "--noise-scale",
        str(args.noise_scale),
        "--rg",
        args.rg,
        "--inner-layer",
        args.inner_layer,
        "--activation",
        args.activation,
        "--weights-init",
        args.weights_init,
        "--data-dtype",
        args.data_dtype,
        "--torch-threads",
        str(args.torch_threads),
        "--adaptive-alpha" if args.adaptive_alpha else "--no-adaptive-alpha",
        "--use-mixing-weights" if args.use_mixing_weights else "--no-use-mixing-weights",
    ]
    if use_estimated_weights:
        command.append("--use-estimated-weights")
    return command


def run_worker(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Probe failed with exit code {return_code}; see {log_path}")


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any], bpd_atol: float
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def exact_check(name: str, baseline_value: Any, candidate_value: Any) -> None:
        checks.append(
            {
                "name": name,
                "equal": baseline_value == candidate_value,
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
        )

    for key in (
        "raw_train_sha256",
        "raw_test_sha256",
        "train_indices_sha256",
        "valid_indices_sha256",
        "structure_indices_sha256",
        "structure_data_sha256",
    ):
        exact_check(f"data.{key}", baseline["data"][key], candidate["data"][key])
    for key in (
        "symbolic_layers",
        "symbolic_layer_types",
        "compiled_layers",
        "compiled_layer_types",
    ):
        exact_check(f"model.{key}", baseline["model"][key], candidate["model"][key])
    exact_check(
        "model.parameter_count",
        baseline["model"]["parameters"]["count"],
        candidate["model"]["parameters"]["count"],
    )
    exact_check(
        "model.partition_parameter_count",
        baseline["model"]["partition_parameters"]["count"],
        candidate["model"]["partition_parameters"]["count"],
    )
    exact_check(
        "model.sharing.shared_storage_pointers",
        baseline["model"]["sharing"]["shared_storage_pointers"],
        candidate["model"]["sharing"]["shared_storage_pointers"],
    )

    baseline_steps = {entry["step"]: entry for entry in baseline["trajectory"]}
    candidate_steps = {entry["step"]: entry for entry in candidate["trajectory"]}
    exact_check("trajectory.steps", sorted(baseline_steps), sorted(candidate_steps))
    trajectory: list[dict[str, Any]] = []
    for step in sorted(baseline_steps.keys() & candidate_steps.keys()):
        baseline_entry = baseline_steps[step]
        candidate_entry = candidate_steps[step]
        bpd_delta = candidate_entry["bpd"] - baseline_entry["bpd"]
        trajectory.append(
            {
                "step": step,
                "baseline_bpd": baseline_entry["bpd"],
                "candidate_bpd": candidate_entry["bpd"],
                "bpd_delta": bpd_delta,
                "absolute_bpd_delta": abs(bpd_delta),
                "baseline_nll": baseline_entry["nll"],
                "candidate_nll": candidate_entry["nll"],
                "nll_delta": candidate_entry["nll"] - baseline_entry["nll"],
                "baseline_log_partition": baseline_entry["log_partition"],
                "candidate_log_partition": candidate_entry["log_partition"],
            }
        )

    maximum_delta = max(
        (entry["absolute_bpd_delta"] for entry in trajectory), default=math.inf
    )
    exact_checks_pass = all(check["equal"] for check in checks)
    passed = exact_checks_pass and maximum_delta <= bpd_atol
    return {
        "passed": passed,
        "bpd_atol": bpd_atol,
        "maximum_absolute_bpd_delta": maximum_delta,
        "exact_checks": checks,
        "trajectory": trajectory,
    }


def print_comparison(scenario: str, comparison: dict[str, Any]) -> None:
    print("", flush=True)
    print(f"=== Differential result: {scenario} ===", flush=True)
    print(" step | baseline BPD | candidate BPD | delta", flush=True)
    print("------+--------------+---------------+-------------", flush=True)
    for entry in comparison["trajectory"]:
        print(
            f"{entry['step']:>5} | {entry['baseline_bpd']:>12.8f} | "
            f"{entry['candidate_bpd']:>13.8f} | {entry['bpd_delta']:>+11.8f}",
            flush=True,
        )
    failed_exact = [check["name"] for check in comparison["exact_checks"] if not check["equal"]]
    if failed_exact:
        print(f"Structural/data differences: {', '.join(failed_exact)}", flush=True)
    status = "PASS" if comparison["passed"] else "DIFF"
    print(
        f"{status}: max |delta BPD|={comparison['maximum_absolute_bpd_delta']:.8f} "
        f"(tolerance={comparison['bpd_atol']:.8f})",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    repo = repository_root()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "random_init": False,
        "estimated_init": True,
    }
    if args.estimated_mode != "both":
        selected = args.estimated_mode == "on"
        scenarios = {"estimated_init" if selected else "random_init": selected}

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(args.seed)
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    comparisons: dict[str, Any] = {}
    reports: dict[str, Any] = {}

    with ExitStack() as stack:
        baseline_root = stack.enter_context(source_tree(repo, args.baseline_ref, "baseline"))
        candidate_root = stack.enter_context(source_tree(repo, args.candidate_ref, "candidate"))
        for scenario, use_estimated_weights in scenarios.items():
            scenario_reports: dict[str, Any] = {}
            for label, source_root, revision_label in (
                ("baseline", baseline_root, args.baseline_ref),
                ("candidate", candidate_root, args.candidate_ref),
            ):
                output = args.output_dir / f"{scenario}-{label}.json"
                log_path = args.output_dir / f"{scenario}-{label}.log"
                command = worker_arguments(
                    args,
                    source_root,
                    f"{scenario}-{label}",
                    revision_label,
                    output,
                    use_estimated_weights,
                )
                run_worker(command, log_path, environment)
                scenario_reports[label] = load_report(output)
            comparison = compare_reports(
                scenario_reports["baseline"], scenario_reports["candidate"], args.bpd_atol
            )
            print_comparison(scenario, comparison)
            reports[scenario] = scenario_reports
            comparisons[scenario] = comparison

    summary = {
        "baseline_ref": args.baseline_ref,
        "candidate_ref": args.candidate_ref,
        "passed": all(comparison["passed"] for comparison in comparisons.values()),
        "comparisons": comparisons,
        "reports": reports,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nFull report: {summary_path}", flush=True)
    if summary["passed"] or args.no_fail_on_diff:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
