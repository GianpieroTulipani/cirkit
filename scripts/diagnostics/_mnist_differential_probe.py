"""Run one deterministic MNIST training probe against a selected Cirkit source tree.

This is the worker used by ``mnist_differential.py``.  It intentionally lives outside the
``cirkit`` package so that the same file can import either the historical or the current source
tree in a fresh Python process.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


NUM_DIMENSIONS = 28 * 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--revision-label", default=None)
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument(
        "--structure-samples",
        type=int,
        default=2048,
        help="Samples used to build/estimate the structure; 0 means the whole train split.",
    )
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
    parser.add_argument("--use-estimated-weights", action="store_true")
    parser.add_argument(
        "--adaptive-alpha", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-mixing-weights", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "WORKTREE"


def configure_reproducibility(seed: int, torch_threads: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(torch_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_raw_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    if args.dataset == "mnist":
        from torchvision import datasets

        train = datasets.MNIST(root=args.dataset_root, train=True, download=True).data
        test = datasets.MNIST(root=args.dataset_root, train=False, download=True).data
        return train.reshape(-1, NUM_DIMENSIONS), test.reshape(-1, NUM_DIMENSIONS)

    if args.synthetic_samples < 32:
        raise ValueError("--synthetic-samples must be at least 32")
    generator = torch.Generator().manual_seed(args.seed + 10_000)
    train = torch.randint(
        0,
        256,
        (args.synthetic_samples, NUM_DIMENSIONS),
        dtype=torch.uint8,
        generator=generator,
    )
    test = torch.randint(
        0,
        256,
        (max(32, args.synthetic_samples // 5), NUM_DIMENSIONS),
        dtype=torch.uint8,
        generator=generator,
    )
    train[0, :2] = torch.tensor([0, 255], dtype=torch.uint8)
    test[0, :2] = torch.tensor([0, 255], dtype=torch.uint8)
    return train, test


def split_data(
    raw_train: torch.Tensor, args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_valid = int(len(raw_train) * args.valid_split)
    if n_valid <= 0 or n_valid >= len(raw_train):
        raise ValueError("--valid-split produces an empty train or validation split")
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(raw_train), generator=generator)
    train_indices = permutation[:-n_valid]
    valid_indices = permutation[-n_valid:]
    train = raw_train.index_select(0, train_indices)
    valid = raw_train.index_select(0, valid_indices)
    dtype = torch.uint8 if args.data_dtype == "uint8" else torch.long
    return train.to(dtype), valid.to(dtype), train_indices, valid_indices


def select_structure_data(
    train: torch.Tensor, args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.structure_samples <= 0 or args.structure_samples >= len(train):
        indices = torch.arange(len(train))
    else:
        generator = torch.Generator().manual_seed(args.seed + 3)
        indices = torch.randperm(len(train), generator=generator)[: args.structure_samples]
    return train.index_select(0, indices), indices


def import_target_cirkit(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root))

    import cirkit
    import cirkit.symbolic.functional as sf
    from cirkit.backend.torch.layers import TorchInputLayer
    from cirkit.pipeline import PipelineContext
    from cirkit.templates.learn_spn_optimized import LearnSPN

    imported_path = Path(cirkit.__file__).resolve()
    try:
        imported_path.relative_to(source_root)
    except ValueError as error:
        raise RuntimeError(
            f"Imported Cirkit from {imported_path}, expected a module below {source_root}"
        ) from error

    return {
        "cirkit": cirkit,
        "sf": sf,
        "TorchInputLayer": TorchInputLayer,
        "PipelineContext": PipelineContext,
        "LearnSPN": LearnSPN,
        "imported_path": imported_path,
    }


def learner_kwargs(learn_spn_cls: type, args: argparse.Namespace) -> dict[str, Any]:
    signature = inspect.signature(learn_spn_cls.__init__)
    supported = signature.parameters
    device = resolve_device(args.device)
    candidates: dict[str, Any] = {
        "alpha": args.alpha,
        "noise_scale": args.noise_scale,
        "use_miwae": False,
        "data_format": "image",
        "image_shape": (1, 28, 28),
        # The historical learner uses ``device`` for row indices during parameter estimation.
        "device": device,
        "weight_dir": None,
        "adaptive_alpha": args.adaptive_alpha,
        "input_sharing": "none",
        "num_categories": 256,
    }
    return {name: value for name, value in candidates.items() if name in supported}


def parameter_summary(module: torch.nn.Module) -> dict[str, Any]:
    parameters = list(module.parameters())
    if not parameters:
        return {"count": 0, "tensors": 0, "finite": True, "min": None, "max": None}
    finite = all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
    return {
        "count": sum(parameter.numel() for parameter in parameters),
        "tensors": len(parameters),
        "finite": finite,
        "min": min(float(parameter.detach().min().cpu()) for parameter in parameters),
        "max": max(float(parameter.detach().max().cpu()) for parameter in parameters),
    }


@torch.inference_mode()
def evaluate(
    circuit: torch.nn.Module,
    partition: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    circuit.eval()
    partition.eval()
    score_sum = 0.0
    nll_sum = 0.0
    count = 0
    partition_value = float(partition().detach().mean().cpu())
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size].to(device=device, dtype=torch.long)
        scores = circuit(batch).flatten()
        log_likelihoods = scores - partition()
        score_sum += float(scores.sum().cpu())
        nll_sum += float((-log_likelihoods).sum().cpu())
        count += batch.size(0)
    average_nll = nll_sum / count
    return {
        "raw_score": score_sum / count,
        "log_partition": partition_value,
        "nll": average_nll,
        "bpd": average_nll / (NUM_DIMENSIONS * math.log(2.0)),
    }


def shared_parameter_summary(
    circuit: torch.nn.Module, partition: torch.nn.Module
) -> dict[str, int]:
    circuit_parameters = list(circuit.parameters())
    partition_parameters = list(partition.parameters())
    circuit_ids = {id(parameter) for parameter in circuit_parameters}
    partition_ids = {id(parameter) for parameter in partition_parameters}
    circuit_ptrs = {parameter.data_ptr() for parameter in circuit_parameters}
    partition_ptrs = {parameter.data_ptr() for parameter in partition_parameters}
    return {
        "circuit_tensors": len(circuit_parameters),
        "partition_tensors": len(partition_parameters),
        "shared_python_objects": len(circuit_ids & partition_ids),
        "shared_storage_pointers": len(circuit_ptrs & partition_ptrs),
    }


def train_probe(
    circuit: torch.nn.Module,
    partition: torch.nn.Module,
    torch_input_layer_cls: type,
    train: torch.Tensor,
    evaluation_data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(
        circuit.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    clamp_parameters = [
        parameter
        for layer in circuit.layers
        if not isinstance(layer, torch_input_layer_cls)
        for parameter in layer.parameters()
    ]
    order_generator = torch.Generator().manual_seed(args.seed + 20_000)
    order = torch.randperm(len(train), generator=order_generator)
    cursor = 0

    trajectory: list[dict[str, Any]] = []

    def record(step: int) -> None:
        metrics = evaluate(circuit, partition, evaluation_data, device, args.batch_size)
        entry = {"step": step, **metrics, "parameters": parameter_summary(circuit)}
        trajectory.append(entry)
        print(
            f"[{args.label}] step={step:>5} nll={metrics['nll']:.6f} "
            f"bpd={metrics['bpd']:.8f} logZ={metrics['log_partition']:.6f}",
            flush=True,
        )

    record(0)
    circuit.train()
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(train), generator=order_generator)
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch = train.index_select(0, indices).to(device=device, dtype=torch.long)

        log_likelihoods = (circuit(batch) - partition()).flatten()
        loss = -log_likelihoods.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.activation == "clamp":
            with torch.no_grad():
                lower_bound = float(np.sqrt(np.finfo(np.float32).tiny))
                for parameter in clamp_parameters:
                    parameter.clamp_(min=lower_bound)

        if step % args.eval_every == 0 or step == args.steps:
            record(step)
            circuit.train()
    return trajectory


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    source_root = args.source_root.resolve()
    target = import_target_cirkit(source_root)
    configure_reproducibility(args.seed, args.torch_threads)
    device = resolve_device(args.device)

    print(f"[{args.label}] Cirkit source: {target['imported_path']}", flush=True)
    print(f"[{args.label}] training device: {device}", flush=True)
    raw_train, raw_test = load_raw_data(args)
    train, valid, train_indices, valid_indices = split_data(raw_train, args)
    test_dtype = torch.uint8 if args.data_dtype == "uint8" else torch.long
    test = raw_test.to(test_dtype)
    structure_data, structure_indices = select_structure_data(train, args)
    evaluation_data = test[: min(args.eval_samples, len(test))]

    # Data loading and model construction consume random numbers.  Reset immediately before the
    # learner so both source trees start from exactly the same RNG state.
    configure_reproducibility(args.seed, args.torch_threads)
    LearnSPN = target["LearnSPN"]
    kwargs = learner_kwargs(LearnSPN, args)
    learner = LearnSPN(**kwargs)
    structure_for_build = structure_data.to(device)

    print(
        f"[{args.label}] building QG-CP K={args.k}, "
        f"estimated={args.use_estimated_weights}, structure_samples={len(structure_data)}",
        flush=True,
    )
    symbolic_circuit = learner.learn_spn(
        structure_for_build,
        input_layer="categorical",
        region_graph=args.rg,
        activation=args.activation,
        weights_init=args.weights_init,
        sum_product_layer=args.inner_layer,
        num_input_units=args.k,
        num_sum_units=args.k,
        use_mixing_weights=args.use_mixing_weights,
        use_estimated_weights=args.use_estimated_weights,
    )
    symbolic_partition = target["sf"].integrate(symbolic_circuit)
    symbolic_layers = list(symbolic_circuit.layers)

    context = target["PipelineContext"](
        backend="torch", semiring="lse-sum", fold=True, optimize=True
    )
    circuit = context.compile(symbolic_circuit).to(device)
    partition = context.compile(symbolic_partition).to(device)

    trajectory = train_probe(
        circuit,
        partition,
        target["TorchInputLayer"],
        train,
        evaluation_data,
        args,
        device,
    )
    report: dict[str, Any] = {
        "label": args.label,
        "revision": args.revision_label or git_revision(source_root),
        "source_root": str(source_root),
        "imported_cirkit": str(target["imported_path"]),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"source_root", "output", "label", "revision_label"}
        },
        "data": {
            "raw_train_shape": list(raw_train.shape),
            "train_shape": list(train.shape),
            "valid_shape": list(valid.shape),
            "test_shape": list(test.shape),
            "dtype": str(train.dtype),
            "minimum": int(train.min()),
            "maximum": int(train.max()),
            "raw_train_sha256": sha256_tensor(raw_train),
            "raw_test_sha256": sha256_tensor(raw_test),
            "train_indices_sha256": sha256_tensor(train_indices),
            "valid_indices_sha256": sha256_tensor(valid_indices),
            "structure_indices_sha256": sha256_tensor(structure_indices),
            "structure_data_sha256": sha256_tensor(structure_data),
        },
        "model": {
            "symbolic_layers": len(symbolic_layers),
            "symbolic_layer_types": dict(
                sorted(Counter(type(layer).__name__ for layer in symbolic_layers).items())
            ),
            "compiled_layers": len(circuit.layers),
            "compiled_layer_types": dict(
                sorted(Counter(type(layer).__name__ for layer in circuit.layers).items())
            ),
            "parameters": parameter_summary(circuit),
            "partition_parameters": parameter_summary(partition),
            "sharing": shared_parameter_summary(circuit, partition),
        },
        "trajectory": trajectory,
        "duration_seconds": time.perf_counter() - started_at,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[{args.label}] report written to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
