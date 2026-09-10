"""Train and evaluate InformedLearnSPN using the existing dataset_eval workflow.

Run from the repository root, for example:
    python -m cirkit.dataset_eval_informed --dataset mnist --k 32 --use-estimated-weights

For local data, use --dataset tensor --root PATH with train.pt, valid.pt and test.pt.
Flat tensors also require --image-shape C H W. The dataset loader, training loop,
checkpoint selection and NLL/BPD evaluation are reused from dataset_eval unchanged.
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import cirkit.symbolic.functional as sf
from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.dataset_eval import (
    evaluate_circuit,
    limit_samples,
    load_discrete_image_dataset,
    logger,
    train_circuit,
    wandb,
)
from cirkit.pipeline import PipelineContext
from cirkit.templates.learn_spn_informed import InformedLearnSPN


def main(argv: list[str] | None = None) -> None:
    """Load images, learn their circuit structure, compile, train and evaluate."""
    parser = argparse.ArgumentParser(
        description="Train an informed LearnSPN circuit on discrete image datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=[
            "mnist",
            "fashion-mnist",
            "cifar",
            "cifar10",
            "celeba",
            "imagenet32",
            "imagenet64",
            "tensor",
        ],
        help="dataset to train on",
    )
    parser.add_argument("--k", type=int, default=512, help="num units per layer")
    parser.add_argument(
        "--activation",
        type=str,
        default="clamp",
        choices=["clamp", "softmax", "softplus", "sigmoid", "none"],
        help="activation function for sum units",
    )
    parser.add_argument(
        "--weights-init", type=str, default="uniform", choices=["uniform", "normal", "dirichlet"]
    )
    parser.add_argument(
        "--root", type=str, default="datasets", help="dataset root or local tensor directory"
    )
    parser.add_argument("--image-shape", type=int, nargs=3, metavar=("C", "H", "W"), default=None)
    parser.add_argument("--ycc", type=str, default="none", choices=["none", "lossy", "lossless"])
    parser.add_argument(
        "--input-sharing",
        type=str,
        default="none",
        choices=["none", "full"],
        help="'full' shares input parameters using the quad-spn input factory",
    )

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--T_0", type=int, default=1, help="T_0 for cosine annealing")
    parser.add_argument("--eta-min", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="limit training samples after loading"
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=None, help="limit validation samples after loading"
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=None, help="limit test samples after loading"
    )
    parser.add_argument(
        "--structure-samples",
        type=int,
        default=None,
        help="number of training samples used to build and estimate the initial circuit",
    )
    parser.add_argument(
        "--validation-steps", type=int, default=250, help="validation every n steps"
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--valid-split", type=float, default=0.05)
    parser.add_argument(
        "--use-scheduler", action="store_true", help="use cosine annealing scheduler"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="best_informed_circuit.pt")
    parser.add_argument("--wandb", action="store_true", help="log to wandb")
    parser.add_argument(
        "--project", type=str, default="pc_informed_dataset_eval", help="wandb project name"
    )
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--noise-scale", type=float, default=2.0)
    parser.add_argument("--use-miwae", action="store_true", help="use MIWAE for LearnSPN")
    parser.add_argument(
        "--adaptive-alpha", action="store_true", help="use adaptive alpha for LearnSPN"
    )
    parser.add_argument(
        "--use-mixing-weights", action="store_true", help="use mixing weights for LearnSPN"
    )
    parser.add_argument(
        "--use-estimated-weights", action="store_true", help="use estimated weights for LearnSPN"
    )
    parser.add_argument(
        "--estimation-device",
        type=str,
        default=None,
        help="device used by LearnSPN while estimating weights, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--miwae-batch-size",
        type=int,
        default=1024,
        help="batch size for MIWAE feature extraction during LearnSPN weight estimation",
    )
    parser.add_argument(
        "--min-instances", type=int, default=1000, help="stop splitting at this sample count"
    )
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument(
        "--local-radius", type=int, default=3, help="L1 radius for local categorical MI"
    )
    parser.add_argument("--mi-quantile", type=float, default=0.6)
    parser.add_argument(
        "--mi-threshold",
        type=float,
        default=None,
        help="absolute MI threshold in nats; overrides the quantile",
    )
    parser.add_argument("--mi-alpha", type=float, default=0.01, help="smoothing for MI only")
    parser.add_argument("--pair-batch-size", type=int, default=32, help="MI pairs per batch")
    parser.add_argument("--sample-batch-size", type=int, default=1024, help="MI samples per batch")
    parser.add_argument("--force-root-mixture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--num-input-units", type=int, default=None, help="input width; defaults to --k"
    )
    args = parser.parse_args(argv)
    if min(args.batch_size, args.max_epochs, args.validation_steps, args.patience, args.k) < 1:
        parser.error("batch-size, max-epochs, validation-steps, patience and k must be positive")
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    for kk, vv in vars(args).items():
        print(f"  {kk}: {vv}")
    print(f"  device: {device}")
    print("=" * 70)

    if args.wandb:
        if wandb is None:
            raise ImportError("Install wandb or run without --wandb")
        wandb.login()
        run = wandb.init(project=args.project, config=vars(args))

    x_train, x_val, x_test, image_shape = load_discrete_image_dataset(args)
    x_train = limit_samples(x_train, args.max_train_samples, args.seed)
    x_val = limit_samples(x_val, args.max_val_samples, args.seed + 1)
    x_test = limit_samples(x_test, args.max_test_samples, args.seed + 2)
    logger.info(
        f"Loaded {args.dataset}: train={tuple(x_train.shape)}, "
        f"val={tuple(x_val.shape)}, test={tuple(x_test.shape)}"
    )
    logger.info(
        f"Using image_shape={image_shape}, ycc={args.ycc}, input_sharing={args.input_sharing}"
    )

    train_loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=args.batch_size, shuffle=False)

    weight_dir = os.path.join(os.getcwd(), "best_categorical_miwae.pt")

    if args.use_miwae and image_shape[0] != 1:
        raise ValueError(
            "MIWAE support in this script is currently limited to single-channel images"
        )

    logger.info("Using InformedLearnSPN")
    spn_learner = InformedLearnSPN(
        alpha=args.alpha,
        noise_scale=args.noise_scale,
        use_miwae=args.use_miwae,
        data_format="image",
        image_shape=image_shape,
        device=device,
        weight_dir=weight_dir,
        adaptive_alpha=args.adaptive_alpha,
        input_sharing=args.input_sharing,
        num_categories=256,
        estimation_device=args.estimation_device,
        miwae_batch_size=args.miwae_batch_size,
        seed=args.seed,
        min_instances=args.min_instances,
        min_cluster_size=args.min_cluster_size,
        num_clusters=args.num_clusters,
        max_depth=args.max_depth,
        local_radius=args.local_radius,
        mi_quantile=args.mi_quantile,
        mi_threshold=args.mi_threshold,
        mi_alpha=args.mi_alpha,
        force_root_mixture=args.force_root_mixture,
    )
    structure_data = limit_samples(x_train, args.structure_samples, args.seed + 3)
    logger.info(f"Using {len(structure_data)} samples to build/estimate the circuit")

    estimation_device = torch.device(args.estimation_device) if args.estimation_device else device
    structure_data_for_build = structure_data.to(device=estimation_device, dtype=torch.long)

    symbolic_circuit = spn_learner.learn_spn(
        structure_data_for_build,
        input_layer="categorical",
        activation=args.activation,
        weights_init=args.weights_init,
        num_input_units=args.num_input_units if args.num_input_units is not None else args.k,
        num_sum_units=args.k,
        use_mixing_weights=args.use_mixing_weights,
        use_estimated_weights=args.use_estimated_weights,
        pair_batch_size=args.pair_batch_size,
        sample_batch_size=args.sample_batch_size,
    )
    del structure_data_for_build, structure_data, spn_learner

    symbolic_partition_function = sf.integrate(symbolic_circuit)

    logger.info(f"Circuit built with {len(list(symbolic_circuit.layers))} layers")

    max_train_steps = int(len(train_loader) * args.max_epochs)

    ctx: PipelineContext = PipelineContext(
        backend="torch", semiring="lse-sum", fold=True, optimize=True
    )

    circuit = ctx.compile(symbolic_circuit).to(device)
    circuit_partition_function = ctx.compile(symbolic_partition_function).to(device)

    logger.info(f"Number of parameters: {sum(p.numel() for p in circuit.parameters())}")

    sum_params = [
        p
        for layer in circuit.layers
        if not isinstance(layer, TorchInputLayer)
        for p in layer.parameters()
    ]

    num_dimensions = int(np.prod(image_shape))
    train_circuit(
        circuit,
        circuit_partition_function,
        train_loader,
        val_loader,
        sum_params=sum_params,
        max_train_steps=max_train_steps,
        lr=args.lr,
        T_0=args.T_0,
        eta_min=args.eta_min,
        weight_decay=args.weight_decay,
        validation_steps=args.validation_steps,
        delta=args.min_delta,
        patience=args.patience,
        num_dimensions=num_dimensions,
        device=device,
        save_path=args.save_path,
        log_to_wandb=args.wandb,
        activation=args.activation,
        use_scheduler=args.use_scheduler,
    )

    torch.cuda.empty_cache()
    gc.collect()

    circuit.load_state_dict(torch.load(args.save_path, map_location=device))

    evaluate_circuit(
        circuit,
        circuit_partition_function,
        test_loader,
        device=device,
        num_dimensions=num_dimensions,
        log_to_wandb=args.wandb,
    )
    if args.wandb:
        run.finish()


if __name__ == "__main__":
    main()
