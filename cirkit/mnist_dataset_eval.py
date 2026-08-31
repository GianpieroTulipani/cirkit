"""Legacy MNIST evaluation protocol using the current optimized LearnSPN.

This script intentionally preserves the data preparation, training loop, validation,
checkpointing, and BPD computation from the pre-RGB MNIST evaluator.  Only the learner comes
from the current working tree, making this useful for separating learner changes from changes
introduced by the generic dataset evaluation script.
"""

import argparse
import gc
import inspect
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

import cirkit.symbolic.functional as sf
from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.pipeline import PipelineContext
from cirkit.templates.learn_spn_optimized import LearnSPN


class CircuitNLL(nn.Module):
    def __init__(self, circuit, partition):
        super().__init__()
        self.circuit = circuit
        self.partition = partition

    def forward(self, x):
        return self.circuit(x) - self.partition()


def train_circuit(
    circuit,
    partition_function,
    train_loader,
    val_loader,
    sum_params,
    max_train_steps,
    lr,
    t_0,
    eta_min,
    weight_decay,
    device,
    save_path,
    validation_steps,
    delta,
    patience,
    log_to_wandb=True,
    activation=None,
    use_scheduler=False,
):
    optimizer = optim.Adam(circuit.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=1, eta_min=eta_min
        )

    best_val_nll = float("inf")
    epochs_no_improve = 0
    total_steps = 0
    avg_train_nll = None
    avg_val_nll = None
    bpd_train = None
    bpd_val = None
    saved_best = False

    print(f"Starting training for a maximum of {max_train_steps} steps.\n")

    stop = False
    while total_steps < max_train_steps and not stop:
        circuit.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in tqdm(train_loader, desc="[Train]", leave=False):
            batch = batch.to(device)

            log_liks = (circuit(batch) - partition_function()).flatten()
            loss = -log_liks.mean()

            train_loss_sum += loss.item() * batch.size(0)
            train_count += batch.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            if activation == "clamp":
                with torch.no_grad():
                    for parameter in sum_params:
                        parameter.data.clamp_(
                            min=float(np.sqrt(np.finfo(np.float32).tiny))
                        )

            total_steps += 1

            if total_steps % validation_steps == 0:
                circuit.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.inference_mode():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        log_liks = (circuit(val_batch) - partition_function()).flatten()
                        val_loss_sum += (-log_liks.mean()).item() * val_batch.size(0)
                        val_count += val_batch.size(0)

                avg_train_nll = train_loss_sum / train_count
                avg_val_nll = val_loss_sum / val_count
                bpd_train = avg_train_nll / (28 * 28 * np.log(2.0))
                bpd_val = avg_val_nll / (28 * 28 * np.log(2.0))

                if log_to_wandb:
                    assert wandb is not None
                    wandb.log(
                        {
                            "step": total_steps,
                            "train_nll": avg_train_nll,
                            "train_bpd": bpd_train,
                            "val_nll": avg_val_nll,
                            "val_bpd": bpd_val,
                        }
                    )

                if avg_val_nll - delta <= best_val_nll:
                    best_val_nll = avg_val_nll
                    torch.save(circuit.state_dict(), save_path)
                    saved_best = True
                    epochs_no_improve = 0
                    logger.success(
                        f"New best model at step {total_steps}, "
                        f"Train NLL: {avg_train_nll:.4f}, Train bpd: {bpd_train:.4f}, "
                        f"Val NLL: {best_val_nll:.4f}, Val bpd: {bpd_val:.4f}"
                    )
                else:
                    epochs_no_improve += 1
                    logger.info(
                        f"No improvement at step {total_steps}, "
                        f"count: {epochs_no_improve}/{patience}"
                    )

                if epochs_no_improve >= patience:
                    logger.info("Early stopping triggered.")
                    stop = True
                    break

                circuit.train()

            if total_steps >= max_train_steps:
                stop = True
                break

    if not saved_best:
        torch.save(circuit.state_dict(), save_path)
        logger.info(f"No validation checkpoint was saved; saved current model to {save_path}.")

    if log_to_wandb:
        assert wandb is not None
        final_logs = {"saved_best_checkpoint": saved_best}
        if avg_train_nll is not None:
            final_logs["final_train_nll"] = avg_train_nll
            final_logs["final_train_bpd"] = bpd_train
        if avg_val_nll is not None:
            final_logs["final_val_nll"] = avg_val_nll
            final_logs["final_val_bpd"] = bpd_val
            final_logs["best_val_nll"] = best_val_nll
        wandb.log(final_logs)


@torch.inference_mode()
def evaluate_circuit(circuit, circuit_partition_function, test_loader, device, log_to_wandb=True):
    circuit.eval()

    test_nll_sum = 0.0
    test_count = 0

    for batch in tqdm(test_loader, desc="[Test]", leave=False):
        batch = batch.to(device)
        log_liks = circuit(batch) - circuit_partition_function()
        loss = -log_liks.mean()
        test_nll_sum += loss.item() * batch.size(0)
        test_count += batch.size(0)

    avg_test_nll = test_nll_sum / test_count
    avg_test_bpd = avg_test_nll / (28 * 28 * np.log(2.0))

    logger.info(f"Test NLL: {avg_test_nll:.4f} | bpd: {avg_test_bpd:.4f}")

    if log_to_wandb:
        assert wandb is not None
        wandb.log({"test_nll": avg_test_nll, "test_bpd": avg_test_bpd})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the legacy MNIST protocol with the current optimized LearnSPN"
    )
    parser.add_argument(
        "--rg",
        type=str,
        default="quad-tree-2",
        choices=["quad-tree-2", "quad-tree-4", "quad-graph"],
        help="region graph",
    )
    parser.add_argument(
        "--inner-layer",
        type=str,
        default="cp",
        choices=["cp", "tucker"],
        help="sum-product layer type",
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
        "--weights-init",
        type=str,
        default="uniform",
        choices=["uniform", "normal", "dirichlet"],
        help="weights initialization for sum units",
    )
    parser.add_argument("--root", type=str, default="datasets", help="path to MNIST dataset")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--T_0", type=int, default=1, help="T_0 for cosine annealing")
    parser.add_argument("--eta-min", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--validation-steps", type=int, default=250)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--valid-split", type=float, default=0.05)
    parser.add_argument("--use-scheduler", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="best_circuit.pt")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="mnist_pc")

    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--noise-scale", type=float, default=2.0)
    parser.add_argument("--use-miwae", action="store_true")
    parser.add_argument("--adaptive-alpha", action="store_true")
    parser.add_argument("--use-mixing-weights", action="store_true")
    parser.add_argument("--use-estimated-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"  device: {device}")
    print(f"  learn_spn_source: {inspect.getfile(LearnSPN)}")
    print("=" * 70)

    run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("Install wandb or run without --wandb")
        wandb.login()
        run = wandb.init(project=args.project, config=vars(args))

    mnist_train = datasets.MNIST(root=args.root, train=True, download=True)
    mnist_test = datasets.MNIST(root=args.root, train=False, download=True)

    X_train = mnist_train.data.view(-1, 28 * 28).long()
    X_test = mnist_test.data.view(-1, 28 * 28).long()

    n_val = int(len(X_train) * args.valid_split)
    n_train = len(X_train) - n_val
    train_data, val_data = torch.utils.data.random_split(
        X_train,
        [n_train, n_val],
        # Deliberately preserved from the legacy script.
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(X_test, batch_size=args.batch_size, shuffle=False)

    weight_dir = os.path.join(os.getcwd(), "best_categorical_miwae.pt")
    learner = LearnSPN(
        alpha=args.alpha,
        noise_scale=args.noise_scale,
        use_miwae=args.use_miwae,
        data_format="image",
        image_shape=(1, 28, 28),
        device=device,
        weight_dir=weight_dir,
        adaptive_alpha=args.adaptive_alpha,
        input_sharing="none",
        num_categories=256,
    )

    train_split = X_train[torch.as_tensor(train_data.indices)]
    logger.info(f"Building the circuit from all {len(train_split)} legacy training samples")
    symbolic_circuit = learner.learn_spn(
        train_split.to(device),
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

    symbolic_partition_function = sf.integrate(symbolic_circuit)
    logger.info(f"Circuit built with {len(list(symbolic_circuit.layers))} layers")

    max_train_steps = int(len(train_loader) * args.max_epochs)
    context = PipelineContext(
        backend="torch", semiring="lse-sum", fold=True, optimize=True
    )
    circuit = context.compile(symbolic_circuit).to(device)
    circuit_partition_function = context.compile(symbolic_partition_function).to(device)

    logger.info(f"Number of parameters: {sum(p.numel() for p in circuit.parameters())}")

    sum_params = [
        parameter
        for layer in circuit.layers
        if not isinstance(layer, TorchInputLayer)
        for parameter in layer.parameters()
    ]

    train_circuit(
        circuit,
        circuit_partition_function,
        train_loader,
        val_loader,
        sum_params=sum_params,
        max_train_steps=max_train_steps,
        lr=args.lr,
        t_0=args.T_0,
        eta_min=args.eta_min,
        weight_decay=args.weight_decay,
        validation_steps=args.validation_steps,
        delta=args.min_delta,
        patience=args.patience,
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
        log_to_wandb=args.wandb,
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
