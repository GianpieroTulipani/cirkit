import os
import gc
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import numpy as np
import wandb
from tqdm.auto import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from loguru import logger
import torch.optim as optim

from cirkit.pipeline import PipelineContext, compile
import cirkit.symbolic.functional as sf
from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.templates.learn_spn import LearnSPN as LearnSPNBase
from cirkit.templates.learn_spn_optimized import LearnSPN as LearnSPNOptimized
import argparse

LEARN_SPN_VARIANTS = {
    "base": LearnSPNBase,
    "optimized": LearnSPNOptimized,
}

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
    T_0,
    eta_min,
    weight_decay,
    device,
    save_path,
    validation_steps,
    delta,
    patience,
    log_to_wandb=True,
    activation=None,
    use_scheduler=False
):

    optimizer = optim.Adam(circuit.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=eta_min)

    best_val_nll = float("inf")
    epochs_no_improve = 0
    total_steps = 0
    avg_train_nll = None
    avg_val_nll = None
    bpd_train = None
    bpd_val = None
    saved_best = False

    logs = {"step": [], "train_nll": [], "val_nll": [], "train_bpd": [], "val_bpd": []}

    print(f'Starting training for a maximum of {max_train_steps} steps.\n')

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
                    for p in sum_params:
                        p.data.clamp_(min=float(np.sqrt(np.finfo(np.float32).tiny)))

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

                logs["step"].append(total_steps)
                logs["train_nll"].append(avg_train_nll)
                logs["val_nll"].append(avg_val_nll)
                logs["train_bpd"].append(bpd_train)
                logs["val_bpd"].append(bpd_val)

                if log_to_wandb:
                    wandb.log({
                        "step": total_steps,
                        "train_nll": avg_train_nll,
                        "train_bpd": bpd_train,
                        "val_nll": avg_val_nll,
                        "val_bpd": bpd_val
                    })

                if avg_val_nll - delta <= best_val_nll:
                    best_val_nll = avg_val_nll
                    torch.save(circuit.state_dict(), save_path)
                    saved_best = True
                    epochs_no_improve = 0
                    logger.success(f"New best model at step {total_steps}, Train NLL: {avg_train_nll:.4f}, Train bpd: {bpd_train:.4f}, Val NLL: {best_val_nll:.4f}, Val bpd: {bpd_val:.4f}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"No improvement at step {total_steps}, count: {epochs_no_improve}/{patience}")

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
def evaluate_circuit(
        circuit,
        circuit_partition_function,
        test_loader,
        device,
        log_to_wandb=True
        ):
    
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
        wandb.log({"test_nll": avg_test_nll, "test_bpd": avg_test_bpd})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train probabilistic circuit on MNIST")
    parser.add_argument("--rg", type=str, default="quad-tree-2",
                        choices=["quad-tree-2", "quad-tree-4", "quad-graph"], help="region graph")
    parser.add_argument("--inner-layer", type=str, default="cp",
                        choices=["cp", "tucker"], help="sum-product layer type")
    parser.add_argument("--k", type=int, default=512, help="num units per layer")
    parser.add_argument("--activation", type=str, default="clamp",
                        choices=["clamp", "softmax", "none"],
                        help="activation function for sum units")
    parser.add_argument("--weights-init", type=str, default="uniform",
                        choices=["uniform", "normal", "dirichlet"],
                        help="weights initialization for sum units")
    parser.add_argument("--root", type=str, default="datasets", help="path to MNIST dataset")
    
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--T_0", type=int, default=1, help="T_0 for cosine annealing")
    parser.add_argument("--eta-min", type=float, default=0.0001, help="eta_min for cosine annealing")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--validation-steps", type=int, default=250, help="validation every n steps")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--valid-split", type=float, default=0.05)
    parser.add_argument("--use-scheduler", action="store_true", help="use cosine annealing scheduler")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="best_circuit.pt", help="path to save the best circuit")
    parser.add_argument("--wandb", action="store_true", help="log to wandb")
    parser.add_argument("--project", type=str, default="mnist_pc", help="wandb project name")
    parser.add_argument("--variant", type=str, default="optimized", choices=["base", "optimized"], help="LearnSPN variant")

    parser.add_argument("--alpha", type=float, default=5.0, help="alpha parameter for LearnSPN")
    parser.add_argument("--noise-scale", type=float, default=2.0, help="noise scale for LearnSPN")
    parser.add_argument("--use-miwae", action="store_true", help="use MIWAE for LearnSPN")
    parser.add_argument("--adaptive-alpha", action="store_true", help="use adaptive alpha for LearnSPN optimized variant")
    parser.add_argument("--subcluster-lambda", type=float, default=0.6, help="subcluster lambda for LearnSPN optimized variant")
    parser.add_argument("--use-mixing-weights", action="store_true", help="use mixing weights for LearnSPN")
    parser.add_argument("--use-estimated-weights", action="store_true", help="use estimated weights for LearnSPN")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
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
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(X_test, batch_size=args.batch_size, shuffle=False)

    weight_dir = os.path.join(os.getcwd(), "best_categorical_miwae.pt")

    variant = str(args.variant)
    if variant not in LEARN_SPN_VARIANTS:
        raise ValueError(
            f"learn_spn.variant deve essere uno di {list(LEARN_SPN_VARIANTS)}, non {variant!r}"
        )
    
    LearnSPNCls = LEARN_SPN_VARIANTS[variant]

    learner_kwargs = dict(
        alpha=args.alpha,
        noise_scale=args.noise_scale,
        use_miwae=args.use_miwae,
        data_format="image",
        image_shape=(1, 28, 28),
        device=device,
        weight_dir=weight_dir
    )

    if variant == "optimized":
        learner_kwargs.update(
            adaptive_alpha=args.adaptive_alpha,
            subcluster_lambda=args.subcluster_lambda,
        )

    logger.info(f"Using LearnSPN variant: {variant}")

    spn_learner = LearnSPNCls(**learner_kwargs)

    train_split = X_train[torch.as_tensor(train_data.indices)]

    symbolic_circuit = spn_learner.learn_spn(
        train_split.to(device),
        input_layer="categorical",
        region_graph=args.rg,
        activation=args.activation,
        weights_init=args.weights_init,
        sum_product_layer=args.inner_layer,
        num_input_units=args.k,
        num_sum_units=args.k,
        use_mixing_weights=args.use_mixing_weights,
        use_estimated_weights=args.use_estimated_weights
    )

    symbolic_partition_function = sf.integrate(symbolic_circuit)

    logger.info(f"Circuit built with {len(list(symbolic_circuit.layers))} layers")
    logger.info(f"Number of parameters: {sum(p.numel() for p in symbolic_circuit.parameters())}")

    max_train_steps = int(len(train_loader) * args.max_epochs)

    ctx = PipelineContext(
        backend="torch",
        semiring='lse-sum',
        fold=True,
        optimize=True
    )

    circuit = ctx.compile(symbolic_circuit).to(device)
    circuit_partition_function = ctx.compile(symbolic_partition_function).to(device)

    sum_params = [
        p for layer in circuit.layers if not isinstance(layer, TorchInputLayer)
        for p in layer.parameters()
    ]

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
        device=device,
        save_path=args.save_path,
        log_to_wandb=args.wandb,
        activation=args.activation,
        use_scheduler=args.use_scheduler
    )

    torch.cuda.empty_cache()
    gc.collect()

    circuit.load_state_dict(torch.load(args.save_path, map_location=device))

    evaluate_circuit(
        circuit,
        circuit_partition_function,
        test_loader,
        device=device,
        log_to_wandb=args.wandb
    )
    if args.wandb:
        run.finish()
