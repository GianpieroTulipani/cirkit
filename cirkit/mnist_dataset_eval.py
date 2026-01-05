import os
import gc
import yaml
import torch
import numpy as np
import wandb
from tqdm.auto import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from loguru import logger
import torch.optim as optim
import argparse

from cirkit.pipeline import PipelineContext
import cirkit.symbolic.functional as sf
from cirkit.templates.learn_spn import LearnSPN

def set_nested_key(d, key_path, value):
    keys = key_path.split('.')
    sub_dict = d
    for k in keys[:-1]:
        sub_dict = sub_dict.setdefault(k, {})
    try:
        value = eval(value)
    except:
        pass
    sub_dict[keys[-1]] = value

def train_circuit(
    symbolic_circuit,
    symbolic_partition_function,
    train_loader,
    val_loader,
    cfg,
    device
):

    ctx = PipelineContext(backend="torch", semiring='lse-sum', fold=True, optimize=True)
    circuit = ctx.compile(symbolic_circuit).to(device)
    Z_module = ctx.compile(symbolic_partition_function).to(device)

    optimizer = optim.Adam(circuit.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["training"]["T_0"],
        eta_min=cfg["training"]["eta_min"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Precompute partition function once
    with torch.inference_mode():
        Z = Z_module()

    best_val_nll = float("inf")
    patience_count = 0
    total_steps = 0
    max_steps = int(len(train_loader) * cfg["training"]["epochs"])

    logger.info(f"Training for {max_steps} steps")

    for epoch in range(cfg["training"]["epochs"]):
        circuit.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                log_liks = circuit(batch) - Z
                loss = -log_liks.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_steps += 1

            del batch, log_liks, loss

            if total_steps % cfg["training"]["validation_steps"] == 0:
                val_nll = validate(circuit, Z, val_loader, device)
                if val_nll < best_val_nll - cfg["training"]["delta"]:
                    best_val_nll = val_nll
                    patience_count = 0
                    torch.save(circuit.state_dict(), cfg["training"]["save_path"])
                    logger.success(f"New best: {val_nll:.4f}")
                else:
                    patience_count += 1
                    logger.info(f"No improvement: {patience_count}/{cfg['training']['patience']}")

                if patience_count >= cfg["training"]["patience"]:
                    return circuit, Z

            if total_steps >= max_steps:
                return circuit, Z

    return circuit, Z

def validate(circuit, Z, loader, device):
    circuit.eval()
    total, count = 0.0, 0

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            log_liks = circuit(batch) - Z
            loss = -log_liks.mean()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
            del batch, log_liks, loss

    return total / count

def evaluate(circuit, Z, loader, cfg, device):
    circuit.load_state_dict(torch.load(cfg["training"]["save_path"], map_location=device))
    circuit.eval()

    total, count = 0.0, 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="[Test]", leave=False):
            batch = batch.to(device, non_blocking=True)
            log_liks = circuit(batch) - Z
            loss = -log_liks.mean()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
            del batch, log_liks, loss

    nll = total / count
    bpd = nll / (28 * 28 * np.log(2))
    logger.info(f"Test NLL: {nll:.4f} | BPD: {bpd:.4f}")

    return nll, bpd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_mnist.yml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for item in args.override:
        k, v = item.split("=")
        set_nested_key(cfg, k, v)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    mnist_train = datasets.MNIST(cfg["dataset"]["root"], train=True, download=True)
    mnist_test = datasets.MNIST(cfg["dataset"]["root"], train=False, download=True)

    X_train = mnist_train.data.view(-1, 28 * 28).long()
    X_test = mnist_test.data.view(-1, 28 * 28).long()

    n_val = int(len(X_train) * cfg["dataset"]["valid_split_percentage"])
    train_data, val_data = torch.utils.data.random_split(X_train, [len(X_train) - n_val, n_val])

    train_loader = DataLoader(train_data, batch_size=cfg["dataset"]["batch_size"], shuffle=True,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=cfg["dataset"]["batch_size"],
                            pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(X_test, batch_size=cfg["dataset"]["batch_size"],
                             pin_memory=True)

    spn = LearnSPN(**cfg["learn_spn"], device=device)
    symbolic_circuit = spn.learn_spn(train_data.dataset, input_layer="categorical")
    symbolic_partition_function = sf.integrate(symbolic_circuit)

    circuit, Z = train_circuit(symbolic_circuit, symbolic_partition_function, train_loader, val_loader, cfg, device)

    torch.cuda.empty_cache()
    gc.collect()

    evaluate(circuit, Z, test_loader, cfg, device)

if __name__ == "__main__":
    main()
