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
import matplotlib.pyplot as plt
import torch.optim as optim

from cirkit.pipeline import PipelineContext, compile
import cirkit.symbolic.functional as sf
from cirkit.templates.learn_spn import LearnSPN
import argparse

def set_nested_key(d, key_path, value):
    """
    Safely updates a nested dictionary given a dotted key path.
    Example:
      set_nested_key(cfg, 'training.lr', 0.001)
    """
    keys = key_path.split('.')
    sub_dict = d
    for k in keys[:-1]:
        if k not in sub_dict or not isinstance(sub_dict[k], dict):
            sub_dict[k] = {}
        sub_dict = sub_dict[k]

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "false"}:
            value = v == "true"
        else:
            try:
                if '.' in v or 'e' in v:  # handles floats like 1e-3
                    value = float(v)
                else:
                    value = int(v)
            except ValueError:
                value = value  # leave as string if conversion fails

    sub_dict[keys[-1]] = value


def train_circuit(
    symbolic_circuit,
    symbolic_partition_function,
    train_loader,
    val_loader,
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
    log_to_wandb=True
):
    ctx = PipelineContext(
        backend="torch",
        semiring='lse-sum',
        fold=True,
        optimize=True
    )

    circuit = ctx.compile(symbolic_circuit).to(device)
    circuit_partition_function = ctx.compile(symbolic_partition_function).to(device)

    optimizer = optim.Adam(circuit.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=eta_min)

    best_val_nll = float("inf")
    epochs_no_improve = 0
    total_steps = 0

    logs = {"step": [], "train_nll": [], "val_nll": [], "train_bpd": [], "val_bpd": []}
    print(f'Starting training for a maximum of {max_train_steps} steps.\n')
    while total_steps < max_train_steps:
        train_loss_sum = 0.0
        train_count = 0

        for batch in tqdm(train_loader, desc="[Train]", leave=False):
            batch = batch.to(device)
            log_liks = circuit(batch) - circuit_partition_function()
            loss = -log_liks.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item() * batch.size(0)
            train_count += batch.size(0)
            total_steps += 1
            
            if total_steps % validation_steps == 0:
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        log_liks = circuit(val_batch) - circuit_partition_function()
                        val_loss_sum += (-log_liks.mean()).item() * val_batch.size(0)
                        val_count += val_batch.size(0)
                avg_val_nll = val_loss_sum / val_count
                avg_train_nll = train_loss_sum / train_count
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
                    epochs_no_improve = 0
                    logger.success(f"New best model at step {total_steps}, Train NLL: {avg_train_nll:.4f}, Train bpd: {bpd_train:.4f}, Val NLL: {best_val_nll:.4f}, Val bpd: {bpd_val:.4f}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"No improvement at step {total_steps}, count: {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience:
                    logger.info("Early stopping triggered.")
                    return circuit, circuit_partition_function, logs
                
            if total_steps >= max_train_steps:
                break

    return circuit, circuit_partition_function, logs


def evaluate_circuit(
        circuit,
        circuit_partition_function,
        test_loader,
        device,
        checkpoint_path,
        log_to_wandb=True
        ):
    
    circuit.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    return avg_test_nll, avg_test_bpd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train probabilistic circuit on MNIST")
    parser.add_argument("--config", type=str, default="config_mnist.yml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config values, e.g. --override training.lr=0.001 dataset.batch_size=256")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for override_str in args.override:
        if "=" not in override_str:
            raise ValueError(f"Invalid override format: {override_str}. Expected key=value")
        key, val = override_str.split("=", 1)
        set_nested_key(cfg, key, val)

    print("⚙️ Final configuration:")
    print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    use_wandb = cfg["logging"].get("use_wandb", False)
    if use_wandb:
        wandb.login()
        run = wandb.init(project=cfg["project"], config=cfg)
        
    valid_split_percentage = cfg["dataset"]["valid_split_percentage"]
    mnist_train = datasets.MNIST(root=cfg["dataset"]["root"], train=True, download=True)
    mnist_test = datasets.MNIST(root=cfg["dataset"]["root"], train=False, download=True)

    X_train = mnist_train.data.view(-1, 28 * 28).long().to(device)
    X_test = mnist_test.data.view(-1, 28 * 28).long().to(device)
 
    n_val = int(len(X_train) * valid_split_percentage)
    n_train = len(X_train) - n_val
    train_data, val_data = torch.utils.data.random_split(
        X_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=cfg["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg["dataset"]["batch_size"], shuffle=False)
    test_loader = DataLoader(X_test, batch_size=cfg["dataset"]["batch_size"], shuffle=False)

    weight_dir = os.path.join(os.getcwd(), "best_categorical_miwae.pt")

    params = cfg["learn_spn"]
    spn_learner = LearnSPN(
        alpha=params["alpha"],
        noise_scale=params["noise_scale"],
        use_miwae=params["use_miwae"],
        data_format="image",
        image_shape=tuple(params["image_shape"]),
        device=device,
        weight_dir=weight_dir
    )
    symbolic_circuit = spn_learner.learn_spn(
        train_data.dataset,
        input_layer="categorical",
        region_graph=params["region_graph"],
        activation=params["activation"],
        weights_init=params["weights_init"],
        sum_product_layer=params["sum_product_layer"],
        num_input_units=params["num_input_units"],
        num_sum_units=params["num_sum_units"],
        use_mixing_weights=params["use_mixing_weights"],
        use_estimated_weights=params["use_estimated_weights"]
    )

    symbolic_partition_function = sf.integrate(symbolic_circuit)

    logger.info(f"Circuit built with {len(list(symbolic_circuit.layers))} layers")

    torch.cuda.empty_cache()
    gc.collect()

    max_train_steps = int(len(train_data) // cfg["dataset"]["batch_size"] * cfg["training"]["epochs"])

    circuit, circuit_partition_function, _ = train_circuit(
        symbolic_circuit,
        symbolic_partition_function,
        train_loader,
        val_loader,
        max_epochs=cfg["training"]["epochs"],
        max_train_steps=max_train_steps,
        lr=cfg["training"]["lr"],
        T_0=cfg["training"]["T_0"],
        eta_min=cfg["training"]["eta_min"],
        weight_decay=cfg["training"]["weight_decay"],
        validation_steps=cfg["training"]["validation_steps"],
        delta=cfg["training"]["delta"],
        patience=cfg["training"]["patience"],
        device=device,
        save_path=cfg["training"]["save_path"],
        log_to_wandb=use_wandb
    )

    torch.cuda.empty_cache()
    gc.collect()

    evaluate_circuit(
        circuit,
        circuit_partition_function,
        test_loader,
        device=device,
        checkpoint_path=cfg["training"]["save_path"],
        log_to_wandb=use_wandb
    )

    if use_wandb:
        run.finish()