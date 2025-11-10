import os
import gc
import yaml
import argparse
import pandas as pd
import torch
import openml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from cirkit.pipeline import PipelineContext
from cirkit.templates.learn_spn import LearnSPN
from cirkit.templates.data_modalities import tabular_data
from cirkit.templates import utils


# ======================================================
# ⚙️ Utility
# ======================================================
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
                if '.' in v or 'e' in v:
                    value = float(v)
                else:
                    value = int(v)
            except ValueError:
                value = value
    sub_dict[keys[-1]] = value


# ======================================================
# 🧠 Core Functions
# ======================================================
def train_circuit(symbolic_circuit, train_loader, val_loader, num_epochs, lr, weight_decay,
                  device, save_path="best_circuit.pth", log_to_wandb=True):
    device = torch.device(device)
    ctx = PipelineContext(backend='torch', semiring='lse-sum', fold=True, optimize=True)
    circuit = ctx.compile(symbolic_circuit).to(device)
    optimizer = torch.optim.Adam(circuit.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_nll = float("inf")
    logs = {"epoch": [], "train_nll": [], "val_nll": []}

    for epoch in range(1, num_epochs + 1):
        circuit.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            batch = batch.to(device)
            log_liks = circuit(batch)
            loss = -log_liks.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_nll = np.mean(train_losses)
        circuit.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                batch = batch.to(device)
                log_liks = circuit(batch)
                val_losses.append((-log_liks.mean()).item())
        avg_val_nll = np.mean(val_losses)

        print(f"Epoch {epoch}: Train NLL={avg_train_nll:.4f} | Val NLL={avg_val_nll:.4f}")

        if log_to_wandb:
            wandb.log({
                "epoch": epoch,
                "train_nll": avg_train_nll,
                "val_nll": avg_val_nll
            })

        if avg_val_nll < best_val_nll:
            best_val_nll = avg_val_nll
            torch.save(circuit.state_dict(), save_path)
            print(f"✅ New best model at epoch {epoch}, Val NLL={best_val_nll:.4f}")

        logs["epoch"].append(epoch)
        logs["train_nll"].append(avg_train_nll)
        logs["val_nll"].append(avg_val_nll)

    return circuit, logs


def evaluate_circuit(circuit, test_loader, device="cpu", checkpoint_path="best_circuit.pth", log_to_wandb=True):
    device = torch.device(device)
    circuit.load_state_dict(torch.load(checkpoint_path, map_location=device))
    circuit.to(device)
    circuit.eval()

    test_losses = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]", leave=False):
            batch = batch.to(device)
            log_liks = circuit(batch)
            test_losses.append((-log_liks.mean()).item())

    avg_test_nll = np.mean(test_losses)
    print(f"📊 Test NLL: {avg_test_nll:.4f}")

    if log_to_wandb:
        wandb.log({"test_nll": avg_test_nll})

    return avg_test_nll


def build_spn_structure(train_data, device, params):
    learner = LearnSPN(
        alpha=params["alpha"],
        min_instances=params["min_instances"],
        mi_quantile=params["mi_quantile"],
        jitter_scale=params["jitter_scale"],
        device="cuda",
        data_format='tabular'
    )

    data_tensor = train_data.dataset[train_data.indices].to(device)
    return learner.learn_spn(
        data_tensor,
        input_layer='categorical',
        activation='softmax',
        initialization=params["initialization"],
        num_input_units=params["num_input_units"],
        num_sum_units=params["num_sum_units"]
    )


def build_random_binary_tree_structure(num_features, dataset, params):
    kwargs = int(dataset.nunique().max())
    return tabular_data(
        region_graph='random-binary-tree',
        num_features=num_features,
        kwargs=kwargs,
        input_layer='categorical',
        num_input_units=params["num_input_units"],
        sum_product_layer='cp',
        num_sum_units=params["num_sum_units"],
        sum_weight_param=utils.Parameterization(
            activation=params["sum_weight_activation"],
            initialization=params["sum_weight_init"]
        )
    )


def build_chow_liu_tree_structure(dataset, params):
    kwargs = int(dataset.nunique().max())
    X_train = torch.tensor(dataset.values, dtype=torch.float32)
    return tabular_data(
        region_graph='chow-liu-tree',
        data=X_train,
        kwargs=kwargs,
        input_layer='categorical',
        num_input_units=params["num_input_units"],
        sum_product_layer='cp',
        num_sum_units=params["num_sum_units"],
        sum_weight_param=utils.Parameterization(
            activation=params["sum_weight_activation"],
            initialization=params["sum_weight_init"]
        )
    )


# ======================================================
# 🚀 Main
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Probabilistic Circuit (with W&B logging)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--mode", type=str, help="Override mode: learn_spn, rbt, clt")
    parser.add_argument("--override", nargs="*", help="Override config values, e.g. --override training.lr=0.001 rbt.num_sum_units=64")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.mode:
        config["mode"] = args.mode

    if args.override:
        for override_str in args.override:
            if "=" not in override_str:
                raise ValueError(f"Invalid override: {override_str} (must be key=value)")
            key, val = override_str.split("=", 1)
            set_nested_key(config, key, val)

    print("⚙️ Final Configuration:")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))

    # Initialize W&B
    use_wandb = config.get("logging", {}).get("use_wandb", True)
    if use_wandb:
        wandb.login()
        wandb.init(project=config.get("project", "cirkit_openml"), config=config)

    # Load dataset
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    dataset = openml.datasets.get_dataset(config["dataset"]["openml_id"], download_all_files=True).get_data()[0]
    dataset = pd.get_dummies(dataset)
    tensor_dataset = torch.tensor(dataset.values, dtype=torch.long)

    n_train = config["dataset"]["split"]["train"]
    n_val = config["dataset"]["split"]["val"]
    n_test = len(tensor_dataset) - n_train - n_val

    train_data, val_data, test_data = torch.utils.data.random_split(
        tensor_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, shuffle=True, batch_size=config["dataset"]["batch_size"])
    val_loader = DataLoader(val_data, shuffle=True, batch_size=config["dataset"]["batch_size"])
    test_loader = DataLoader(test_data, shuffle=False, batch_size=config["dataset"]["batch_size"])

    # Build circuit
    mode = config["mode"].lower()
    if mode == "learn_spn":
        print("🧠 Building LearnSPN structure...")
        symbolic_circuit = build_spn_structure(train_data, device, config["learn_spn"])
    elif mode in ["rbt", "random_binary_tree"]:
        print("🌲 Building Random Binary Tree structure...")
        symbolic_circuit = build_random_binary_tree_structure(dataset.shape[1], dataset, config["rbt"])
    elif mode in ["clt", "chow_liu_tree"]:
        print("🌳 Building Chow–Liu Tree structure...")
        symbolic_circuit = build_chow_liu_tree_structure(dataset, config["clt"])
    else:
        raise ValueError("Invalid mode in config.yaml or CLI override.")

    print(f"✅ Circuit built with {len(list(symbolic_circuit.layers))} layers")

    torch.cuda.empty_cache()
    gc.collect()

    circuit, _ = train_circuit(
        symbolic_circuit,
        train_loader,
        val_loader,
        num_epochs=config["training"]["num_epochs"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        device=device,
        save_path=config["training"].get("save_path", "best_circuit.pth"),
        log_to_wandb=use_wandb
    )

    evaluate_circuit(
        circuit,
        test_loader,
        device=device,
        checkpoint_path=config["training"].get("save_path", "best_circuit.pth"),
        log_to_wandb=use_wandb
    )

    if use_wandb:
        wandb.finish()
