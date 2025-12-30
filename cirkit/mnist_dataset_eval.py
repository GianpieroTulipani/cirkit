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

from cirkit.pipeline import compile
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
    train_loader,
    val_loader,
    num_epochs,
    lr,
    weight_decay,
    device,
    save_path,
    log_to_wandb=True
):
    circuit = compile(symbolic_circuit).to(device)
    optimizer = optim.Adam(circuit.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_nll = float("inf")

    epoch_list = []
    train_nll_list = []
    val_nll_list = []
    train_bpd_list = []
    val_bpd_list = []

    logs = {"epoch": [], "train_nll": [], "val_nll": [], "train_bpd": [], "val_bpd": []}

    for epoch in range(1, num_epochs + 1):
        circuit.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            batch = batch.to(device)
            log_liks = circuit(batch)
            loss = -log_liks.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch.size(0)
            train_count += batch.size(0)
        avg_train_nll = train_loss_sum / train_count
        bpd_train = avg_train_nll / (28 * 28 * np.log(2.0))
        logger.info(f"Epoch {epoch} — Train NLL: {avg_train_nll:.4f} | bpd: {bpd_train:.4f}")

        circuit.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                batch = batch.to(device)
                log_liks = circuit(batch)
                loss = -log_liks.mean()
                val_loss_sum += loss.item() * batch.size(0)
                val_count += batch.size(0)
        avg_val_nll = val_loss_sum / val_count
        bpd_val = avg_val_nll / (28 * 28 * np.log(2.0))
        logger.info(f"Epoch {epoch} — Val NLL: {avg_val_nll:.4f} | bpd: {bpd_val:.4f}")

        epoch_list.append(epoch)
        train_nll_list.append(avg_train_nll)
        val_nll_list.append(avg_val_nll)
        train_bpd_list.append(bpd_train)
        val_bpd_list.append(bpd_val)

        logs["epoch"].append(epoch)
        logs["train_nll"].append(avg_train_nll)
        logs["val_nll"].append(avg_val_nll)
        logs["train_bpd"].append(bpd_train)
        logs["val_bpd"].append(bpd_val)

        if log_to_wandb:
            wandb.log({
                "epoch": epoch,
                "train_nll": avg_train_nll,
                "train_bpd": bpd_train,
                "val_nll": avg_val_nll,
                "val_bpd": bpd_val
            })

        if avg_val_nll < best_val_nll:
            best_val_nll = avg_val_nll
            torch.save(circuit.state_dict(), save_path)
            logger.success(f"New best model at epoch {epoch}, Val NLL: {best_val_nll:.4f}")

    if log_to_wandb:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        ax = axes[0]
        ax.scatter(epoch_list, train_nll_list, label='Train NLL', marker='o')
        ax.plot(epoch_list, train_nll_list, linestyle='-', alpha=0.6)
        ax.scatter(epoch_list, val_nll_list, label='Val NLL', marker='x')
        ax.plot(epoch_list, val_nll_list, linestyle='--', alpha=0.6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NLL')
        ax.set_title('NLL per Epoch')
        ax.grid(True)
        ax.legend()
    
        ax = axes[1]
        ax.scatter(epoch_list, train_bpd_list, label='Train bpd', marker='o')
        ax.plot(epoch_list, train_bpd_list, linestyle='-', alpha=0.6)
        ax.scatter(epoch_list, val_bpd_list, label='Val bpd', marker='x')
        ax.plot(epoch_list, val_bpd_list, linestyle='--', alpha=0.6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('bpd')
        ax.set_title('bits-per-dimension (bpd) per Epoch')
        ax.grid(True)
        ax.legend()
    
        plt.tight_layout()
        fig_path = 'learning_curve.png'
        fig.savefig(fig_path, dpi=150)
        try:
            wandb.log({"learning_curve": wandb.Image(fig_path)})
        except Exception as e:
            logger.error(f"Failed to log learning curve to wandb: {e}")
        plt.close(fig)
    return circuit, logs


def evaluate_circuit(circuit, test_loader, device, checkpoint_path, log_to_wandb=True):
    circuit.load_state_dict(torch.load(checkpoint_path, map_location=device))
    circuit.eval()
    test_nll_sum = 0.0
    test_count = 0
    for batch in tqdm(test_loader, desc="[Test]", leave=False):
        batch = batch.to(device)
        log_liks = circuit(batch)
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

    # Optional logging
    use_wandb = cfg["logging"].get("use_wandb", False)
    if use_wandb:
        wandb.login()
        run = wandb.init(project=cfg["project"], config=cfg)

    mnist_train = datasets.MNIST(root=cfg["dataset"]["root"], train=True, download=True)
    mnist_test = datasets.MNIST(root=cfg["dataset"]["root"], train=False, download=True)

    X_train = mnist_train.data.view(-1, 28 * 28).long().to(device)
    X_test = mnist_test.data.view(-1, 28 * 28).long().to(device)

    n_train = cfg["dataset"]["n_train"]
    n_val = cfg["dataset"]["n_val"]
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
        region_graph=params["region_graph"],
        input_layer="categorical",
        activation=params["activation"],
        weights_init=params["weights_init"],
        sum_product_layer=params["sum_product_layer"],
        num_input_units=params["num_input_units"],
        num_sum_units=params["num_sum_units"],
        use_mixing_weights=params["use_mixing_weights"],
        use_estimated_weights=params["use_estimated_weights"]
    )

    logger.info(f"Circuit built with {len(list(symbolic_circuit.layers))} layers")

    torch.cuda.empty_cache()
    gc.collect()

    circuit, _ = train_circuit(
        symbolic_circuit,
        train_loader,
        val_loader,
        num_epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        device=device,
        save_path=cfg["training"]["save_path"],
        log_to_wandb=use_wandb
    )

    evaluate_circuit(
        circuit,
        test_loader,
        device=device,
        checkpoint_path=cfg["training"]["save_path"],
        log_to_wandb=use_wandb
    )

    if use_wandb:
        run.finish()