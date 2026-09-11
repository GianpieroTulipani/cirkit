import os
import gc
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torchvision import datasets
except ImportError:
    datasets = None

try:
    from loguru import logger
except ImportError:
    class _Logger:
        @staticmethod
        def info(message):
            print(message)

        @staticmethod
        def success(message):
            print(message)

    logger = _Logger()

import cirkit.symbolic.functional as sf
from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.pipeline import PipelineContext


def _forward_lift(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    diff = (y - x).remainder(256)
    average = (x + torch.div(diff, 2, rounding_mode="floor")).remainder(256)
    return average, diff


def rgb_to_ycocg_lossless(images: torch.Tensor) -> torch.Tensor:
    images = images.long()
    red, green, blue = images[:, 0], images[:, 1], images[:, 2]
    tmp, co = _forward_lift(red, blue)
    y, cg = _forward_lift(green, tmp)
    return torch.stack((y, co, cg), dim=1).long()


def rgb_to_ycocg_lossy(images: torch.Tensor) -> torch.Tensor:
    deq_img = (images.float() / 127.5) - 1.0
    red = (deq_img[:, 0] + 1.0) / 2.0
    green = (deq_img[:, 1] + 1.0) / 2.0
    blue = (deq_img[:, 2] + 1.0) / 2.0
    co = red - blue
    tmp = blue + co / 2
    cg = green - tmp
    y = tmp + cg / 2
    y = y * 2.0 - 1.0
    transformed_img = torch.stack((y, co, cg), dim=1)
    return torch.floor(((transformed_img + 1.0) / 2.0) * 256).long().clip(0, 255)


def apply_color_transform(images: torch.Tensor, ycc: str) -> torch.Tensor:
    if ycc == "none":
        return images.long()
    if images.ndim != 4 or images.size(1) != 3:
        raise ValueError(f"YCoCg transforms require RGB tensors of shape (N, 3, H, W), found {tuple(images.shape)}")
    if ycc == "lossy":
        return rgb_to_ycocg_lossy(images)
    if ycc == "lossless":
        return rgb_to_ycocg_lossless(images)
    raise ValueError(f"Unknown YCoCg mode {ycc!r}")


def flatten_images(images: torch.Tensor, ycc: str) -> torch.Tensor:
    images = apply_color_transform(images, ycc)
    return images.reshape(images.size(0), -1).to(torch.uint8)


def _infer_flat_shape(data: torch.Tensor, image_shape: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if image_shape is None:
        raise ValueError("--image-shape C H W is required for flat tensor datasets")
    expected = int(np.prod(image_shape))
    if data.ndim != 2 or data.size(1) != expected:
        raise ValueError(
            f"Expected flat tensor data with shape (N, {expected}) for image_shape={image_shape}, "
            f"found {tuple(data.shape)}"
        )
    return image_shape


def _load_tensor_file(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, TensorDataset):
        tensor = tensor.tensors[0]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, found {type(tensor).__name__}")
    return tensor


def _load_local_tensor_dataset(
    root: str,
    image_shape: tuple[int, int, int] | None,
    ycc: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, tuple[int, int, int]]:
    root_path = Path(root)
    train = _load_tensor_file(root_path / "train.pt")
    valid_path = root_path / "valid.pt"
    test_path = root_path / "test.pt"
    valid = _load_tensor_file(valid_path) if valid_path.exists() else None
    test = _load_tensor_file(test_path) if test_path.exists() else valid
    if test is None:
        raise FileNotFoundError(f"Expected either {valid_path} or {test_path}")

    if train.ndim == 4:
        shape = tuple(train.shape[1:])
        if len(shape) != 3:
            raise ValueError(f"Expected image tensors of shape (N, C, H, W), found {tuple(train.shape)}")
        train = flatten_images(train, ycc)
        valid = flatten_images(valid, ycc) if valid is not None and valid.ndim == 4 else valid
        test = flatten_images(test, ycc) if test.ndim == 4 else test
        return train, valid, test, shape

    shape = _infer_flat_shape(train, image_shape)
    if ycc != "none":
        c, h, w = shape
        if c != 3:
            raise ValueError(f"YCoCg transforms require 3 channels, found image_shape={shape}")
        train = flatten_images(train.reshape(-1, c, h, w), ycc)
        valid = flatten_images(valid.reshape(-1, c, h, w), ycc) if valid is not None else None
        test = flatten_images(test.reshape(-1, c, h, w), ycc)
    return train, valid, test, shape


def _load_vision_dataset(dataset: str, root: str, ycc: str) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    if datasets is None:
        raise ImportError("Install torchvision to load mnist, fashion-mnist, or cifar10")
    if dataset == "mnist":
        train_ds = datasets.MNIST(root=root, train=True, download=True)
        test_ds = datasets.MNIST(root=root, train=False, download=True)
        return flatten_images(train_ds.data[:, None], ycc), flatten_images(test_ds.data[:, None], ycc), (1, 28, 28)
    if dataset == "fashion-mnist":
        train_ds = datasets.FashionMNIST(root=root, train=True, download=True)
        test_ds = datasets.FashionMNIST(root=root, train=False, download=True)
        return flatten_images(train_ds.data[:, None], ycc), flatten_images(test_ds.data[:, None], ycc), (1, 28, 28)
    if dataset in {"cifar", "cifar10"}:
        train_ds = datasets.CIFAR10(root=root, train=True, download=True)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True)
        train = torch.as_tensor(train_ds.data).permute(0, 3, 1, 2)
        test = torch.as_tensor(test_ds.data).permute(0, 3, 1, 2)
        return flatten_images(train, ycc), flatten_images(test, ycc), (3, 32, 32)
    raise ValueError(f"Unsupported torchvision dataset {dataset!r}")


def load_discrete_image_dataset(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    image_shape = tuple(args.image_shape) if args.image_shape is not None else None
    if args.dataset in {"tensor", "imagenet32", "imagenet64", "celeba"}:
        if args.dataset == "imagenet32" and image_shape is None:
            image_shape = (3, 32, 32)
        if args.dataset in {"imagenet64", "celeba"} and image_shape is None:
            image_shape = (3, 64, 64)
        train, valid, test, shape = _load_local_tensor_dataset(args.root, image_shape, args.ycc)
    else:
        train, test, shape = _load_vision_dataset(args.dataset, args.root, args.ycc)
        valid = None

    if valid is None:
        n_val = int(len(train) * args.valid_split)
        n_train = len(train) - n_val
        if n_val <= 0 or n_train <= 0:
            raise ValueError(
                f"valid_split={args.valid_split} gives train/val sizes {n_train}/{n_val}; "
                "use a larger dataset or a split in (0, 1)"
            )
        train_subset, val_subset = torch.utils.data.random_split(
            train,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train = train[torch.as_tensor(train_subset.indices)]
        valid = train_subset.dataset[torch.as_tensor(val_subset.indices)]

    return train.to(torch.uint8), valid.to(torch.uint8), test.to(torch.uint8), shape


def limit_samples(data: torch.Tensor, max_samples: int | None, seed: int) -> torch.Tensor:
    if max_samples is None or max_samples <= 0 or max_samples >= len(data):
        return data
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(data), generator=generator)[:max_samples]
    return data[indices]


def make_nll_function(
    circuit,
    partition_function,
    *,
    use_compile=False,
    backend="inductor",
    mode="default",
    fullgraph=False,
):
    """Build the shared train/eval loss, optionally compiling its tensor operations.

    Keep the original modules for optimizer parameters and checkpoint keys. The
    callable reads their current parameters, including after loading a checkpoint.
    Data transfer, logging and optimizer updates stay outside the compiled region.
    """
    def nll(batch):
        return -(circuit(batch) - partition_function()).mean()

    if not use_compile:
        return nll
    if backend != "inductor" and mode != "default":
        raise ValueError("Non-default compile modes require --compile-backend=inductor")
    logger.info(
        f"Enabling torch.compile for NLL: backend={backend}, mode={mode}, fullgraph={fullgraph}. "
        "The first training and evaluation batches include compilation overhead."
    )
    return torch.compile(nll, backend=backend, mode=mode, fullgraph=fullgraph)


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
    num_dimensions,
    log_to_wandb=True,
    activation=None,
    use_scheduler=False,
    nll_fn=None,
):
    if nll_fn is None:
        nll_fn = make_nll_function(circuit, partition_function)
    optimizer = optim.Adam(circuit.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=eta_min
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
        for (batch,) in tqdm(train_loader, desc="[Train]", leave=False):
            batch = batch.to(device).long()

            loss = nll_fn(batch)

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
                    for (val_batch,) in val_loader:
                        val_batch = val_batch.to(device).long()
                        val_loss_sum += nll_fn(val_batch).item() * val_batch.size(0)
                        val_count += val_batch.size(0)

                avg_train_nll = train_loss_sum / train_count
                avg_val_nll = val_loss_sum / val_count
                bpd_train = avg_train_nll / (num_dimensions * np.log(2.0))
                bpd_val = avg_val_nll / (num_dimensions * np.log(2.0))

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
                        f"New best model at step {total_steps}, Train NLL: {avg_train_nll:.4f}, "
                        f"Train bpd: {bpd_train:.4f}, Val NLL: {best_val_nll:.4f}, "
                        f"Val bpd: {bpd_val:.4f}"
                    )
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
def evaluate_circuit(
    circuit,
    circuit_partition_function,
    test_loader,
    device,
    num_dimensions,
    log_to_wandb=True,
    nll_fn=None,
):
    circuit.eval()
    if nll_fn is None:
        nll_fn = make_nll_function(circuit, circuit_partition_function)

    test_nll_sum = 0.0
    test_count = 0

    for (batch,) in tqdm(test_loader, desc="[Test]", leave=False):
        batch = batch.to(device).long()
        loss = nll_fn(batch)
        test_nll_sum += loss.item() * batch.size(0)
        test_count += batch.size(0)

    avg_test_nll = test_nll_sum / test_count
    avg_test_bpd = avg_test_nll / (num_dimensions * np.log(2.0))

    logger.info(f"Test NLL: {avg_test_nll:.4f} | bpd: {avg_test_bpd:.4f}")

    if log_to_wandb:
        assert wandb is not None
        wandb.log({"test_nll": avg_test_nll, "test_bpd": avg_test_bpd})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a probabilistic circuit on discrete image datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion-mnist", "cifar", "cifar10", "celeba", "imagenet32", "imagenet64", "tensor"],
        help="dataset to train on",
    )
    parser.add_argument("--rg", type=str, default="quad-tree-2", choices=["quad-tree-2", "quad-tree-4", "quad-graph"])
    parser.add_argument("--inner-layer", type=str, default="cp", choices=["cp", "tucker"])
    parser.add_argument("--k", type=int, default=512, help="num units per layer")
    parser.add_argument(
        "--activation",
        type=str,
        default="clamp",
        choices=["clamp", "softmax", "softplus", "sigmoid", "none"],
        help="activation function for sum units",
    )
    parser.add_argument("--weights-init", type=str, default="uniform", choices=["uniform", "normal", "dirichlet"])
    parser.add_argument("--root", type=str, default="datasets", help="dataset root or local tensor directory")
    parser.add_argument("--image-shape", type=int, nargs=3, metavar=("C", "H", "W"), default=None)
    parser.add_argument("--ycc", type=str, default="none", choices=["none", "lossy", "lossless"])
    parser.add_argument(
        "--input-sharing",
        type=str,
        default="none",
        choices=["none", "full", "channel", "global"],
        help="'full' matches ten-pics full_sharing for RGB inputs; 'channel' is accepted as an alias",
    )

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--T_0", type=int, default=1, help="T_0 for cosine annealing")
    parser.add_argument("--eta-min", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=None, help="limit training samples after loading")
    parser.add_argument("--max-val-samples", type=int, default=None, help="limit validation samples after loading")
    parser.add_argument("--max-test-samples", type=int, default=None, help="limit test samples after loading")
    parser.add_argument(
        "--structure-samples",
        type=int,
        default=None,
        help="number of training samples used to build and estimate the initial circuit",
    )
    parser.add_argument("--validation-steps", type=int, default=250, help="validation every n steps")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--valid-split", type=float, default=0.05)
    parser.add_argument("--use-scheduler", action="store_true", help="use cosine annealing scheduler")
    parser.add_argument(
        "--torch-compile", action="store_true",
        help="compile the NLL (circuit and partition function) for training, validation and test",
    )
    parser.add_argument(
        "--compile-backend", default="inductor", choices=["inductor", "eager", "aot_eager"],
        help="torch.compile backend; eager/aot_eager are for compiler diagnostics",
    )
    parser.add_argument(
        "--compile-mode", default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="Inductor compilation mode",
    )
    parser.add_argument(
        "--compile-fullgraph", action="store_true",
        help="require one compiled graph; fail on graph breaks instead of allowing partial capture",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="best_circuit.pt")
    parser.add_argument("--wandb", action="store_true", help="log to wandb")
    parser.add_argument("--project", type=str, default="pc_dataset_eval", help="wandb project name")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--noise-scale", type=float, default=2.0)
    parser.add_argument("--use-miwae", action="store_true", help="use MIWAE for LearnSPN")
    parser.add_argument("--adaptive-alpha", action="store_true", help="use adaptive alpha for LearnSPN")
    parser.add_argument("--use-mixing-weights", action="store_true", help="use mixing weights for LearnSPN")
    parser.add_argument("--use-estimated-weights", action="store_true", help="use estimated weights for LearnSPN")
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
    args = parser.parse_args()
    if args.torch_compile and args.compile_backend != "inductor" and args.compile_mode != "default":
        parser.error("Non-default --compile-mode requires --compile-backend=inductor")

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

    X_train, X_val, X_test, image_shape = load_discrete_image_dataset(args)
    X_train = limit_samples(X_train, args.max_train_samples, args.seed)
    X_val = limit_samples(X_val, args.max_val_samples, args.seed + 1)
    X_test = limit_samples(X_test, args.max_test_samples, args.seed + 2)
    logger.info(f"Loaded {args.dataset}: train={tuple(X_train.shape)}, val={tuple(X_val.shape)}, test={tuple(X_test.shape)}")
    logger.info(f"Using image_shape={image_shape}, ycc={args.ycc}, input_sharing={args.input_sharing}")

    train_loader = DataLoader(TensorDataset(X_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=args.batch_size, shuffle=False)

    weight_dir = os.path.join(os.getcwd(), "best_categorical_miwae.pt")

    if args.use_miwae and image_shape[0] != 1:
        raise ValueError("MIWAE support in this script is currently limited to single-channel images")

    learner_kwargs = dict(
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
    )

    logger.info("Using LearnSPN")
    from cirkit.templates.learn_spn import LearnSPN

    spn_learner = LearnSPN(**learner_kwargs)
    structure_data = limit_samples(X_train, args.structure_samples, args.seed + 3)
    logger.info(f"Using {len(structure_data)} samples to build/estimate the circuit")

    estimation_device = torch.device(args.estimation_device) if args.estimation_device else device
    if args.use_estimated_weights:
        structure_data_for_build = structure_data.to(estimation_device)
    else:
        structure_data_for_build = structure_data.to(device)

    symbolic_circuit = spn_learner.learn_spn(
        structure_data_for_build,
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

    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=True)

    circuit = ctx.compile(symbolic_circuit).to(device)
    circuit_partition_function = ctx.compile(symbolic_partition_function).to(device)
    nll_fn = make_nll_function(
        circuit,
        circuit_partition_function,
        use_compile=args.torch_compile,
        backend=args.compile_backend,
        mode=args.compile_mode,
        fullgraph=args.compile_fullgraph,
    )

    logger.info(f"Number of parameters: {sum(p.numel() for p in circuit.parameters())}")

    sum_params = [
        p for layer in circuit.layers if not isinstance(layer, TorchInputLayer)
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
        nll_fn=nll_fn,
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
        nll_fn=nll_fn,
    )
    if args.wandb:
        run.finish()
