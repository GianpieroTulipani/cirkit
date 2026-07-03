import os
import sys
import math
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

try:
    
except Exception:
    def tqdm(x, **kwargs):
        return x

from cirkit.pipeline import PipelineContext
import cirkit.symbolic.functional as sf
from cirkit.templates.region_graph.algorithms import QuadTree, QuadGraph
from cirkit.templates.utils import (
    Parameterization,
    parameterization_to_factory,
    name_to_input_layer_factory,
)
from cirkit.backend.torch.layers import TorchInputLayer

LOG2 = math.log(2.0)


# ====================================================================================
#  Costruzione del circuito simbolico QT-CP-512
# ====================================================================================
def build_symbolic_circuit(args):
    """Costruisce il circuito simbolico (region graph + CP + input categorico)."""
    C, H, W = 1, 28, 28
    shape = (C, H, W)

    # --- region graph -----------------------------------------------------------------
    if args.rg == "quad-tree":
        rg = QuadTree(shape, num_patch_splits=args.num_patch_splits)
    elif args.rg == "quad-graph":
        rg = QuadGraph(shape)
    else:
        raise ValueError(f"region graph sconosciuto: {args.rg}")

    # --- parametrizzazione dei pesi delle somme --------------------------------------
    if args.weight_mode == "clamp":
        # Fedele al paper: pesi grezzi positivi, non normalizzati (clamp fatto dopo lo step).
        sum_param = Parameterization(activation="none", initialization="uniform")
    elif args.weight_mode == "softmax":
        # Idiomatico cirkit: pesi localmente normalizzati via softmax.
        sum_param = Parameterization(activation="softmax", initialization="normal")
    else:
        raise ValueError(f"weight-mode sconosciuto: {args.weight_mode}")
    sum_weight_factory = parameterization_to_factory(sum_param)

    # --- layer di input categorico (256 livelli di grigio) ---------------------------
    # In ENTRAMBE le modalita' l'input usa la parametrizzazione softmax di default
    # (categoriale localmente normalizzata). NOTA di fedelta': tenpcs tiene anche
    # l'input non normalizzato e lo clampa; in cirkit una categoriale con `probs`
    # NON normalizzate rende il circuito IMPROPRIO (la sua integrazione assume probs
    # gia' normalizzate), quindi qui l'input resta normalizzato. Cio' non cambia la
    # famiglia del modello (QT-CP-512 categoriale): la parte che il paper allena in
    # modo non-normalizzato — e che qui replichiamo — sono i PESI DELLE SOMME.
    input_factory = name_to_input_layer_factory(
        "categorical", num_categories=args.num_categories
    )

    # --- assemblaggio -----------------------------------------------------------------
    symbolic_circuit = rg.build_circuit(
        input_factory=input_factory,
        sum_product=args.inner_layer,          # 'cp' oppure 'tucker'
        sum_weight_factory=sum_weight_factory,
        nary_sum_weight_factory=sum_weight_factory,  # usato solo da quad-graph (mixing)
        num_input_units=args.k,
        num_sum_units=args.k,
        num_classes=1,
        factorize_multivariate=True,
    )
    return symbolic_circuit


# ====================================================================================
#  Valutazione: NLL medio (nats) e bits-per-dimension
# ====================================================================================
@torch.no_grad()
def evaluate(circuit, partition, loader, device, num_features):
    circuit.eval()
    total_ll, count = 0.0, 0
    logZ = partition()  # scalare (log costante di normalizzazione), shape (1, 1)
    for batch in loader:
        batch = batch.to(device)
        lls = (circuit(batch) - logZ).flatten()   # (B,)
        total_ll += lls.sum().item()
        count += batch.size(0)
    mean_ll = total_ll / count                     # log-lik medio per campione (nats)
    nll = -mean_ll
    bpd = nll / (LOG2 * num_features)
    return nll, bpd


# ====================================================================================
#  Training loop (fedele a trainers.training_pc)
# ====================================================================================
def train(circuit, partition, sum_params, train_loader, valid_loader, args, device, num_features):
    optimizer = torch.optim.Adam(circuit.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    max_train_steps = len(train_loader) * args.max_epochs
    best_val_nll = float("inf")
    epochs_no_improve = 0
    total_steps = 0

    print(f"Inizio training: fino a {max_train_steps} step "
          f"({len(train_loader)} step/epoca x {args.max_epochs} epoche max).")

    stop = False
    while total_steps < max_train_steps and not stop:
        circuit.train()
        for batch in tqdm(train_loader, desc="[Train]", leave=False):
            batch = batch.to(device)

            lls = (circuit(batch) - partition()).flatten()  # (B,)
            loss = -lls.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- clamp fedele al paper: tieni POSITIVI i pesi delle somme dopo ogni step
            # (solo i pesi sum/CP, NON i logit softmax dell'input categorico).
            if args.weight_mode == "clamp":
                with torch.no_grad():
                    for p in sum_params:
                        p.data.clamp_(min=args.clamp_min)

            total_steps += 1

            if total_steps % args.valid_freq == 0:
                val_nll, val_bpd = evaluate(circuit, partition, valid_loader, device, num_features)
                train_nll = loss.item()
                train_bpd = train_nll / (LOG2 * num_features)

                if val_nll < best_val_nll - args.min_delta:
                    best_val_nll = val_nll
                    epochs_no_improve = 0
                    torch.save(circuit.state_dict(), args.save_path)
                    tag = "  <-- best (salvato)"
                else:
                    epochs_no_improve += 1
                    tag = f"  (no-improve {epochs_no_improve}/{args.patience})"

                print(f"step {total_steps:>6} | train NLL {train_nll:.3f} | train bpd {train_bpd:.4f} | "
                      f"val NLL {val_nll:.3f} | val bpd {val_bpd:.4f}{tag}")

                circuit.train()
                if epochs_no_improve >= args.patience:
                    print("Early stopping.")
                    stop = True
                    break

            if total_steps >= max_train_steps:
                stop = True
                break

    # se non e' mai stato salvato nulla (es. --smoke), salva l'ultimo stato
    if not os.path.exists(args.save_path):
        torch.save(circuit.state_dict(), args.save_path)


# ====================================================================================
#  Dati MNIST: pixel interi 0-255 come categorie, flatten a (N, 784)
# ====================================================================================
def load_mnist(args):
    mnist_train = datasets.MNIST(root=args.data_root, train=True, download=True)
    mnist_test = datasets.MNIST(root=args.data_root, train=False, download=True)

    X_train = mnist_train.data.view(-1, 28 * 28).long()   # (60000, 784), valori 0-255
    X_test = mnist_test.data.view(-1, 28 * 28).long()      # (10000, 784)

    n_val = int(len(X_train) * args.valid_split)
    n_train = len(X_train) - n_val
    train_data, val_data = random_split(
        X_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.smoke:  # sottoinsieme minuscolo per il test rapido
        train_data = torch.utils.data.Subset(train_data, list(range(args.batch_size * 4)))
        val_data = torch.utils.data.Subset(val_data, list(range(args.batch_size)))
        X_test = X_test[: args.batch_size]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(X_test, batch_size=args.batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


# ====================================================================================
#  Main
# ====================================================================================
def main():
    parser = argparse.ArgumentParser(description="Replica QT-CP-512 (PC non-PIC) su MNIST con cirkit")
    # architettura
    parser.add_argument("--rg", type=str, default="quad-tree",
                        choices=["quad-tree", "quad-graph"], help="region graph (paper: QT)")
    parser.add_argument("--num-patch-splits", type=int, default=2, choices=[2, 4],
                        help="split per quad-tree (2 = binario, come tenpcs)")
    parser.add_argument("--inner-layer", type=str, default="cp",
                        choices=["cp", "tucker"], help="layer sum-product (paper: CP)")
    parser.add_argument("--k", type=int, default=512, help="num unita' (input e sum). Paper: 512")
    parser.add_argument("--num-categories", type=int, default=256, help="livelli pixel (grigio: 256)")
    parser.add_argument("--weight-mode", type=str, default="clamp",
                        choices=["clamp", "softmax"],
                        help="'clamp' = fedele al paper; 'softmax' = idiomatico cirkit")
    parser.add_argument("--clamp-min", type=float, default=1e-19, help="floor positivo dei pesi")
    # training
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate costante (paper: 0.01)")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--valid-freq", type=int, default=250, help="validazione ogni n step")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)

    parser.add_argument("--valid-split", type=float, default=0.05)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="qtcp512_mnist.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--smoke", action="store_true", help="test rapido di coerenza")
    args = parser.parse_args()

    if args.smoke:
        args.k = min(args.k, 8)
        args.max_epochs = 1
        args.valid_freq = 2
        args.patience = 10

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    for kk, vv in vars(args).items():
        print(f"  {kk}: {vv}")
    print(f"  device: {device}")
    print("=" * 70)

    # dati
    train_loader, valid_loader, test_loader = load_mnist(args)
    num_features = 28 * 28  # 784 (1 canale)

    # circuito simbolico + funzione di partizione, compilati nello STESSO contesto
    symbolic_circuit = build_symbolic_circuit(args)
    symbolic_partition = sf.integrate(symbolic_circuit)

    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=True)
    circuit = ctx.compile(symbolic_circuit).to(device)
    partition = ctx.compile(symbolic_partition).to(device)

    n_params = sum(p.numel() for p in circuit.parameters())
    print(f"Circuito compilato. Parametri allenabili: {n_params:,}")

    # parametri dei soli layer NON di input (pesi sum/CP): sono gli unici da clampare
    # in weight-mode 'clamp' (l'input categorico resta softmax-normalizzato).
    sum_params = [
        p for layer in circuit.layers if not isinstance(layer, TorchInputLayer)
        for p in layer.parameters()
    ]

    # training
    train(circuit, partition, sum_params, train_loader, valid_loader, args, device, num_features)

    # test sul miglior checkpoint
    circuit.load_state_dict(torch.load(args.save_path, map_location=device))
    test_nll, test_bpd = evaluate(circuit, partition, test_loader, device, num_features)
    print("-" * 70)
    print(f"RISULTATO FINALE  |  Test NLL: {test_nll:.4f}  |  Test bpd: {test_bpd:.4f}")
    print(f"(riferimento paper QT-CP-512, PC: 1.175 bpd)")


if __name__ == "__main__":
    main()
