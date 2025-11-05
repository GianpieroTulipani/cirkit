import pandas as pd
import torch
import openml
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from cirkit.pipeline import PipelineContext
from cirkit.templates.learn_spn import LearnSPN

def compute_circuit_likelihood(circuit, train_data, test_data, batch_size=512, device='cpu'):
    device = torch.device(device)
    circuit.to(device)
    circuit.eval()

    def compute_log_likelihood(circuit, dataset):
        data_loader = DataLoader(dataset, batch_size=batch_size)
        log_likelihoods = []
        with torch.no_grad():
            for batch in data_loader:
                # Handle whether batch is a tuple (from TensorDataset) or raw Tensor
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                ll = circuit(batch).cpu()
                log_likelihoods.append(ll)
        return torch.cat(log_likelihoods).mean().item()

    print("Computing train log-likelihood in batches ...")
    train_ll = compute_log_likelihood(circuit, train_data)

    print("Computing test log-likelihood in batches ...")
    test_ll = compute_log_likelihood(circuit, test_data)

    print(f"Avg. log-likelihoods:\n"
          f"  Train: {train_ll:.4f}\n"
          f"  Test:  {test_ll:.4f}")


def train_circuit(
    symbolic_circuit,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    device: str = 'cpu',
    save_path: str = "best_circuit.pth"
):
    device = torch.device(device)
    ctx = PipelineContext(
        backend='torch',
        semiring='lse-sum',
        fold=True,
        optimize=True
    )

    circuit = ctx.compile(symbolic_circuit).to(device)
    optimizer = torch.optim.Adam(circuit.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_nll = float("inf")

    for epoch in range(1, num_epochs + 1):
        circuit.train()
        train_log_liks = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            batch = batch.to(device)
            log_liks = circuit(batch)
            loss = -log_liks.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_log_liks.append(log_liks.detach())

        train_log_liks = torch.cat(train_log_liks)
        avg_train_log_ll = train_log_liks.mean().item()
        avg_train_nll = -avg_train_log_ll
        avg_train_ll = torch.exp(train_log_liks).mean().item()

        print(f"Epoch {epoch} - Train NLL: {avg_train_nll:.4f} | "
              f"Log-LL: {avg_train_log_ll:.4f} | Likelihood: {avg_train_ll:.6e}")

        circuit.eval()
        val_log_liks = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                batch = batch.to(device)
                log_liks = circuit(batch)
                val_log_liks.append(log_liks)

        val_log_liks = torch.cat(val_log_liks)
        avg_val_log_ll = val_log_liks.mean().item()
        avg_val_nll = -avg_val_log_ll
        avg_val_ll = torch.exp(val_log_liks).mean().item()

        print(f"Epoch {epoch} - Val NLL: {avg_val_nll:.4f} | "
              f"Log-LL: {avg_val_log_ll:.4f} | Likelihood: {avg_val_ll:.6e}")

        if avg_val_nll < best_val_nll:
            best_val_nll = avg_val_nll
            torch.save(circuit.state_dict(), save_path)
            print(f"✅ New best model at epoch {epoch}, Val NLL: {best_val_nll:.4f}")

    return circuit


def evaluate_circuit(
    circuit,
    test_loader: DataLoader,
    device: str = "cpu",
    checkpoint_path: str = "best_circuit.pth"
):
    device = torch.device(device)
    circuit.load_state_dict(torch.load(checkpoint_path, map_location=device))
    circuit.to(device)
    circuit.eval()

    test_log_liks = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test Evaluation]", leave=False):
            batch = batch.to(device)
            log_liks = circuit(batch)
            test_log_liks.append(log_liks)

    test_log_liks = torch.cat(test_log_liks)
    avg_log_likelihood = test_log_liks.mean().item()
    avg_nll = -avg_log_likelihood
    avg_likelihood = torch.exp(test_log_liks).mean().item()

    print(f"📊 Test Results:\n"
          f"  Log-Likelihood: {avg_log_likelihood:.4f}\n"
          f"  Negative Log-Likelihood: {avg_nll:.4f}\n"
          f"  Likelihood: {avg_likelihood:.6e}")

    return {
        "avg_log_likelihood": avg_log_likelihood,
        "avg_likelihood": avg_likelihood,
        "avg_nll": avg_nll
    }

    
if __name__ == "__main__":
    batch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = openml.datasets.get_dataset(40668, download_all_files=True).get_data()[0]
    dataset = pd.get_dummies(dataset)
    tensor_dataset = torch.tensor(dataset.values, dtype=torch.long)

    n_total = len(tensor_dataset)
    n_train = 16000
    n_val = 4000
    n_test = n_total - n_train - n_val

    train_data, val_data, test_data = torch.utils.data.random_split(
        tensor_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader  = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("🧮 Please provide SPN learning hyperparameters (press Enter for defaults):")

    try:
        alpha = float(input("Enter α (Laplace smoothing, default=0.5): ") or 0.5)
    except ValueError:
        alpha = 0.5

    try:
        min_instances = int(input("Enter minimum instances per region (default=100): ") or 100)
    except ValueError:
        min_instances = 100

    try:
        mi_quantile = float(input("Enter MI quantile threshold (default=0.5): ") or 0.5)
    except ValueError:
        mi_quantile = 0.5

    try:
        jitter_scale = float(input("Enter the jitter scale (default=1e-1): ") or 1e-1)
    except ValueError:
        jitter_scale = 1e-1

    try:
        weight_decay = float(input("Enter the weight decay (default=1e-6): ") or 1e-6)
    except ValueError:
        weight_decay = 1e-6

    try:
        initialization = str(input("Enter intialization value (default='estimated'): ") or 'estimated')
    except ValueError:
        initialization = 'estimated'

    try:
        num_input_units = int(input("Enter the number of input units (default=1): ") or 1)
    except ValueError:
        num_input_units = 1
    
    try:
        num_sum_units = int(input("Enter the number of sum units (default=1): ") or 1)
    except ValueError:
        num_sum_units = 1

    print("Learning PCs structure...")

    learner = LearnSPN(
        alpha=alpha,
        min_instances=min_instances,
        mi_quantile=mi_quantile,
        jitter_scale=jitter_scale,
        device=device,
        data_format='tabular'
    )

    symbolic_circuit = learner.learn_spn(
        train_data.dataset[train_data.indices],
        input_layer='categorical',
        activation='softmax',
        initialization=initialization,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units
    )

    print(f'The Circuit have {len(list(symbolic_circuit.layers))} layers')

    torch.cuda.empty_cache()
    gc.collect()

    """ctx = PipelineContext(
        backend='torch',
        semiring='lse-sum',
        fold=True,
        optimize=True
    )

    circuit = ctx.compile(symbolic_circuit).to(device)

    compute_circuit_likelihood(
        circuit,
        train_data,
        test_data,
        device=device
    )"""

    print("Training the circuit...")

    circuit = train_circuit(
        symbolic_circuit,
        train_loader,
        val_loader,
        num_epochs=10,
        lr=1e-2,
        weight_decay=weight_decay,
        device=device
    )

    print("Evaluating on test set...")

    evaluate_circuit(
        circuit,
        test_loader,
        device=device,
        checkpoint_path='best_circuit.pth'
        )
    

"""
Enter α (Laplace smoothing, default=0.5): 50.0
Enter minimum instances per region (default=100): 200
Enter MI quantile threshold (default=0.5): 0.6
Enter the jitter scale (default=1e-1): 5e-1
Enter the weight decay (default=1e-6): 0.0
Enter intialization value (default='estimated'): estimated
Enter the number of input units (default=1): 4
Enter the number of sum units (default=1): 4
Learning PCs structure...
The Circuit have 9550 layers
Training the circuit...
epoch 1 - Train NLL: 26.2422
Epoch 1 - Val NLL: 22.5393
✅ New best model at epoch 1, Val NLL: 22.5393
epoch 2 - Train NLL: 21.0519
Epoch 2 - Val NLL: 20.3991
✅ New best model at epoch 2, Val NLL: 20.3991
epoch 3 - Train NLL: 19.6565
Epoch 3 - Val NLL: 19.6223
✅ New best model at epoch 3, Val NLL: 19.6223
epoch 4 - Train NLL: 19.0488
Epoch 4 - Val NLL: 19.2410
✅ New best model at epoch 4, Val NLL: 19.2410
epoch 5 - Train NLL: 18.7073
Epoch 5 - Val NLL: 19.0098
✅ New best model at epoch 5, Val NLL: 19.0098
epoch 6 - Train NLL: 18.4874
Epoch 6 - Val NLL: 18.8496
✅ New best model at epoch 6, Val NLL: 18.8496
epoch 7 - Train NLL: 18.3346
Epoch 7 - Val NLL: 18.7352
✅ New best model at epoch 7, Val NLL: 18.7352
epoch 8 - Train NLL: 18.2279
Epoch 8 - Val NLL: 18.6708
✅ New best model at epoch 8, Val NLL: 18.6708
epoch 9 - Train NLL: 18.1473
Epoch 9 - Val NLL: 18.6143
✅ New best model at epoch 9, Val NLL: 18.6143
epoch 10 - Train NLL: 18.0768
Epoch 10 - Val NLL: 18.5568
✅ New best model at epoch 10, Val NLL: 18.5568
Evaluating on test set...
Test NLL: 18.5184

Test NLL: 16.7325 with 16 units
Test NLL: 16.4114 with 32 units
Test NLL: 20.4713 normal with 32 units
"""