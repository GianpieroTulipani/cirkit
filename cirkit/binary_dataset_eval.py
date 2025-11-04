import pandas as pd
import torch
import openml
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from cirkit.pipeline import PipelineContext
from cirkit.templates.learn_spn import LearnSPN

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
    device=torch.device(device)
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

        print(f"epoch {epoch} - Train NLL: {avg_train_nll:.4f}")

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

        print(f"Epoch {epoch} - Val NLL: {avg_val_nll:.4f}")

        if avg_val_nll < best_val_nll:
            best_val_nll = avg_val_nll
            torch.save(circuit.state_dict(), save_path)
            print(f"✅ New best model at epoch {epoch}, Val NLL: {best_val_nll:.4f}")
            
    return circuit

def evaluate_circuit(circuit,
                     test_loader: DataLoader,
                     device: str = "cpu",
                     checkpoint_path: str = "best_circuit.pth"
                    ):
    
    device = torch.device(device)
    circuit.load_state_dict(torch.load(checkpoint_path, map_location=device))
    circuit.to(device)
    circuit.eval()

    test_nll_sum = 0.0
    test_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]", leave=False):
            batch = batch.to(device)
            log_liks = circuit(batch)
            loss = -log_liks.mean()
            test_nll_sum += loss.item() * batch.size(0)
            test_count += batch.size(0)

    avg_test_nll = test_nll_sum / test_count
    print(f"Test NLL: {avg_test_nll:.4f}")
    return avg_test_nll
    
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

    learner = LearnSPN(
        alpha=0.5,
        min_instances=100,
        mi_quantile=0.5,
        device=device,
        data_format='tabular'
    )

    symbolic_circuit = learner.learn_spn(
        train_data.dataset[train_data.indices],
        input_layer='categorical',
        activation='softmax',
        initialization='estimated',
        num_input_units=1,
        num_sum_units=1
    )

    torch.cuda.empty_cache()
    gc.collect()

    circuit = train_circuit(
        symbolic_circuit,
        train_loader,
        val_loader,
        num_epochs=10,
        lr=1e-2,
        weight_decay=0.0,
        device=device
    )

    evaluate_circuit(
        circuit,
        test_loader,
        device=device,
        checkpoint_path='best_circuit.pth'
        )