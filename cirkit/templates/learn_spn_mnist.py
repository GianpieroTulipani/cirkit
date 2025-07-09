from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from cirkit.templates.learn_spn import LearnSPN
from cirkit.symbolic.compile import compile
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_train = datasets.MNIST(root="datasets", train=True,  download=True)
mnist_test  = datasets.MNIST(root="datasets", train=False, download=True)

X_train = (
    mnist_train.data
      .view(-1, 28*28)
      .long()
      .to(device)
)

X_test  = (
    mnist_test.data
      .view(-1, 28*28)
      .long()
      .to(device)
)

spn_learner = LearnSPN(alpha=0.5, min_instances=1000, mi_quantile=0.5)
print("Learning SPN structure on nltcs train split ...")
symbolic_circuit = spn_learner.learn(X_train, 'categorical')
circuit = compile(symbolic_circuit).to(device)

def compute_log_likelihood_in_batches(circuit, data, batch_size=512):
  data_loader = DataLoader(TensorDataset(data), batch_size=batch_size)
  log_likelihoods = []

  with torch.no_grad():
      for (batch,) in data_loader:
          ll = circuit(batch).cpu()
          log_likelihoods.append(ll)

  return torch.cat(log_likelihoods).mean().item()

print("Computing train log-likelihood in batches ...")
train_ll = compute_log_likelihood_in_batches(circuit, X_train)

print("Computing test log-likelihood in batches ...")
test_ll = compute_log_likelihood_in_batches(circuit, X_test)

print(f"Avg. log-likelihoods:\n"
      f"  Train: {train_ll:.4f}\n"
      f"  Test:  {test_ll:.4f}")

bpd_train = (-train_ll) / (28 * 28 * np.log(2.0))
bdp_test = (-test_ll) / (28 * 28 * np.log(2.0))

print(f"BPDs:\n"
      f"  Train: {bpd_train:.4f}\n"
      f"  Test:  {bdp_test:.4f}")