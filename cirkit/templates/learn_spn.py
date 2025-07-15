import time
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Optional

import torch
from loguru import logger
import numpy as np
from torch import LongTensor, Tensor
from fast_pytorch_kmeans import KMeans

from cirkit.pipeline import compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, SumLayer, Layer
from cirkit.symbolic.parameters import TensorParameter, Parameter
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.templates.utils import name_to_input_layer_factory, InputLayerFactory
from cirkit.utils.scope import Scope

@dataclass
class Task:
  id: int
  V_s: LongTensor
  T_s: LongTensor
  parent: Optional[Layer]

class LearnSPN:
  def __init__(
      self,
      alpha: float = 1.0,
      min_instances: int = 10,
      mi_quantile: float = 0.8,
      local_radius: int = 3,
      image_shape: Tuple[int,int] = (28,28),
      seed: Optional[int] = 42
  ):
      """
      SPN learner using local mutual-information-based vertical splits.

      Args:
          alpha: Laplace smoothing constant for MI and leaf estimates.
          min_instances: Minimum instances to keep splitting.
          mi_quantile: Quantile threshold for MI connectivity.
          local_radius: L1-neighborhood radius on 2D feature grid.
          image_shape: Height and width of the image grid (e.g., (28,28) for MNIST).
      """

      if seed is not None:
        self._set_seed(seed)

      self.alpha = alpha
      self.min_instances = min_instances
      self.mi_quantile = mi_quantile
      self.local_radius = local_radius
      self.image_shape = image_shape
      self.total_feature_split_time = 0.0
      self.total_instance_split_time = 0.0

      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      self._build_neighbor_map()

  def _set_seed(self, seed: int):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

  def _build_neighbor_map(self):
    """
    Build a map from each feature index to its neighbors within L1 radius.
    """
    H, W = self.image_shape
    n = H*W
    coords = {i: (i // W, i % W) for i in range(n)}
    self.neighbor_map: Dict[int,List[int]] = {}
    for i in range(n):
      y_i, x_i = coords[i]
      nbrs = []
      for j in range(n):
        y_j, x_j = coords[j]
        if abs(y_i - y_j) + abs(x_i - x_j) <= self.local_radius:
          nbrs.append(j)
      self.neighbor_map[i] = nbrs

  def learn(
      self,
      data: LongTensor,
      input_layer: str = 'categorical',
      num_input_units: int = 1,
      num_sum_units: int = 1
  ) -> Circuit:
      N, D = data.shape
      num_categories = int(data.max().item() + 1)
      input_factory = name_to_input_layer_factory(input_layer, num_categories=num_categories)

      tid = 0
      layers: List[Layer] = []
      in_layers: Dict[Layer, List[Layer]] = {}
      root: List[Layer] = []
      stack = [Task(tid, torch.arange(D, device=self.device), torch.arange(N, device=self.device), None)]
      tid += 1

      while stack:
          task = stack.pop()
          V_s, T_s, parent = task.V_s, task.T_s, task.parent

          if V_s.numel() == 1:
              leaf = self._make_leaf_layer(int(V_s), T_s, data,
                                            num_input_units, num_categories, input_factory)
              layers.append(leaf)
              in_layers.setdefault(parent, []).append(leaf)
              continue

          if T_s.numel() <= self.min_instances:
              feats = [self._make_leaf_layer(int(f), T_s, data,
                            num_input_units, num_categories, input_factory)
                        for f in V_s.tolist()]
              node = HadamardLayer(num_input_units, arity=len(feats))
              layers.extend(feats); layers.append(node)
              in_layers[node] = feats
              in_layers.setdefault(parent, []).append(node)
              continue


          if task.id !=0:
            start_fs = time.time()
            V_dep, V_indep = self._split_features_local(V_s, T_s, data, num_categories)
            fs_time = time.time() - start_fs
            self.total_feature_split_time += fs_time
            #print(f"[Feature Split] Task {task.id} | Time: {fs_time:.4f} sec | Dep: {V_dep.numel()} Indep: {V_indep.numel()}")

            if V_indep.numel() > 0:
                node = HadamardLayer(num_input_units, arity=2)
                layers.append(node); in_layers[node] = []
                in_layers.setdefault(parent, []).append(node)
                stack.append(Task(tid, V_indep, T_s, node)); tid+=1
                stack.append(Task(tid, V_dep, T_s, node)); tid+=1
                continue

          start_is = time.time()
          T1, T2 = self._cluster_instances(V_s, T_s, data)
          is_time = time.time() - start_is
          self.total_instance_split_time += is_time
          #print(f"[Instance Split] Task {task.id} | Time: {is_time:.4f} sec | T1: {T1.numel()} T2: {T2.numel()}")

          if T1.numel() == 0 or T2.numel() == 0:
              feats = [ self._make_leaf_layer(int(f), T_s, data,
                            num_input_units, num_categories, input_factory)
                        for f in V_s.tolist() ]
              node = HadamardLayer(num_input_units, arity=len(feats))
              layers.extend(feats); layers.append(node)
              in_layers[node] = feats
              in_layers.setdefault(parent, []).append(node)
              continue

          w1 = T1.numel()/float(T1.numel()+T2.numel()); w2 = 1.0-w1
          mix = np.array([[w1,w2]],dtype=float)
          tp = TensorParameter(num_sum_units, 2, initializer=ConstantTensorInitializer(mix),learnable=True)
          mixing = Parameter.from_input(tp)
          node = SumLayer(num_input_units=num_sum_units,num_output_units=1,arity=2,weight=mixing)
          layers.append(node)
          if parent:
            in_layers.setdefault(parent, []).append(node)
          else:
            root.append(node)
          stack.append(Task(tid, V_s, T2, node)); tid+=1
          stack.append(Task(tid, V_s, T1, node)); tid+=1

      print("\n=== Split Timing Summary ===")
      print(f"Total Feature Split Time: {self.total_feature_split_time:.4f} sec")
      print(f"Total Instance Split Time: {self.total_instance_split_time:.4f} sec\n")

      return Circuit(layers, in_layers, root)

  def _split_features_local(
      self,
      V_s: LongTensor,
      T_s: LongTensor,
      data: Tensor,
      num_categories: int,
      chunk_size: int = 1000
  ) -> Tuple[LongTensor, LongTensor]:
      """
      Vertical split using local MI: compute MI only for neighbor pairs in parallel.
      Returns (dependent_features, independent_features).
      """
      sub = data.index_select(0, T_s).index_select(1, V_s)
      n = V_s.numel()

      idx_map = {int(v): i for i, v in enumerate(V_s.tolist())}

      pairs = []
      for i, feat_i in enumerate(V_s.tolist()):
        for feat_j in self.neighbor_map[int(feat_i)]:
          if feat_j in idx_map:
            j = idx_map[feat_j]
            if j > i:
              pairs.append((i, j))

      if not pairs:
          return V_s, torch.empty(0, dtype=torch.long, device=data.device)

      idx_i = torch.tensor([p[0] for p in pairs], device=sub.device)
      idx_j = torch.tensor([p[1] for p in pairs], device=sub.device)

      mi_mat = torch.zeros((n, n), device=sub.device)

      for start in range(0, len(idx_i), chunk_size):
          end = start + chunk_size
          i_chunk = idx_i[start:end]
          j_chunk = idx_j[start:end]

          mi_vals = _pairwise_mutual_info(
              sub[:, i_chunk], sub[:, j_chunk],
              self.alpha, num_categories
          )

          mi_mat[i_chunk, j_chunk] = mi_vals
          mi_mat[j_chunk, i_chunk] = mi_vals

      triu = mi_mat.triu(diagonal=1)
      vals = triu.flatten()[triu.flatten() > 0]
      threshold = float(torch.quantile(vals, self.mi_quantile)) if vals.numel() > 0 else 0.0

      adj = mi_mat > threshold

      visited = torch.zeros(n, dtype=torch.bool, device=sub.device)
      stack = [0]
      visited[0] = True
      while stack:
          u = stack.pop()
          nbrs = torch.nonzero(adj[u] & ~visited, as_tuple=False).squeeze(1)
          for v in nbrs.tolist():
              visited[v] = True
              stack.append(v)

      V_dep = V_s[visited]
      V_indep = V_s[~visited]
      return V_dep, V_indep


  def _cluster_instances(
      self,
      V_s: LongTensor,
      T_s: LongTensor,
      data: Tensor
  ) -> Tuple[LongTensor, LongTensor]:
      """
      Binary instance split via k-means on features V_s,
      seeded for reproducibility.
      """
      sub = data.index_select(0, T_s).index_select(1, V_s).float()
      kmeans = KMeans(
          n_clusters=2,
          mode='euclidean',
          verbose=0
      )
      labels = kmeans.fit_predict(sub)
      return T_s[labels == 0], T_s[labels == 1]

  def _make_leaf_layer(
      self,
      feat_idx: int,
      instance_ids: LongTensor,
      data: LongTensor,
      num_input_units: int,
      num_categories: int,
      input_factory: InputLayerFactory
  ) -> Any:
      col = data[instance_ids, feat_idx]
      counts = torch.bincount(col, minlength=num_categories).float()
      counts += self.alpha
      probs = counts / counts.sum()
      proto = probs.cpu().numpy().astype(float)
      tp = TensorParameter(num_input_units, probs.numel(),
                            initializer=ConstantTensorInitializer(proto),
                            learnable=True)
      param = Parameter.from_input(tp)
      return input_factory(Scope([feat_idx]), num_input_units, probs=param)

def _pairwise_mutual_info(
    x1: LongTensor,
    x2: LongTensor,
    alpha: float,
    num_categories: int
) -> Tensor:
    """
    Compute MI for each (x1[i], x2[i]) pair. Assumes x1 and x2 are (N, K) tensors.
    Returns shape: (K,)
    """
    N, K = x1.shape
    x1_flat = x1.T.contiguous()
    x2_flat = x2.T.contiguous()
    joint = x1_flat * num_categories + x2_flat  # (K, N)
    counts = torch.zeros(K, num_categories * num_categories, device=x1.device)
    counts.scatter_add_(1, joint, torch.ones_like(joint, dtype=torch.float))
    counts = counts.view(K, num_categories, num_categories)

    x1_counts = counts.sum(dim=2); x2_counts = counts.sum(dim=1)

    joint_probs = (counts + alpha) / (N + num_categories ** 2 * alpha)
    x1_probs = (x1_counts + num_categories * alpha) / (N + num_categories ** 2 * alpha)
    x2_probs = (x2_counts + num_categories * alpha) / (N + num_categories ** 2 * alpha)

    x1_probs = x1_probs.unsqueeze(2); x2_probs = x2_probs.unsqueeze(1)
    prod = x1_probs * x2_probs

    mi = joint_probs * (joint_probs.log() - prod.log())
    return mi.sum(dim=(1,2))


from torch.utils.data import DataLoader, TensorDataset
from cirkit.pipeline import compile
from torchvision import datasets
import itertools
import torch
import gc

def compute_log_likelihood_in_batches(circuit, data, batch_size=512):
  data_loader = DataLoader(TensorDataset(data), batch_size=batch_size)
  log_likelihoods = []

  with torch.no_grad():
      for (batch,) in data_loader:
          ll = circuit(batch).cpu()
          log_likelihoods.append(ll)

  return torch.cat(log_likelihoods).mean().item()


if __name__ == "__main__":
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

    alphas = [0,1, 0.5]
    min_instances_list = [100, 500, 1000]
    mi_quantiles = [0.7, 0.5, 0.3]
    local_radii = [3, 4]

    results = []

    for alpha, min_instances, mi_quantile, local_radius in itertools.product(alphas, min_instances_list, mi_quantiles, local_radii):
        logger.info(f"Training with alpha={alpha}, min_instances={min_instances}, mi_quantile={mi_quantile}, local_radius={local_radius}")
        
        spn_learner = LearnSPN(
            alpha=alpha,
            min_instances=min_instances,
            mi_quantile=mi_quantile,
            local_radius=local_radius
        )
        
        symbolic_circuit = spn_learner.learn(X_train, 'categorical')
        circuit = compile(symbolic_circuit).to(device)

        test_ll = compute_log_likelihood_in_batches(circuit, X_test)

        bpd_test = (-test_ll) / (28 * 28 * np.log(2.0))

        result = {
            "alpha": alpha,
            "min_instances": min_instances,
            "mi_quantile": mi_quantile,
            "local_radius": local_radius,
            "test_ll": test_ll,
            "bpd_test": bpd_test,
        }

        del spn_learner
        del symbolic_circuit
        del circuit
        del test_ll
        del bpd_test
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Results: {result}")
        results.append(result)

    results.sort(key=lambda x: x['bpd_test'])

    logger.info("\nTop 3 configurations:")
    for res in results[:3]:
        logger.info(res)