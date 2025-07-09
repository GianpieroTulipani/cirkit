import time
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Optional

import torch
import numpy as np
from torch import LongTensor, Tensor
from fast_pytorch_kmeans import KMeans

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, SumLayer, Layer
from cirkit.symbolic.parameters import TensorParameter, Parameter
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.templates.utils import name_to_input_layer_factory
from cirkit.templates.region_graph.algorithms.chow_liu import _categorical_mutual_info
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
                                             num_input_units, num_categories, input_layer)
                layers.append(leaf)
                in_layers.setdefault(parent, []).append(leaf)
                continue

            if T_s.numel() <= self.min_instances:
                feats = [self._make_leaf_layer(int(f), T_s, data,
                              num_input_units, num_categories, input_layer)
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
              print(f"[Feature Split] Task {task.id} | Time: {fs_time:.4f} sec | Dep: {V_dep.numel()} Indep: {V_indep.numel()}")

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
            print(f"[Instance Split] Task {task.id} | Time: {is_time:.4f} sec | T1: {T1.numel()} T2: {T2.numel()}")

            if T1.numel() == 0 or T2.numel() == 0:
                feats = [ self._make_leaf_layer(int(f), T_s, data,
                              num_input_units, num_categories, input_layer)
                          for f in V_s.tolist() ]
                node = HadamardLayer(num_input_units, arity=len(feats))
                layers.extend(feats); layers.append(node)
                in_layers[node] = feats
                in_layers.setdefault(parent, []).append(node)
                continue

            w1 = T1.numel()/float(T1.numel()+T2.numel()); w2 = 1.0-w1
            mix = np.array([[w1,w2]],dtype=float)
            tp = TensorParameter(num_sum_units,2,initializer=ConstantTensorInitializer(mix),learnable=False)
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
        num_categories: int
    ) -> Tuple[LongTensor, LongTensor]:
        """
        Vertical split using local MI: only compute MI for neighbor pairs.
        Returns (dependent_features, independent_features).
        """
        sub = data.index_select(0, T_s).index_select(1, V_s)
        n = V_s.numel()

        mi_vals = []
        idx_map = {int(v.item()): i for i, v in enumerate(V_s)}

        mi_mat = torch.zeros(n, n, device=sub.device)
        for i, v_i in enumerate(V_s.tolist()):
            nbrs = self.neighbor_map[int(v_i)]
            for v_j in nbrs:
              if v_j not in idx_map:
                continue
              j = idx_map[v_j]
              if j <= i:
                continue

              pair = sub[:, [i, j]]
              mi_ij = _categorical_mutual_info(pair, alpha=self.alpha, num_categories=num_categories)
              val = mi_ij[0,1]
              mi_mat[i, j] = val
              mi_mat[j, i] = val
              if val>0:
                mi_vals.append(val)

        if mi_vals:
            thresh = float(torch.quantile(torch.tensor(mi_vals, device=sub.device), self.mi_quantile))
        else:
            thresh = 0.0
        adj = mi_mat > thresh

        visited = torch.zeros(n, dtype=torch.bool, device=sub.device)
        queue = [0]
        visited[0] = True
        while queue:
            u = queue.pop()
            neigh = torch.nonzero(adj[u] & ~visited, as_tuple=False).squeeze(1).tolist()
            for v in neigh:
                visited[v] = True
                queue.append(v)
        return V_s[visited], V_s[~visited]

    def _cluster_instances(
        self,
        V_s: LongTensor,
        T_s: LongTensor,
        data: Tensor
    ) -> Tuple[LongTensor, LongTensor]:
        """
        Binary instance split via k-means on features V_s.
        """
        sub = data.index_select(0, T_s).index_select(1, V_s).float()
        kmeans = KMeans(n_clusters=2,
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
        input_layer: str
    ) -> Any:
        col = data[instance_ids, feat_idx]
        counts = torch.bincount(col, minlength=num_categories + 1).float()
        counts += self.alpha
        probs = counts / counts.sum()
        proto = probs.cpu().numpy().astype(float)
        tp = TensorParameter(num_input_units, probs.numel(),
                             initializer=ConstantTensorInitializer(proto),
                             learnable=False)
        param = Parameter.from_input(tp)
        factory = name_to_input_layer_factory(input_layer, num_categories=probs.numel())
        return factory(Scope([feat_idx]), num_input_units, probs=param)