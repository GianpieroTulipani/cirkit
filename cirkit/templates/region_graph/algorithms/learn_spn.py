from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import math
import numpy as np
import torch
from torch import Tensor, LongTensor

from cirkit.templates.region_graph.algorithms.chow_liu import _categorical_mutual_info
from cirkit.utils.algorithms import RootedDiAcyclicGraph, topological_ordering


# === Utility Functions ===

def g_test(
    feature_id_1: int,
    feature_id_2: int,
    instance_ids: LongTensor,
    data: LongTensor,
    cardinalities: LongTensor,
    g_factor: float,
) -> bool:
    """
    Perform a G‐test for independence between two categorical features over a subset of instances.
    Returns True if the G‐statistic indicates independence; False otherwise.
    """
    if feature_id_1 > feature_id_2:
        feature_id_1, feature_id_2 = feature_id_2, feature_id_1

    n_instances = instance_ids.numel()
    fsize1 = int(cardinalities[feature_id_1].item())
    fsize2 = int(cardinalities[feature_id_2].item())

    subset = data.index_select(0, instance_ids)
    col1 = subset[:, feature_id_1]
    col2 = subset[:, feature_id_2]
    flat_idx = col1 * fsize2 + col2

    flat_counts = torch.bincount(flat_idx, minlength=fsize1 * fsize2).float()
    co_occ = flat_counts.view(fsize1, fsize2)

    marg1 = co_occ.sum(dim=1)
    marg2 = co_occ.sum(dim=0)
    nz1 = torch.count_nonzero(marg1).item()
    nz2 = torch.count_nonzero(marg2).item()
    dof = (nz1 - 1) * (nz2 - 1)

    g_val = 0.0
    nonzero_rows = torch.nonzero(marg1, as_tuple=False).squeeze(1)
    for i in nonzero_rows:
        total_i = marg1[i].item()
        nonzero_cols = torch.nonzero(co_occ[i] > 0, as_tuple=False).squeeze(1)
        for j in nonzero_cols:
            count_ij = co_occ[i, j].item()
            expected = total_i * marg2[j].item() / float(n_instances)
            g_val += count_ij * math.log(count_ij / expected)
    g_val *= 2.0

    threshold = 2.0 * dof * g_factor + 1e-12
    return g_val < threshold


def hard_em_categorical(
    X: LongTensor,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    smoothing: float = 0.1,
    cardinalities: LongTensor = None,
) -> LongTensor:
    """
    Perform hard‐EM clustering on categorical data X into k clusters.
    X: [N, V] long tensor of categorical variables (0..M_j-1)
    Returns: [N] tensor of cluster labels in {0, …, k-1}.
    """
    N, V = X.shape
    device = X.device

    # Initialize random assignments, ensure nonempty clusters
    assignments = torch.randint(0, k, (N,), device=device)
    unique_clusters = torch.unique(assignments)
    if unique_clusters.numel() < k:
        for c in range(k):
            if c not in unique_clusters:
                idx = torch.randint(0, N, (1,), device=device).item()
                assignments[idx] = c
        unique_clusters = torch.unique(assignments)

    # Compute cardinalities if not provided
    if cardinalities is None:
        cardinalities = torch.tensor(
            [int(X[:, j].max().item()) + 1 for j in range(V)], device=device
        )

    for _ in range(max_iters):
        prev = assignments.clone()
        counts = torch.bincount(assignments, minlength=k).float()
        P_c = counts / counts.sum()

        P_x_given_c: List[List[Tensor]] = []
        for c in range(k):
            mask_c = assignments == c
            if mask_c.sum().item() == 0:
                # Empty cluster → uniform counts
                cluster_probs = [torch.ones(cardinalities[j], device=device) for j in range(V)]
            else:
                cluster_probs = []
                X_c = X[mask_c, :]
                for j in range(V):
                    vals = X_c[:, j]
                    cnt_j = (
                        torch.bincount(vals, minlength=cardinalities[j]).float()
                        if vals.numel() > 0
                        else torch.zeros(cardinalities[j], device=device)
                    )
                    cnt_j += smoothing
                    cluster_probs.append(cnt_j)
            P_x_given_c.append(cluster_probs)

        max_cat = int(cardinalities.max().item())
        logP = torch.full((k, V, max_cat), float("-inf"), device=device)
        for c in range(k):
            for j in range(V):
                probs = P_x_given_c[c][j] / P_x_given_c[c][j].sum()
                logP[c, j, : cardinalities[j]] = torch.log(probs)

        log_P_c = torch.log(P_c)
        log_liks = torch.zeros((N, k), device=device)
        for c in range(k):
            # Gather log‐probs for each instance under cluster c
            logP_c_expand = logP[c].unsqueeze(0).expand(N, -1, -1)  # [N, V, max_cat]
            idx = X.unsqueeze(2)  # [N, V, 1]
            gathered = torch.gather(logP_c_expand, 2, idx)  # [N, V, 1]
            log_sum = gathered.sum(dim=1).squeeze(1)  # [N]
            log_liks[:, c] = log_sum + log_P_c[c]

        assignments = torch.argmax(log_liks, dim=1)
        changed = (assignments != prev).sum().item()
        if (changed / N) < tol:
            break

    return assignments


# === SPN Node Classes ===

class Node(ABC):
    """
    Abstract base class for SPN nodes.
    """

    def __init__(self, scope: frozenset):
        self.scope = scope
        self.children: List["Node"] = []

    def add_child(self, child: "Node") -> None:
        self.children.append(child)

    @abstractmethod
    def eval_batch(self, x: Tensor) -> Tensor:
        """
        Evaluate this node on a batch of inputs.
        x: [batch_size, num_features], long tensor of categorical values.
        Returns: [batch_size] tensor of log‐probabilities.
        """
        pass


class SumNode(Node):
    def __init__(self, scope: frozenset, weights: List[float]):
        super().__init__(scope)
        self.weights = torch.tensor(weights, dtype=torch.float)

    def eval_batch(self, x: Tensor) -> Tensor:
        # Stack child log‐probs: [B, C]
        child_logits = torch.stack([c.eval_batch(x) for c in self.children], dim=1)
        max_logits, _ = child_logits.max(dim=1, keepdim=True)  # [B, 1]
        centered = child_logits - max_logits  # [B, C]
        w = self.weights.to(child_logits.device)  # [C]
        log_w = torch.log(w)  # [C]
        weighted = centered + log_w.unsqueeze(0)  # [B, C]
        return max_logits.squeeze(1) + torch.logsumexp(weighted, dim=1)


class ProdNode(Node):
    def __init__(self, scope: frozenset):
        super().__init__(scope)

    def eval_batch(self, x: Tensor) -> Tensor:
        child_logits = torch.stack([c.eval_batch(x) for c in self.children], dim=1)
        return child_logits.sum(dim=1)


class LeafNode(Node):
    def __init__(self, scope: frozenset, probs: Tensor):
        """
        probs: 1D tensor of length = (#categories) for that feature.
        scope must be a singleton {feature_index}.
        """
        super().__init__(scope)
        self.probs = probs

    def eval_batch(self, x: Tensor) -> Tensor:
        feature_idx = next(iter(self.scope))
        idx = x[:, feature_idx].long()
        return torch.log(self.probs[idx])


class FactorNode(Node):
    def __init__(self, scope: frozenset, marginals: Dict[int, Tensor]):
        """
        marginals[v] = 1D tensor of length = (#categories for feature v), P(X_v).
        scope may contain multiple features, assumed independent.
        """
        super().__init__(scope)
        self.marginals = marginals

    def eval_batch(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        device = x.device
        log_probs = torch.zeros(batch_size, device=device)
        for v in self.scope:
            idx = x[:, v].long()
            log_probs += torch.log(self.marginals[v][idx])
        return log_probs


# === SPN Circuit Wrapper ===

class SPNCircuit(RootedDiAcyclicGraph):
    """
    Wraps SPN nodes into a DAG with efficient batch evaluation via topological order.
    """

    def __init__(
        self,
        nodes: Sequence[Node],
        in_nodes: Dict[Node, List[Node]],
        root: Node,
    ):
        super().__init__(nodes, in_nodes, outputs=[root])
        self._root = root
        self._topo_order = list(self.topological_ordering())

    def eval(self, x: Tensor) -> Tensor:
        """
        Evaluate the entire circuit on batch x: [batch_size, num_features].
        Returns: [batch_size] tensor of log‐likelihoods.
        """
        cache: Dict[Node, Tensor] = {}
        for node in self._topo_order:
            if isinstance(node, (LeafNode, FactorNode)):
                cache[node] = node.eval_batch(x)
            elif isinstance(node, SumNode):
                child_vals = torch.stack([cache[c] for c in node.children], dim=1)  # [B, C]
                max_vals, _ = child_vals.max(dim=1, keepdim=True)  # [B, 1]
                centered = child_vals - max_vals  # [B, C]
                w = node.weights.to(child_vals.device)  # [C]
                weighted = centered + torch.log(w).unsqueeze(0)  # [B, C]
                cache[node] = max_vals.squeeze(1) + torch.logsumexp(weighted, dim=1)
            else:  # ProdNode
                stacked = torch.stack([cache[c] for c in node.children], dim=1)  # [B, C]
                cache[node] = stacked.sum(dim=1)
        return cache[self._root]


# === Slice and Learning ===

@dataclass
class Slice:
    """
    Represents a subset ("slice") of the dataset, defined by active features and instances.
    """
    features: LongTensor
    instances: LongTensor
    parent: Node = None

    id_counter: int = 0

    def __post_init__(self):
        self.id = Slice.id_counter
        Slice.id_counter += 1


class LearnSPN:
    """
    Learns a Sum‐Product Network (SPN) structure from categorical data.
    """

    def __init__(self, alpha: float = 1.0, min_instances: int = 10, g_factor: float = 1.0):
        """
        alpha: Laplace smoothing constant.
        min_instances: if |instances| ≤ min_instances, create a FactorNode over all features.
        g_factor: multiplier for the G‐test threshold in feature splitting.
        """
        self.alpha = alpha
        self.min_instances = min_instances
        self.g_factor = g_factor

        self.queue: deque[Slice] = deque()
        self.nodes: List[Node] = []
        self.in_nodes: Dict[Node, List[Node]] = defaultdict(list)
        self.data: LongTensor = None
        self.circuit: SPNCircuit = None
        self.feature_cardinalities: LongTensor = None

    def learn(self, data: LongTensor) -> None:
        """
        Learn the SPN structure from data: [N, D], dtype=torch.long.
        """
        assert data.dtype == torch.long and data.ndim == 2

        N, D = data.shape
        # Compute cardinalities per feature
        card_list = [int(data[:, j].max().item()) + 1 for j in range(D)]
        self.feature_cardinalities = torch.tensor(card_list, dtype=torch.long, device=data.device)

        self.data = data
        self.circuit = self._learn_structure()

    def log_likelihood(self, X: Tensor) -> Tensor:
        """
        Evaluate log‐likelihood of X under the learned SPN.
        """
        assert self.circuit is not None, "Call learn() before evaluating."
        return self.circuit.eval(X)

    def _learn_structure(self) -> SPNCircuit:
        """
        Build the SPN by processing slices in a queue.
        """
        N, D = self.data.shape
        all_feats = torch.arange(D, device=self.data.device)
        all_insts = torch.arange(N, device=self.data.device)

        root_slice = Slice(features=all_feats, instances=all_insts, parent=None)
        self.queue.append(root_slice)
        root_node: Node = None

        while self.queue:
            sl = self.queue.popleft()
            V_s, T_s = sl.features, sl.instances

            if V_s.numel() == 1:
                node = self._make_leaf(V_s, T_s)
            elif T_s.numel() <= self.min_instances:
                node = self._make_factor(V_s, T_s)
            else:
                V_dep, V_indep = self._split_features(V_s, T_s)
                if V_indep.numel() > 0 and sl.id != 0:
                    node = ProdNode(scope=frozenset(V_s.tolist()))
                    for part in (V_dep, V_indep):
                        child = Slice(features=part, instances=T_s, parent=node)
                        self.queue.append(child)
                else:
                    T1, T2 = self._cluster_instances(V_s, T_s)
                    w1 = T1.numel() / float(T_s.numel())
                    w2 = T2.numel() / float(T_s.numel())
                    node = SumNode(scope=frozenset(V_s.tolist()), weights=[w1, w2])
                    for part_insts in (T1, T2):
                        child = Slice(features=V_s, instances=part_insts, parent=node)
                        self.queue.append(child)

            self.nodes.append(node)
            if sl.parent is not None:
                self.in_nodes[sl.parent].append(node)
                sl.parent.add_child(node)
            else:
                root_node = node

        return SPNCircuit(nodes=self.nodes, in_nodes=self.in_nodes, root=root_node)

    def _split_features(self, V_s: LongTensor, T_s: LongTensor) -> Tuple[LongTensor, LongTensor]:
        """
        Greedily split features in V_s into dependent and independent subsets via G‐tests.
        """
        n_feats = V_s.numel()
        device = self.data.device

        mask_remaining = torch.ones(n_feats, dtype=torch.bool, device=device)
        mask_dependent = torch.zeros(n_feats, dtype=torch.bool, device=device)

        # Start with a random feature as dependent
        seed = torch.randint(0, n_feats, (1,), device=device).item()
        mask_remaining[seed] = False
        mask_dependent[seed] = True

        queue = deque([seed])
        while queue:
            cur = queue.popleft()
            feat1 = int(V_s[cur].item())

            to_remove = torch.zeros(n_feats, dtype=torch.bool, device=device)
            for other in torch.nonzero(mask_remaining, as_tuple=False).squeeze(1):
                feat2 = int(V_s[other].item())
                independent = g_test(
                    feat1,
                    feat2,
                    T_s,
                    self.data,
                    self.feature_cardinalities,
                    self.g_factor,
                )
                if not independent:
                    to_remove[other] = True
                    mask_dependent[other] = True
                    queue.append(int(other))

            mask_remaining[to_remove] = False

        idx_dep = torch.nonzero(mask_dependent, as_tuple=False).squeeze(1)
        idx_indep = torch.nonzero(~mask_dependent, as_tuple=False).squeeze(1)

        V_dep = V_s[idx_dep]
        V_indep = V_s[idx_indep]
        return V_dep, V_indep

    def _cluster_instances(self, V_s: LongTensor, T_s: LongTensor) -> Tuple[LongTensor, LongTensor]:
        """
        Cluster instances T_s into two groups via hard‐EM on data[:, V_s].
        """
        submatrix = self.data.index_select(0, T_s).index_select(1, V_s)
        labels = hard_em_categorical(
            submatrix, k=2, smoothing=self.alpha, cardinalities=self.feature_cardinalities
        )
        return T_s[labels == 0], T_s[labels == 1]

    def _make_leaf(self, V_s: LongTensor, T_s: LongTensor) -> LeafNode:
        """
        Create a LeafNode for a single feature V_s and instances T_s.
        """
        assert V_s.numel() == 1
        feat = int(V_s.item())
        col_vals = self.data.index_select(0, T_s)[:, feat]
        max_label = int(self.data[:, feat].max().item())
        counts = torch.bincount(col_vals, minlength=max_label + 1).float()
        counts += self.alpha
        probs = counts / counts.sum()
        return LeafNode(scope=frozenset({feat}), probs=probs)

    def _make_factor(self, V_s: LongTensor, T_s: LongTensor) -> FactorNode:
        """
        Create a FactorNode over features V_s and instances T_s (treat as independent).
        """
        marginals: Dict[int, Tensor] = {}
        for feat in V_s.tolist():
            col_vals = self.data.index_select(0, T_s)[:, feat]
            max_label = int(self.data[:, feat].max().item())
            counts = torch.bincount(col_vals, minlength=max_label + 1).float()
            counts += self.alpha
            marginals[feat] = counts / counts.sum()

        return FactorNode(scope=frozenset(V_s.tolist()), marginals=marginals)