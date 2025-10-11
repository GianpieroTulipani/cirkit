import random
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Any, List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

from fast_pytorch_kmeans import KMeans

from cirkit.symbolic.circuit import Circuit
from cirkit.templates.miwae import ConvVAE
from cirkit.symbolic.layers import HadamardLayer, SumLayer, Layer
from cirkit.symbolic.parameters import TensorParameter, Parameter
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.templates.utils import (
    Parameterization,
    name_to_input_layer_factory,
    parameterization_to_factory,
    name_to_parameter_activation,
)
from cirkit.templates.region_graph.algorithms import QuadGraph
from cirkit.templates.region_graph import RegionNode, PartitionNode, RegionGraphNode
from cirkit.utils.scope import Scope

@dataclass
class Task:
    feat_ids: LongTensor
    instance_ids: LongTensor
    parent: Optional[Layer]

class LearnSPN:
    def __init__(
        self,
        alpha: float = 0.5,
        min_instances: int = 500,
        mi_quantile: float = 0.5,
        local_radius: int = 4,
        image_shape: Tuple[int, int] = (1, 28, 28),
        seed: Optional[int] = 42,
        jitter_scale: float = 1e-2,
        use_miwae: bool = False,
        weight_dir: str = None,
        device: Optional[torch.device] = None
    ):
        if seed is not None:
            self._set_seed(seed)

        self.alpha = alpha
        self.min_instances = min_instances
        self.mi_quantile = mi_quantile
        self.local_radius = local_radius
        assert len(image_shape) == 3, "image_shape should be (C, H, W)"
        self.image_shape = image_shape

        self.device = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.jitter_scale = jitter_scale
        self.use_miwae = use_miwae

        if use_miwae:
            self.miwae = ConvVAE(input_channel=1, latent_dim=50).to(self.device)
            if weight_dir is not None:
                self.miwae.load_state_dict(torch.load(weight_dir, map_location=self.device))
        
        self._build_neighbor_map()

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_neighbor_map(self):
        _, H, W = self.image_shape
        n = H * W
        self.coords = {i: (i // W, i % W) for i in range(n)}
        self.neighbor_map: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            y_i, x_i = self.coords[i]
            nbrs = []
            for j in range(n):
                y_j, x_j = self.coords[j]
                if abs(y_i - y_j) + abs(x_i - x_j) <= self.local_radius:
                    nbrs.append(j)
            self.neighbor_map[i] = nbrs

    def learn_spn(
        self,
        data: LongTensor,
        input_layer: str = "categorical",
        activation: str = "softmax",
        initialization: str = "estimated",
        num_input_units: int = 1,
        num_sum_units: int = 1,
    ) -> Circuit:

        use_estimated = initialization == "estimated"

        layers: List[Layer] = []
        in_layers: Dict[Layer, List[Layer]] = {}
        root: List[Layer] = []

        N, D = data.shape
        num_categories = int(data.max().item() + 1)
        input_factory = name_to_input_layer_factory(input_layer, num_categories=num_categories)

        if not use_estimated:
            sum_weight_param = Parameterization(activation=activation, initialization=initialization)
            sum_weight_factory = parameterization_to_factory(sum_weight_param)

        def _make_leaf_and_attach(feat_ids: int, instance_ids: LongTensor, parent):
            if use_estimated:
                layer = self._make_leaf_layer_estimated(
                    feat_ids,
                    instance_ids,
                    data,
                    num_input_units,
                    num_categories,
                    activation,
                    input_factory,
                )
            else:
                layer = input_factory(Scope([feat_ids]), num_input_units)
            layers.append(layer)
            in_layers.setdefault(parent, []).append(layer)


        def _handle_small_instances(feat_ids: LongTensor, instance_ids: LongTensor, parent):
            if use_estimated:
                feats = [
                    self._make_leaf_layer_estimated(int(f), instance_ids, data, num_input_units, num_categories, activation, input_factory)
                    for f in feat_ids.tolist()
                ]
            else:
                feats = [input_factory(Scope([int(v)]), num_input_units) for v in feat_ids.tolist()]
                
            layer = HadamardLayer(num_input_units, arity=len(feats))
            layers.extend(feats)
            layers.append(layer)
            in_layers[layer] = feats
            in_layers.setdefault(parent, []).append(layer)

        def _handle_cluster_split(feat_ids: LongTensor, instance_ids: LongTensor, parent):
            clusters = self._cluster_instances(feat_ids, instance_ids, data)
            if clusters[0].numel() == 0 or clusters[1].numel() == 0:
                _handle_small_instances(feat_ids, instance_ids, parent)
                return

            if use_estimated:
                if parent is not None:
                    layer = self._make_sum_layer_estimated(clusters, num_input_units, num_sum_units, activation)
                else:
                    layer = self._make_sum_layer_estimated(clusters, num_input_units, 1, activation)
                    root.append(layer)
            else:
                if parent is not None:
                    layer = SumLayer(num_input_units=parent.num_input_units, num_output_units=num_sum_units, arity=2, weight_factory=sum_weight_factory)
                else:
                    layer = SumLayer(num_input_units=num_sum_units, num_output_units=1, arity=2, weight_factory=sum_weight_factory)
                    root.append(layer)

            layers.append(layer)
            if parent is not None:
                in_layers.setdefault(parent, []).append(layer)

            stack.append(Task(instance_ids, clusters[1], layer))
            stack.append(Task(instance_ids, clusters[0], layer))

        stack = [Task(torch.arange(D, device=self.device), torch.arange(N, device=self.device), None)]

        while stack:
            task = stack.pop()
            feat_ids, instance_ids, parent = task.feat_ids, task.instance_ids, task.parent

            if feat_ids.numel() == 1:
                _make_leaf_and_attach(feat_ids, instance_ids, parent)
                continue

            if instance_ids.numel() <= self.min_instances:
                _handle_small_instances(feat_ids, instance_ids, parent)
                continue

            if parent is not None:
                V_dep, V_indep = self._split_features_local(feat_ids, instance_ids, data, num_categories)
                if V_indep.numel() > 0:
                    layer = HadamardLayer(num_input_units, arity=2)
                    layers.append(layer)
                    in_layers.setdefault(parent, []).append(layer)

                    stack.append(Task(V_indep, instance_ids, layer))
                    stack.append(Task(V_dep, instance_ids, layer))
                    continue

            _handle_cluster_split(feat_ids, instance_ids, parent)

        return Circuit(layers, in_layers, root)

    def quad_spn(
        self,
        data: LongTensor,
        input_layer: str = 'categorical',
        activation: str = 'softmax',
        num_input_units: int = 1,
        num_sum_units: int = 1
    ) -> Circuit:

        layers: List[Layer] = []
        in_layers: Dict[Layer, List[Layer]] = {}
        node_to_layer: Dict[RegionGraphNode, Layer] = {}
    
        qg = QuadGraph(self.image_shape)
        num_categories = int(data.max().item() + 1)
        input_factory = name_to_input_layer_factory(input_layer, num_categories=num_categories)
    
        all_rows = torch.arange(data.size(0), device=self.device, dtype=torch.long)

        queue = deque([(out, None, all_rows) for out in qg.outputs])

        while queue:
            node, parent_layer, rows_idx = queue.popleft()
            if isinstance(node, RegionNode) and not qg.region_inputs(node):
                scope_vars = list(node.scope)
    
                if len(scope_vars) == 1:
                    feat = int(scope_vars[0])

                    layer = self._make_leaf_layer_estimated(
                        feat_idx=feat,
                        instance_ids=rows_idx,
                        data=data,
                        num_input_units=num_input_units,
                        num_categories=num_categories,
                        activation=activation,
                        input_factory=input_factory
                    )
                    layers.append(layer)
                    node_to_layer[node] = layer
    
                else:
                    feat_layers: List[Layer] = []
                    for sc in scope_vars:
                        in_layer = self._make_leaf_layer_estimated(
                            feat_idx=int(sc),
                            instance_ids=rows_idx,
                            data=data,
                            num_input_units=num_input_units,
                            num_categories=num_categories,
                            activation=activation,
                            input_factory=input_factory
                        )
                        layers.append(in_layer)
                        feat_layers.append(in_layer)
    
                    layer = HadamardLayer(num_input_units, arity=len(feat_layers))
                    layers.append(layer)
                    in_layers[layer] = feat_layers
                    node_to_layer[node] = layer
    
            elif isinstance(node, RegionNode):
                children = qg.region_inputs(node)

                arity = len(children)
    
                feat_ids = torch.tensor(list(node.scope), dtype=torch.long, device=data.device)
                
                clusters = self._cluster_instances(data, feat_ids, rows_idx, arity)
                if parent_layer is None:
                    layer = self._make_sum_layer_estimated(
                        clusters=clusters,
                        num_input_units=num_input_units,
                        num_sum_units=1,
                        activation=activation,
                    )
                else:
                    layer = self._make_sum_layer_estimated(
                        clusters=clusters,
                        num_input_units=num_input_units,
                        num_sum_units=num_sum_units,
                        activation=activation,
                    )
                layers.append(layer)
                node_to_layer[node] = layer
    
                for child_node, cluster_ids in zip(children, clusters):
                    queue.append((child_node, layer, cluster_ids))
    
            elif isinstance(node, PartitionNode):
                children = qg.partition_inputs(node)
                layer = HadamardLayer(num_sum_units, arity=len(children))
                layers.append(layer)
                node_to_layer[node] = layer
    
                for child_node in children:
                    queue.append((child_node, layer, rows_idx))
    
            if parent_layer is not None:
                parent_list = in_layers.setdefault(parent_layer, [])
                parent_list.append(node_to_layer[node])
    
        outputs = [node_to_layer[rgn] for rgn in qg.outputs]
        return Circuit(layers, in_layers, outputs)

    def _split_features_local(
        self,
        feat_ids: LongTensor,
        instance_ids: LongTensor,
        data: Tensor,
        num_categories: int,
        chunk_size: int = 1000,
    ) -> Tuple[LongTensor, LongTensor]:
        sub = data.index_select(0, instance_ids).index_select(1, feat_ids)
        n = feat_ids.numel()

        idx_map = {int(v): i for i, v in enumerate(feat_ids.tolist())}

        pairs = []
        for i, feat_i in enumerate(feat_ids.tolist()):
            for feat_j in self.neighbor_map[int(feat_i)]:
                if feat_j in idx_map:
                    j = idx_map[feat_j]
                    if j > i:
                        pairs.append((i, j))

        if not pairs:
            return feat_ids, torch.empty(0, dtype=torch.long, device=data.device)

        idx_i = torch.tensor([p[0] for p in pairs], device=sub.device)
        idx_j = torch.tensor([p[1] for p in pairs], device=sub.device)

        mi_mat = torch.zeros((n, n), device=sub.device)

        for start in range(0, len(idx_i), chunk_size):
            end = start + chunk_size
            i_chunk = idx_i[start:end]
            j_chunk = idx_j[start:end]

            mi_vals = _pairwise_mutual_info(sub[:, i_chunk], sub[:, j_chunk], self.alpha, num_categories)

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

        V_dep = feat_ids[visited]
        V_indep = feat_ids[~visited]
        return V_dep, V_indep

    def _cluster_instances(
        self, 
        feat_ids: LongTensor, 
        instance_ids: LongTensor, 
        data: Tensor, 
        n_clusters: int = 2, 
        mode: str = "euclidean"
    ) -> Tuple[LongTensor, LongTensor]:
        
        if instance_ids.numel() == 0:
            return [instance_ids.new_empty((0,), dtype=torch.long) for _ in range(n_clusters)]
        
        kmeans = KMeans(n_clusters=n_clusters,
                        mode=mode,
                        verbose=0
                        )
        
        if self.use_miwae:
            C, H, W = self.image_shape
            N = instance_ids.numel()
            sub = data.index_select(0, instance_ids)
            imgs = torch.zeros((N, C, H, W), device=self.device, dtype=torch.float32)

            for feat_idx in feat_ids.tolist():
                y, x = self.coords[feat_idx]
                imgs[:, 0, y, x] = sub[:, feat_idx].float() / 255.0
            
            with torch.no_grad():
                mu, log_var, _, _ = self.miwae.encoder(imgs)
                embeddings = torch.cat([mu, log_var], dim=1).detach()

            labels = kmeans.fit_predict(embeddings)
        else:
            sub = data.index_select(0, instance_ids).index_select(1, feat_ids).float()
            labels = kmeans.fit_predict(sub)
        
        clusters: List[LongTensor] = []
        for c in range(n_clusters):
            mask = (labels == c)
            clusters.append(instance_ids[mask])

        return clusters

    def _make_leaf_layer_estimated(
        self,
        feat_idx: int,
        instance_ids: LongTensor,
        data: LongTensor,
        num_input_units: int,
        num_categories: int,
        activation: str,
        input_factory: Any,
    ):
        col = data[instance_ids, feat_idx]
        counts = torch.bincount(col, minlength=num_categories).float()
        counts += self.alpha
        probs = counts / counts.sum()

        probs_np = probs.cpu().numpy().astype(float)

        if num_input_units == 1:
            logits = np.log(probs_np).reshape(1, num_categories)
        else:
            base = np.log(probs_np)
            logits = np.tile(base.reshape(1, num_categories), (num_input_units, 1))
            if self.jitter_scale and self.jitter_scale > 0.0:
                logits = logits + np.random.normal(loc=0.0, scale=self.jitter_scale, size=logits.shape)

        tp = TensorParameter(num_input_units, num_categories, initializer=ConstantTensorInitializer(logits), learnable=True)
        unary_op_factory = name_to_parameter_activation(activation)
        param = Parameter.from_unary(unary_op_factory((num_input_units, num_categories)), tp)

        return input_factory(Scope([feat_idx]), num_input_units, probs=param)

    def _make_sum_layer_estimated(
        self,
        clusters: List[LongTensor],
        num_input_units: int,
        num_sum_units: int,
        activation: str
    ):
        arity = len(clusters)
        cluster_sizes = [int(c.numel()) for c in clusters]
        smoothed = [sz + self.alpha for sz in cluster_sizes]
        total = float(sum(smoothed))
        weights = [sz / total for sz in smoothed]
        mix_weights = np.array(weights, dtype=float)
        
        if num_input_units == 1:
            logits = np.log(mix_weights).reshape(1, arity)
        else:
            rep_weights = np.tile(mix_weights.reshape(1, arity), (num_sum_units, 1))
            rep_weights_expandend = np.tile(rep_weights.reshape(num_sum_units, arity, 1), (1, 1, num_input_units))
            rep_weights_flat = rep_weights_expandend.reshape(num_sum_units, arity * num_input_units)
            
            logits = np.log(rep_weights_flat)
            if self.jitter_scale and self.jitter_scale > 0.0:
                logits = logits + np.random.normal(loc=0.0, scale=self.jitter_scale, size=logits.shape)

        tp = TensorParameter(
            num_sum_units,
            arity * num_input_units,
            initializer=ConstantTensorInitializer(logits),
            learnable=True
        )

        unary_op_factory = name_to_parameter_activation(activation)
        
        param = Parameter.from_unary(
            unary_op_factory((num_sum_units, arity * num_input_units)),
            tp
        )

        return SumLayer(num_input_units=num_input_units, num_output_units=num_sum_units, arity=arity, weight=param)


def _pairwise_mutual_info(x1: LongTensor, x2: LongTensor, alpha: float, num_categories: int) -> Tensor:
    N, K = x1.shape
    x1_flat = x1.T.contiguous()
    x2_flat = x2.T.contiguous()
    joint = x1_flat * num_categories + x2_flat
    counts = torch.zeros(K, num_categories * num_categories, device=x1.device)
    counts.scatter_add_(1, joint, torch.ones_like(joint, dtype=torch.float))
    counts = counts.view(K, num_categories, num_categories)

    x1_counts = counts.sum(dim=2)
    x2_counts = counts.sum(dim=1)

    joint_probs = (counts + alpha) / (N + num_categories ** 2 * alpha)
    x1_probs = (x1_counts + num_categories * alpha) / (N + num_categories ** 2 * alpha)
    x2_probs = (x2_counts + num_categories * alpha) / (N + num_categories ** 2 * alpha)

    x1_probs = x1_probs.unsqueeze(2)
    x2_probs = x2_probs.unsqueeze(1)
    prod = x1_probs * x2_probs

    mi = joint_probs * (joint_probs.log() - prod.log())
    return mi.sum(dim=(1, 2))