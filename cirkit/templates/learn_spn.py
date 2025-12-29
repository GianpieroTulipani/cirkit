import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import LongTensor, Tensor

from fast_pytorch_kmeans import KMeans

from cirkit.symbolic.circuit import Circuit
from cirkit.templates.miwae import ConvVAE
from cirkit.symbolic.layers import SumLayer, InputLayer
from cirkit.symbolic.parameters import TensorParameter, Parameter
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.templates.utils import (
    Parameterization,
    name_to_input_layer_factory,
    parameterization_to_factory,
    name_to_parameter_activation,
)
from cirkit.templates.region_graph.algorithms import QuadGraph, QuadTree

class LearnSPN:
    def __init__(
        self,
        alpha: float = 0.5,
        image_shape: Tuple[int, int] = (1, 28, 28),
        seed: Optional[int] = 42,
        noise_scale: float = 1e-1,
        use_miwae: bool = False,
        weight_dir: str = None,
        device: Optional[torch.device] = None,
        data_format: str = None
    ):
        
        assert data_format in ('image', 'tabular'), "data_format should be either 'image' or 'tabular'"
    
        self.alpha = alpha
        self.use_miwae = use_miwae
        self.noise_scale = noise_scale
        self.image_shape = image_shape
        self.data_format = data_format

        self.device = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            self._set_seed(seed)

        if data_format == 'image':
            assert len(image_shape) == 3, "image_shape should be (C, H, W)"

            if use_miwae:
                _, H, W = image_shape
                self.coords = {i: (i // W, i % W) for i in range(H*W)}
                self.miwae = ConvVAE(input_channel=1, latent_dim=50).to(self.device)
                if weight_dir is not None:
                    self.miwae.load_state_dict(torch.load(weight_dir, map_location=self.device))
        
    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def quad_spn(
            self,
            data: LongTensor,
            region_graph: str='quad-graph',
            input_layer: str = 'categorical',
            activation: str = 'softmax',
            sum_product_layer = 'cp',
            num_input_units: int = 1,
            num_sum_units: int = 1,
            num_classes: int = 1
            ) -> Circuit:

        if region_graph == 'quad-graph':
            rg = QuadGraph(self.image_shape)
        elif region_graph == 'quad-tree-2':
            rg = QuadTree(self.image_shape, num_patch_splits=2)
        elif region_graph == 'quad-tree-4':
            rg = QuadTree(self.image_shape, num_patch_splits=4)
        else:
            raise ValueError(f"Unknown region graph called {region_graph}")
        
        num_categories = int(data.max().item() + 1)
        input_factory = name_to_input_layer_factory(input_layer, num_categories=num_categories)
        all_rows = torch.arange(data.size(0), device=self.device, dtype=torch.long)

        sc = rg.build_circuit(
            input_factory=input_factory,
            sum_product=sum_product_layer,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            num_classes=num_classes,
            factorize_multivariate=True
        )        

        queue = deque([(out, all_rows) for out in sc.outputs])
        visited = set()
        layer_count = 1

        while queue:
            layer, rows_idx = queue.popleft()

            if layer in visited:
                continue
            visited.add(layer)
            layer_count += 1

            layer_in = sc.layer_inputs(layer)
            layer_out = sc.layer_outputs(layer)
            
            if isinstance(layer, InputLayer):
                scope = list(layer.scope)
                
                param = self._make_input_param_estimated(
                    feat_idx=int(scope[0]),
                    instance_ids=all_rows,
                    data=data,
                    num_input_units=layer.num_output_units,
                    num_categories=layer.num_categories,
                    activation=activation
                )
                
                layer.probs=param
                    
            elif isinstance(layer, SumLayer):
                feat_ids = torch.tensor(list(sc._scopes[layer]), dtype=torch.long, device=data.device)
                cluster=self._cluster_instances(feat_ids, rows_idx, data, len(layer_in))

                param = self._make_sum_param_estimated(
                    clusters=cluster,
                    num_input_units=layer.num_input_units,
                    num_sum_units=(1 if layer_out is None else layer.num_output_units),
                    activation=activation
                )
                
                layer.weight=param

                for child, cluster_ids in zip(layer_in, cluster):
                    if child not in visited:
                        queue.append((child, cluster_ids))
            else:
                for child in layer_in:
                    if child not in visited:
                        queue.append((child, rows_idx))

        print(f"Total layers processed: {layer_count}")
        return sc
        
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

    def _make_input_param_estimated(
        self,
        feat_idx: int,
        instance_ids: LongTensor,
        data: LongTensor,
        num_input_units: int,
        num_categories: int,
        activation: str
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
            if self.noise_scale and self.noise_scale > 0.0:
                logits = logits + np.random.normal(loc=0.0, scale=self.noise_scale, size=logits.shape)

        tp = TensorParameter(
            num_input_units,
            num_categories,
            initializer=ConstantTensorInitializer(logits),
            learnable=True
            )
        unary_op_factory = name_to_parameter_activation(activation)

        return Parameter.from_unary(unary_op_factory((num_input_units, num_categories)), tp)

    def _make_sum_param_estimated(
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
            if self.noise_scale and self.noise_scale > 0.0:
                logits = logits + np.random.normal(loc=0.0, scale=self.noise_scale, size=logits.shape)

        tp = TensorParameter(
            num_sum_units,
            arity * num_input_units,
            initializer=ConstantTensorInitializer(logits),
            learnable=True
        )

        unary_op_factory = name_to_parameter_activation(activation)

        return Parameter.from_unary(unary_op_factory((num_sum_units, arity * num_input_units)), tp)