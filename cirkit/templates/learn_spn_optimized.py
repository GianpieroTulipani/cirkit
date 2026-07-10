import functools
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import LongTensor, Tensor

from fast_pytorch_kmeans import KMeans

from cirkit.symbolic.circuit import Circuit
from cirkit.templates.miwae import ConvVAE
from cirkit.symbolic.layers import SumLayer, InputLayer
from cirkit.symbolic.parameters import (
    TensorParameter,
    Parameter,
    ParameterFactory,
    mixing_weight_factory
)
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
        noise_scale: float = 0.2,            
        use_miwae: bool = False,
        weight_dir: str = None,
        device: Optional[torch.device] = None,
        data_format: str = None,
        adaptive_alpha: bool = True
    ):

        assert data_format in ('image', 'tabular'), "data_format should be either 'image' or 'tabular'"
        assert noise_scale >= 0, "noise_scale should be non-negative"
        assert alpha >= 0, "alpha should be non-negative"

        self.alpha = alpha
        self.use_miwae = use_miwae
        self.noise_scale = noise_scale
        self.image_shape = image_shape
        self.data_format = data_format

        self.adaptive_alpha = adaptive_alpha
        self._data: Optional[LongTensor] = None
        self._all_rows: Optional[LongTensor] = None

        self.device = device if device is not None else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            self._set_seed(seed)

        if data_format == 'image':
            assert len(image_shape) == 3, "image_shape should be (C, H, W)"

            if use_miwae:
                _, H, W = image_shape
                self.coords = {i: (i // W, i % W) for i in range(H * W)}
                self.miwae = ConvVAE(input_channel=1, latent_dim=50).to(self.device)
                if weight_dir is not None:
                    self.miwae.load_state_dict(torch.load(weight_dir, map_location=self.device))

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _to_preactivation(self, probs: np.ndarray, activation: str) -> np.ndarray:
        if activation == 'clamp' or activation == 'none':
            return probs

        if activation == 'softplus':
            p = np.clip(probs, 1e-12, None)
            return np.log(np.expm1(p))

        if activation == 'sigmoid':
            p = np.clip(probs, 1e-12, 1 - 1e-12)
            return np.log(p / (1 - p))

        p = np.clip(probs, 1e-12, None)
        return np.log(p)

    def _apply_symmetry_breaking(self, theta: np.ndarray, activation: str) -> np.ndarray:
        s = self.noise_scale
        if activation == 'clamp' or activation == 'none':
            # spazio lineare: rumore moltiplicativo positivo
            noisy = theta * np.exp(np.random.normal(loc=0.0, scale=s, size=theta.shape))
            return np.clip(noisy, float(np.sqrt(np.finfo(np.float32).tiny)), None)
        # spazio log: rumore additivo
        return theta + np.random.normal(loc=0.0, scale=s, size=theta.shape)

    def _alpha_per_bin(self, num_bins: int) -> float:
        if self.adaptive_alpha:
            return self.alpha / max(num_bins, 1)
        return self.alpha

    def learn_spn(
        self,
        data: LongTensor,
        region_graph: str = 'quad-graph',
        input_layer: str = 'categorical',
        activation: str = 'softmax',
        weights_init: str = 'normal',
        sum_product_layer='cp',
        sum_weight_param: Optional[Parameterization] = None,
        num_input_units: int = 1,
        num_sum_units: int = 1,
        num_classes: int = 1,
        use_estimated_weights: bool = True,
        use_mixing_weights: bool = True,
    ) -> Circuit:

        assert weights_init in ('normal', 'uniform', 'dirichlet', 'None'), (
            "weights_init should be 'normal', 'uniform', 'dirichlet' or 'None'"
        )

        if region_graph == 'quad-graph':
            rg = QuadGraph(self.image_shape)
        elif region_graph == 'quad-tree-2':
            rg = QuadTree(self.image_shape, num_patch_splits=2)
        elif region_graph == 'quad-tree-4':
            rg = QuadTree(self.image_shape, num_patch_splits=4)
        else:
            raise ValueError(f"Unknown region graph called {region_graph}")

        nary_sum_weight_factory: ParameterFactory
        num_categories = int(data.max().item() + 1)
        input_factory = name_to_input_layer_factory(input_layer, num_categories=num_categories)

        if activation == 'clamp':
            sum_weight_param = Parameterization(
                activation='none',
                initialization='uniform',
            )
        else:
            sum_weight_param = Parameterization(
                activation=activation,
                initialization=weights_init,
            )
        sum_weight_factory = parameterization_to_factory(sum_weight_param)

        if use_mixing_weights:
            nary_sum_weight_factory = functools.partial(
                mixing_weight_factory,
                param_factory=sum_weight_factory,
            )
        else:
            nary_sum_weight_factory = sum_weight_factory

        sc = rg.build_circuit(
            input_factory=input_factory,
            sum_product=sum_product_layer,
            sum_weight_factory=sum_weight_factory,
            nary_sum_weight_factory=nary_sum_weight_factory,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            num_classes=num_classes,
            factorize_multivariate=True,
        )

        if use_estimated_weights:
            sc = self._estimate_parameters(sc, data, activation=activation)
        
        return sc

    def _estimate_parameters(
        self,
        sc: Circuit,
        data: LongTensor,
        activation: str,
    ) -> Circuit:

        visited = set()
        self._all_rows = torch.arange(data.size(0), device=self.device, dtype=torch.long)
        self._data = data
        self._subpop_cache = {}
        queue = deque([(out, self._all_rows) for out in sc.outputs])

        while queue:
            layer, rows_idx = queue.popleft()

            if layer in visited:
                continue
            visited.add(layer)

            layer_in = sc.layer_inputs(layer)
            layer_out = sc.layer_outputs(layer)

            if isinstance(layer, InputLayer):
                scope = list(layer.scope)

                param = self._make_input_param_estimated(
                    feat_idx=int(scope[0]),
                    instance_ids=rows_idx,
                    data=data,
                    num_input_units=layer.num_output_units,
                    num_categories=layer.num_categories,
                )

                layer.probs = param

            elif isinstance(layer, SumLayer):
                feat_ids = torch.tensor(list(sc._scopes[layer]), dtype=torch.long, device=data.device)
                clusters = self._cluster_instances(feat_ids, rows_idx, data, len(layer_in))

                param = self._make_sum_param_estimated(
                    clusters=clusters,
                    rows_idx=rows_idx,
                    num_input_units=layer.num_input_units,
                    num_sum_units=(1 if layer_out is None else layer.num_output_units),
                    activation=activation,
                )

                layer.weight = param

                for child, cluster_ids in zip(layer_in, clusters):
                    if child not in visited:
                        queue.append((child, cluster_ids))
            else:
                for child in layer_in:
                    if child not in visited:
                        queue.append((child, rows_idx))

        return sc

    def _cluster_instances(
        self,
        feat_ids: LongTensor,
        instance_ids: LongTensor,
        data: Tensor,
        n_clusters: int = 2,
        mode: str = "euclidean",
        verbose: int = 0
    ) -> List[LongTensor]:

        if instance_ids.numel() == 0:
            return [instance_ids.new_empty((0,), dtype=torch.long) for _ in range(n_clusters)]

        kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=verbose)

        if self.use_miwae:
            C, H, W = self.image_shape
            N = instance_ids.numel()
            sub = data.index_select(0, instance_ids)
            imgs = torch.zeros((N, C, H, W), device=self.device, dtype=torch.float32)

            for feat_idx in feat_ids.tolist():
                y, x = self.coords[feat_idx]
                imgs[:, 0, y, x] = sub[:, feat_idx].float() / 255.0

            with torch.no_grad():
                mu, _, _, _ = self.miwae.encoder(imgs) #log_var in pos 2
                feats = mu.detach() #torch.cat([mu, log_var], dim=1).detach()
        else:
            feats = data.index_select(0, instance_ids).index_select(1, feat_ids).float()
            
        labels = kmeans.fit_predict(feats)

        clusters: List[LongTensor] = []
        for c in range(n_clusters):
            mask = (labels == c)
            clusters.append(instance_ids[mask])

        return clusters

    def _estimate_marginal(
        self,
        rows: LongTensor,
        data: LongTensor,
        feat_idx: int,
        num_categories: int,
    ) -> np.ndarray:
        if rows is None or len(rows) == 0:
            return np.full(num_categories, 1.0 / num_categories, dtype=float)
        col = data[rows, feat_idx]
        counts = torch.bincount(col, minlength=num_categories).float().cpu().numpy()
        counts = counts + self._alpha_per_bin(num_categories)
        return counts / counts.sum()
    

    def _make_input_param_estimated(
        self,
        feat_idx: int,
        instance_ids: LongTensor,
        data: LongTensor,
        num_input_units: int,
        num_categories: int,
    ) -> Parameter:
        
        input_activation = 'softmax'

        probs = self._estimate_marginal(instance_ids, data, feat_idx, num_categories)

        if num_input_units == 1:
            logits = probs.reshape(1, -1)
            logits = np.tile(probs.reshape(1, -1), (num_input_units, 1))

        theta = self._to_preactivation(logits, input_activation)
        theta = self._apply_symmetry_breaking(theta, input_activation)

        tp = TensorParameter(
            num_input_units,
            num_categories,
            initializer=ConstantTensorInitializer(theta),
            learnable=True,
        )
        unary_op_factory = name_to_parameter_activation(input_activation)
        return Parameter.from_unary(unary_op_factory((num_input_units, num_categories)), tp)


    def _cluster_mixture_weights(self, clusters: List[LongTensor]) -> np.ndarray:
        sizes = np.array([int(c.numel()) for c in clusters], dtype=float)
        sizes = sizes + self._alpha_per_bin(len(clusters))
        total = sizes.sum()
        if total <= 0:
            return np.full(len(clusters), 1.0 / len(clusters), dtype=float)
        return sizes / total


    def _make_sum_param_estimated(
        self,
        clusters: List[LongTensor],
        rows_idx: LongTensor,
        num_input_units: int,
        num_sum_units: int,
        activation: str,
    ) -> Parameter:

        arity = len(clusters)
        base_mix = self._cluster_mixture_weights(clusters)

        if activation == 'clamp':
            activation = 'none'

        if num_sum_units == 1 and num_input_units == 1:
            theta = self._to_preactivation(base_mix.reshape(1, arity), activation)
            theta = self._apply_symmetry_breaking(theta, activation)
            tp = TensorParameter(1, arity, initializer=ConstantTensorInitializer(theta), learnable=True)
            unary_op_factory = name_to_parameter_activation(activation)
            if unary_op_factory is None:
                return Parameter.from_input(tp)
            return Parameter.from_unary(unary_op_factory((1, arity)), tp)
        
        per_unit_mix = np.tile(base_mix.reshape(1, -1), (num_sum_units, 1))

        expanded = np.repeat(per_unit_mix[:, :, None] / num_input_units, num_input_units, axis=2)
        expanded = expanded.reshape(num_sum_units, arity * num_input_units)

        theta = self._to_preactivation(expanded, activation)                  
        theta = self._apply_symmetry_breaking(theta, activation)             

        tp = TensorParameter(
            num_sum_units,
            arity * num_input_units,
            initializer=ConstantTensorInitializer(theta),
            learnable=True,
        )

        unary_op_factory = name_to_parameter_activation(activation)

        if unary_op_factory is None:
            return Parameter.from_input(tp)
        return Parameter.from_unary(unary_op_factory((num_sum_units, num_input_units * arity)), tp)