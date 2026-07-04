import random
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
    MixingWeightParameter,
    mixing_weight_factory,
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
        adaptive_alpha: bool = True,
        subcluster_lambda: float = 0.7
    ):

        assert data_format in ('image', 'tabular'), "data_format should be either 'image' or 'tabular'"
        assert 0.0 <= subcluster_lambda <= 1.0, "subcluster_lambda must be in [0, 1]"

        self.alpha = alpha
        self.use_miwae = use_miwae
        self.noise_scale = noise_scale
        self.image_shape = image_shape
        self.data_format = data_format

        self.adaptive_alpha = adaptive_alpha
        self.subcluster_lambda = float(subcluster_lambda)
        self._all_rows: Optional[LongTensor] = None

        self._subpop_cache: dict = {}
        self._data: Optional[LongTensor] = None

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

    @staticmethod
    def _clamp_floor() -> float:
        return float(np.sqrt(np.finfo(np.float32).tiny))

    def _activation_kwargs(self, activation: str) -> dict:
        if activation == 'positive-clamp':
            return {'vmin': self._clamp_floor()}
        return {}

    def _to_preactivation(self, probs: np.ndarray, activation: str) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
        if activation == 'positive-clamp':
            return p
        if activation == 'softplus':
            return np.log(np.expm1(p))
        return np.log(p)

    def _apply_symmetry_breaking(self, theta: np.ndarray, activation: str) -> np.ndarray:
        s = self.noise_scale
        if not s or s <= 0.0:
            return theta
        if activation == 'positive-clamp':
            # spazio lineare: rumore moltiplicativo positivo -> resta sopra il floor
            noisy = theta * np.exp(np.random.normal(loc=0.0, scale=s, size=theta.shape))
            return np.clip(noisy, self._clamp_floor(), None)
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

        if activation == 'positive-clamp' and weights_init == 'normal':
            weights_init = 'uniform'   

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

        activation_dict = {}
        nary_sum_weight_factory: ParameterFactory
        num_categories = int(data.max().item() + 1)
        input_factory = name_to_input_layer_factory(input_layer, num_categories=num_categories)

        if activation == 'positive-clamp':
            activation_dict['vmin'] = self._clamp_floor()

        if sum_weight_param is None:
            sum_weight_param = Parameterization(
                activation=activation,
                initialization=weights_init,
                activation_kwargs=activation_dict,
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
                if isinstance(layer.weight.output, MixingWeightParameter):
                    for child in layer_in:
                        if child not in visited:
                            queue.append((child, rows_idx))
                    continue

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
    ) -> List[LongTensor]:

        if instance_ids.numel() == 0:
            return [instance_ids.new_empty((0,), dtype=torch.long) for _ in range(n_clusters)]

        kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=0)

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
                embeddings = mu.detach() #torch.cat([mu, log_var], dim=1).detach()

            labels = kmeans.fit_predict(embeddings)
        else:
            sub = data.index_select(0, instance_ids).index_select(1, feat_ids).float()
            labels = kmeans.fit_predict(sub)

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

    def _leaf_pool_rows(self, instance_ids: LongTensor) -> LongTensor:
        if self.leaf_pool == 'global' and self._all_rows is not None:
            return self._all_rows
        return instance_ids

    # ------------------------------------------------------------------ #
    # diversify='subcluster': clustering globale in K sotto-popolazioni.
    # Le K unità di una regione sono le K componenti di una mixture: invece
    # del bootstrap (la cui diversità svanisce ~1/sqrt(|pool|)), si partiziona
    # il dataset in K gruppi coerenti e l'unità k prende le statistiche del
    # gruppo k. Diversità strutturale che scala con K e non dipende da N.
    # ------------------------------------------------------------------ #
    def _encode_all(self, data: LongTensor, batch: int = 512) -> np.ndarray:
        """Embedding MIWAE di TUTTE le immagini (a batch), per il clustering globale."""
        C, H, W = self.image_shape
        N = int(data.size(0))
        embs = []
        with torch.no_grad():
            for s in range(0, N, batch):
                chunk = data[s: s + batch]
                n = int(chunk.size(0))
                imgs = torch.zeros((n, C, H, W), device=self.device, dtype=torch.float32)
                for idx in range(H * W):
                    y, x = self.coords[idx]
                    imgs[:, 0, y, x] = chunk[:, idx].float() / 255.0
                mu, _, _, _ = self.miwae.encoder(imgs) #log_var in pos 2
                embs.append(mu.detach().cpu()) #torch.cat([mu, log_var], dim=1).detach().cpu()
        return torch.cat(embs, dim=0).numpy()

    def _subpop_labels(self, K: int) -> np.ndarray:
        """Etichetta di sotto-popolazione (in [0, K)) per ogni riga del dataset, cachata per K."""
        if K in self._subpop_cache:
            return self._subpop_cache[K]

        data = self._data
        assert data is not None, "_subpop_labels richiede _data (impostato in _estimate_parameters)"
        N = int(data.size(0))
        Keff = max(1, min(K, N))

        if Keff == 1:
            labels = np.zeros(N, dtype=int)
        else:
            if self.use_miwae:
                feats = torch.as_tensor(
                    self._encode_all(data), dtype=torch.float32, device=self.device
                )
            else:
                feats = data.float()
            km = KMeans(n_clusters=Keff, mode="euclidean", verbose=0)
            labels = km.fit_predict(feats).detach().cpu().numpy().astype(int)

        self._subpop_cache[K] = labels
        return labels

    def _per_unit_distributions(
        self,
        pool: LongTensor,
        data: LongTensor,
        feat_idx: int,
        num_units: int,
        num_categories: int,
    ) -> np.ndarray:
        if num_units == 1 or self.diversify == 'replicate':
            base = self._estimate_marginal(pool, data, feat_idx, num_categories)
            return np.tile(base.reshape(1, -1), (num_units, 1))

        if self.diversify == 'subcluster':
            # unità k = marginale del pixel sulla sotto-popolazione k del pool,
            # con shrinkage verso la base per gestire gruppi piccoli.
            base = self._estimate_marginal(pool, data, feat_idx, num_categories)
            if pool is None or pool.numel() == 0:
                return np.tile(base.reshape(1, -1), (num_units, 1))

            subpop = self._subpop_labels(num_units)
            pool_np = pool.detach().cpu().numpy()
            pool_sub = subpop[pool_np]
            lam = self.subcluster_lambda

            out = np.empty((num_units, num_categories), dtype=float)
            for k in range(num_units):
                grp = pool_np[pool_sub == k]
                if grp.size == 0:
                    out[k] = base  # sotto-gruppo vuoto -> base informativa
                else:
                    sub = torch.as_tensor(grp, dtype=torch.long, device=data.device)
                    dist_k = self._estimate_marginal(sub, data, feat_idx, num_categories)
                    out[k] = (1.0 - lam) * base + lam * dist_k
            return out

        rows = pool.detach().cpu().numpy()
        out = np.empty((num_units, num_categories), dtype=float)

        if rows.size == 0:
            out[:] = 1.0 / num_categories
            return out

        for k in range(num_units):
            samp = rows[np.random.randint(0, rows.size, size=rows.size)]
            sub = torch.as_tensor(samp, dtype=torch.long, device=data.device)
            out[k] = self._estimate_marginal(sub, data, feat_idx, num_categories)

        return out

    def _make_input_param_estimated(
        self,
        feat_idx: int,
        instance_ids: LongTensor,
        data: LongTensor,
        num_input_units: int,
        num_categories: int,
    ) -> Parameter:

        # L'input categorico DEVE restare una distribuzione normalizzata: si usa sempre
        # softmax, a prescindere dall'attivazione dei pesi sum. Le attivazioni non
        # normalizzanti (positive-clamp, softplus, sigmoid, none) lascerebbero la
        # categorica non normalizzata durante il training -> la partition function non la
        # insegue e il PC diventa improprio (c(x) > Z, NLL/bpd negativa). Questo replica il
        # default di cirkit (CategoricalLayer -> SoftmaxParameter).
        input_activation = 'softmax'

        pool = self._leaf_pool_rows(instance_ids)
        per_unit_probs = self._per_unit_distributions(
            pool=pool, data=data, feat_idx=feat_idx,
            num_units=num_input_units, num_categories=num_categories,
        )
        theta = self._to_preactivation(per_unit_probs, input_activation)   # log(p): logits per softmax
        theta = self._apply_symmetry_breaking(theta, input_activation)

        tp = TensorParameter(
            num_input_units,
            num_categories,
            initializer=ConstantTensorInitializer(theta),
            learnable=True,
        )
        unary_op_factory = name_to_parameter_activation(input_activation, **self._activation_kwargs(input_activation))
        return Parameter.from_unary(unary_op_factory((num_input_units, num_categories)), tp)


    def _cluster_mixture_weights(self, clusters: List[LongTensor]) -> np.ndarray:
        sizes = np.array([int(c.numel()) for c in clusters], dtype=float)
        sizes = sizes + self._alpha_per_bin(len(clusters))
        total = sizes.sum()
        if total <= 0:
            return np.full(len(clusters), 1.0 / len(clusters), dtype=float)
        return sizes / total

    def _row_cluster_labels(self, clusters: List[LongTensor], rows_idx: LongTensor) -> np.ndarray:
        label_of = {}
        for c, ids in enumerate(clusters):
            for i in ids.tolist():
                label_of[i] = c
        return np.array([label_of[int(i)] for i in rows_idx.tolist()], dtype=int)

    def _per_unit_mixtures(
        self,
        base_mix: np.ndarray,
        labels: np.ndarray,
        num_sum_units: int,
        arity: int,
    ) -> np.ndarray:
        if num_sum_units == 1 or self.diversify == 'replicate' or labels.size == 0:
            return np.tile(base_mix.reshape(1, -1), (num_sum_units, 1))

        a = self._alpha_per_bin(arity)
        out = np.empty((num_sum_units, arity), dtype=float)
        for k in range(num_sum_units):
            samp = labels[np.random.randint(0, labels.size, size=labels.size)]
            counts = np.bincount(samp, minlength=arity).astype(float) + a
            out[k] = counts / counts.sum()
        return out

    def _per_unit_mixtures_subcluster(
        self,
        clusters: List[LongTensor],
        rows_idx: LongTensor,
        num_sum_units: int,
        arity: int,
    ) -> np.ndarray:
        # unità k = proporzioni di mixture (sui figli) calcolate sulla sotto-popolazione k,
        # con shrinkage verso la mixture base.
        base_mix = self._cluster_mixture_weights(clusters)
        if num_sum_units == 1 or rows_idx.numel() == 0:
            return np.tile(base_mix.reshape(1, -1), (num_sum_units, 1))

        arity_labels = self._row_cluster_labels(clusters, rows_idx)  # figlio per riga di rows_idx
        rows_np = rows_idx.detach().cpu().numpy()
        subpop = self._subpop_labels(num_sum_units)
        row_sub = subpop[rows_np]

        a = self._alpha_per_bin(arity)
        lam = self.subcluster_lambda
        out = np.empty((num_sum_units, arity), dtype=float)
        for k in range(num_sum_units):
            mask = row_sub == k
            if not mask.any():
                out[k] = base_mix
            else:
                counts = np.bincount(arity_labels[mask], minlength=arity).astype(float) + a
                mix_k = counts / counts.sum()
                out[k] = (1.0 - lam) * base_mix + lam * mix_k
        return out

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

        if num_sum_units == 1 and num_input_units == 1:
            theta = self._to_preactivation(base_mix.reshape(1, arity), activation)
            theta = self._apply_symmetry_breaking(theta, activation)
            tp = TensorParameter(1, arity, initializer=ConstantTensorInitializer(theta), learnable=True)
            unary_op_factory = name_to_parameter_activation(activation, **self._activation_kwargs(activation))
            return Parameter.from_unary(unary_op_factory((1, arity)), tp)

        if self.diversify == 'subcluster':
            per_unit_mix = self._per_unit_mixtures_subcluster(clusters, rows_idx, num_sum_units, arity)
        else:
            labels = self._row_cluster_labels(clusters, rows_idx)
            per_unit_mix = self._per_unit_mixtures(base_mix, labels, num_sum_units, arity)

        expanded = np.repeat(per_unit_mix[:, :, None] / num_input_units, num_input_units, axis=2)
        expanded = expanded.reshape(num_sum_units, arity * num_input_units)

        theta = self._to_preactivation(expanded, activation)                  # OPT 1
        theta = self._apply_symmetry_breaking(theta, activation)              # OPT 4

        tp = TensorParameter(
            num_sum_units,
            arity * num_input_units,
            initializer=ConstantTensorInitializer(theta),
            learnable=True,
        )
        unary_op_factory = name_to_parameter_activation(activation, **self._activation_kwargs(activation))
        return Parameter.from_unary(unary_op_factory((num_sum_units, num_input_units * arity)), tp)