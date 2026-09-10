"""LearnSPN with local categorical MI and direct Cirkit symbolic layers.

Like the original learn-spn branch, recursively split variables or instances and
assemble a Circuit. Clustering and parameter initialization are inherited unchanged
from quad-spn's LearnSPN. Compilation and training are left to the caller.
"""

import functools
from collections.abc import Callable
from math import isfinite, prod
from typing import Any, cast

import torch
from torch import LongTensor, Tensor

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, Layer, SumLayer
from cirkit.symbolic.parameters import mixing_weight_factory
from cirkit.templates.learn_spn import LearnSPN
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from cirkit.utils.scope import Scope


def local_image_pairs(
    shape: tuple[int, int, int], radius: int, *, device: torch.device | str = "cpu"
) -> Tensor:
    """Return unique variable pairs within a spatial L1 radius, including cross-channel pairs.

    Variables use CHW order. Channels at the same pixel are neighbors even at radius
    zero. Work and storage are O(H W C^2 radius^2), rather than all-pairs in H W.
    """
    if len(shape) != 3 or any(not isinstance(v, int) or v <= 0 for v in shape):
        raise ValueError("image_shape must contain three positive integers (C, H, W)")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    channels, height, width = shape
    grid = torch.arange(channels * height * width, device=device).reshape(shape)
    pairs = []
    for dy in range(-min(radius, height - 1), min(radius, height - 1) + 1):
        reach = min(radius - abs(dy), width - 1)
        for dx in range(-reach, reach + 1):
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            src = grid[:, y0:y1, x0:x1].reshape(channels, -1)
            dst = grid[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx].reshape(channels, -1)
            src = src[:, None, :].expand(channels, channels, -1)
            dst = dst[None, :, :].expand(channels, channels, -1)
            keep = src < dst
            pairs.append(torch.stack((src[keep], dst[keep]), dim=1))
    return torch.cat(pairs)


def categorical_mutual_information(
    data: Tensor,
    pairs: Tensor,
    *,
    num_categories: int,
    rows: Tensor | None = None,
    alpha: float = 0.01,
    pair_batch_size: int = 32,
    sample_batch_size: int = 1024,
) -> Tensor:
    """Estimate categorical MI in nats for selected pairs, in two-dimensional batches.

    ``alpha`` is the pseudocount per joint cell; marginals are derived from the
    smoothed joint, including when alpha is zero. Histograms retain all categories.
    Scratch storage is O(pair_batch_size * (num_categories^2 + sample_batch_size)).
    Inputs must contain valid category IDs; the learner validates the full dataset
    once before entering the recursive estimator.
    """
    if data.ndim != 2 or data.dtype != torch.long:
        raise ValueError("data must be a two-dimensional torch.long tensor")
    if num_categories < 2 or not isfinite(alpha) or alpha < 0:
        raise ValueError("num_categories must be >= 2 and alpha must be finite and non-negative")
    if pair_batch_size <= 0 or sample_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("pairs must be a torch.long tensor of shape (num_pairs, 2)")
    pairs = pairs.to(data.device)
    if rows is None:
        rows = torch.arange(len(data), device=data.device)
    if rows.ndim != 1 or rows.dtype != torch.long or not rows.numel():
        raise ValueError("rows must be a non-empty vector of torch.long indices")
    rows = rows.to(data.device)
    result = torch.empty(len(pairs), dtype=torch.float64, device=data.device)
    for start in range(0, len(pairs), pair_batch_size):
        batch = pairs[start : start + pair_batch_size]
        counts = torch.zeros((len(batch), num_categories**2), dtype=torch.long, device=data.device)
        for row_batch in rows.split(sample_batch_size):
            left = data[row_batch[:, None], batch[:, 0]]
            right = data[row_batch[:, None], batch[:, 1]]
            joint_ids = (left * num_categories + right).T
            counts.scatter_add_(1, joint_ids, torch.ones_like(joint_ids))
        joint = counts.to(torch.float64).reshape(-1, num_categories, num_categories)
        joint.add_(alpha).div_(len(rows) + alpha * num_categories**2)
        px, py = joint.sum(dim=2), joint.sum(dim=1)
        # Entropy form handles zero counts without log(0) or NaNs when alpha=0.
        mi = (
            torch.xlogy(joint, joint).sum(dim=(1, 2))
            - torch.xlogy(px, px).sum(dim=1)
            - torch.xlogy(py, py).sum(dim=1)
        )
        result[start : start + len(batch)] = mi.clamp_min(0)
    return result


class InformedLearnSPN(LearnSPN):
    """Add informed structure learning to the existing quad-spn initializer.

    Extra keyword arguments are passed to LearnSPN unchanged, including alpha,
    noise_scale, adaptive_alpha, input_sharing, image_shape, device and seed.
    mi_alpha controls only MI estimation, not circuit parameter initialization.

    split_features, if supplied, receives (feature_ids, row_ids, data) and returns
    two disjoint feature tensors covering the parent, or None to try clustering.
    Image columns use flattened CHW order. Tabular inputs use all variable pairs
    unless neighbor_pairs is supplied to learn_spn.
    """

    def __init__(
        self,
        *,
        min_instances: int = 1000,
        min_cluster_size: int = 10,
        num_clusters: int = 2,
        max_depth: int = 32,
        local_radius: int = 3,
        mi_quantile: float = 0.6,
        mi_threshold: float | None = None,
        mi_alpha: float = 0.01,
        force_root_mixture: bool = True,
        split_features: (
            Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor] | None] | None
        ) = None,
        data_format: str = "image",
        **kwargs: Any,
    ) -> None:
        super().__init__(data_format=data_format, **kwargs)
        if min_instances < 1 or min_cluster_size < 1 or num_clusters < 2:
            raise ValueError("Sample limits must be positive and num_clusters must be >= 2")
        if max_depth < 0 or local_radius < 0 or not 0 <= mi_quantile <= 1:
            raise ValueError("Invalid depth, radius or MI quantile")
        if not isfinite(mi_alpha) or mi_alpha < 0:
            raise ValueError("mi_alpha must be finite and non-negative")
        if mi_threshold is not None and (not isfinite(mi_threshold) or mi_threshold < 0):
            raise ValueError("mi_threshold must be finite and non-negative")
        self.min_instances = min_instances
        self.min_cluster_size = min_cluster_size
        self.num_clusters = num_clusters
        self.max_depth = max_depth
        self.local_radius = local_radius
        self.mi_quantile = mi_quantile
        self.mi_threshold = mi_threshold
        self.mi_alpha = mi_alpha
        self.force_root_mixture = force_root_mixture
        self.split_features = split_features

    @torch.no_grad()
    def learn_spn(  # type: ignore[override]
        self,
        data: Tensor,
        input_layer: str = "categorical",
        activation: str = "softmax",
        weights_init: str = "normal",
        num_input_units: int = 1,
        num_sum_units: int = 1,
        num_classes: int = 1,
        use_estimated_weights: bool = True,
        use_mixing_weights: bool = True,
        sum_weight_param: Parameterization | None = None,
        *,
        neighbor_pairs: Tensor | None = None,
        pair_batch_size: int = 32,
        sample_batch_size: int = 1024,
    ) -> Circuit:
        """Build symbolic sum/product layers; optionally run quad-spn initialization.

        Only categorical data is supported. Products use CP-style projections to
        num_sum_units before a Hadamard product, allowing different input widths.
        No compiler is invoked here. The initialization pass is exactly the inherited
        _estimate_parameters, including its instance clustering and sharing behavior.
        """
        if input_layer != "categorical":
            raise ValueError("Informed structure learning currently requires categorical inputs")
        if data.ndim != 2 or data.dtype != torch.long or not all(data.shape):
            raise ValueError("data must be a non-empty (N, D) torch.long tensor")
        if data.min().item() < 0 or data.max().item() >= self.num_categories:
            raise ValueError("Data categories must lie in [0, num_categories)")
        if self.data_format == "image" and prod(self.image_shape) != data.shape[1]:
            raise ValueError("image_shape does not match the number of data columns")
        if min(num_input_units, num_sum_units, num_classes, pair_batch_size, sample_batch_size) < 1:
            raise ValueError("Unit counts and batch sizes must be positive")
        data = self._prepare_estimation_data(cast(LongTensor, data))
        if neighbor_pairs is None:
            pairs = (
                local_image_pairs(self.image_shape, self.local_radius, device=data.device)
                if self.data_format == "image"
                else torch.triu_indices(
                    data.shape[1], data.shape[1], offset=1, device=data.device
                ).T
            )
        else:
            pairs = neighbor_pairs.to(data.device)
            if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("neighbor_pairs must have shape (E, 2) and dtype torch.long")
            pairs = pairs.sort(dim=1).values
            if pairs.numel() and (pairs.min().item() < 0 or pairs.max().item() >= data.shape[1]):
                raise ValueError("Neighbor variable IDs are outside the data scope")
            if torch.any(pairs[:, 0] == pairs[:, 1]) or len(pairs.unique(dim=0)) != len(pairs):
                raise ValueError("Neighbor pairs must be unique and exclude self-pairs")

        # Use the same factories and parameterization as quad-spn.
        input_factory = self._make_input_factory(input_layer, self.num_categories)
        if sum_weight_param is None:
            sum_weight_param = Parameterization(
                activation="none" if activation == "clamp" else activation,
                initialization="uniform" if activation == "clamp" else weights_init,
            )
        sum_factory = parameterization_to_factory(sum_weight_param)
        mixing_factory = (
            functools.partial(mixing_weight_factory, param_factory=sum_factory)
            if use_mixing_weights
            else sum_factory
        )
        layers: list[Layer] = []
        in_layers: dict[Layer, list[Layer]] = {}

        def add_sum(children: list[Layer], units: int) -> SumLayer:
            layer = SumLayer(
                children[0].num_output_units,
                units,
                arity=len(children),
                weight_factory=mixing_factory if len(children) > 1 else sum_factory,
            )
            layers.append(layer)
            in_layers[layer] = children
            return layer

        def add_product(children: list[Layer]) -> Layer:
            projected: list[Layer] = [add_sum([child], num_sum_units) for child in children]
            if len(projected) == 1:
                return projected[0]
            layer = HadamardLayer(num_sum_units, arity=len(projected))
            layers.append(layer)
            in_layers[layer] = projected
            return layer

        def add_leaves(features: Tensor) -> Layer:
            if self._use_multichannel_inputs(input_layer):
                pixels = self.image_shape[1] * self.image_shape[2]
                scopes = [
                    Scope(features[features % pixels == p].tolist())
                    for p in (features % pixels).unique().tolist()
                ]
            else:
                scopes = [Scope([feature]) for feature in features.tolist()]
            leaves = [input_factory(scope, num_input_units) for scope in scopes]
            layers.extend(leaves)
            return add_product(leaves)

        def learn(features: Tensor, rows: Tensor, depth: int) -> Layer:
            num_variables = len(features)
            if self._use_multichannel_inputs(input_layer):
                num_variables //= self.image_shape[0]
            if num_variables == 1 or len(rows) <= self.min_instances or depth >= self.max_depth:
                return add_leaves(features)

            split = None
            if not (depth == 0 and self.force_root_mixture):
                if self.split_features is not None:
                    split = self.split_features(features, rows, data)
                else:
                    split = self._split_features_local(
                        features, rows, data, pairs, pair_batch_size, sample_batch_size
                    )
            if split is not None:
                left, right = self._validate_split(features, split, input_layer)
                return add_product([learn(left, rows, depth + 1), learn(right, rows, depth + 1)])

            if len(rows) < self.num_clusters * self.min_cluster_size:
                return add_leaves(features)
            clusters = self._cluster_instances(
                cast(LongTensor, features), cast(LongTensor, rows), data, self.num_clusters
            )
            if any(len(c) < self.min_cluster_size or len(c) >= len(rows) for c in clusters):
                return add_leaves(features)
            children = [learn(features, c, depth + 1) for c in clusters]
            return add_sum(children, num_sum_units)

        output = learn(
            torch.arange(data.shape[1], device=data.device),
            torch.arange(len(data), device=data.device),
            0,
        )
        output = add_sum([output], num_classes)
        circuit = Circuit(layers, in_layers, [output])
        if use_estimated_weights:
            circuit = self._estimate_parameters(circuit, data, activation)
        return circuit

    def _split_features_local(
        self,
        features: Tensor,
        rows: Tensor,
        data: Tensor,
        pairs: Tensor,
        pair_batch_size: int,
        sample_batch_size: int,
    ) -> tuple[Tensor, Tensor] | None:
        """Separate the component of the first variable from the remaining variables."""
        pairs = pairs[torch.isin(pairs, features).all(dim=1)]
        mi = categorical_mutual_information(
            data,
            pairs,
            rows=rows,
            num_categories=self.num_categories,
            alpha=self.mi_alpha,
            pair_batch_size=pair_batch_size,
            sample_batch_size=sample_batch_size,
        )
        positive = mi[mi > 1e-12]
        threshold = self.mi_threshold
        if threshold is None:
            threshold = float(torch.quantile(positive, self.mi_quantile)) if len(positive) else 0.0

        # Group channels of a pixel when the existing shared RGB input factory is used.
        pixels = self.image_shape[1] * self.image_shape[2]
        shared_rgb = self._use_multichannel_inputs("categorical")
        ids = features % pixels if shared_rgb else features
        edges = pairs[(mi >= threshold) & (mi > 1e-12)]
        if shared_rgb:
            edges = edges % pixels
        neighbors: dict[int, list[int]] = {v: [] for v in ids.tolist()}
        for left, right in edges.cpu().tolist():
            neighbors[left].append(right)
            neighbors[right].append(left)
        visited = {int(ids[0])}
        stack = list(visited)
        while stack:
            for neighbor in neighbors[stack.pop()]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        mask = torch.isin(ids, torch.tensor(list(visited), device=data.device))
        return (features[mask], features[~mask]) if not mask.all() else None

    def _validate_split(
        self, features: Tensor, split: tuple[Tensor, Tensor], input_layer: str
    ) -> tuple[Tensor, Tensor]:
        """Reject overlapping, incomplete or empty user partitions."""
        if len(split) != 2 or any(
            part.ndim != 1 or part.dtype != torch.long or not len(part) for part in split
        ):
            raise ValueError("A split must contain two non-empty torch.long feature vectors")
        left, right = (part.to(features.device) for part in split)
        merged = torch.cat((left, right))
        if len(merged) != len(features) or not torch.equal(
            merged.sort().values, features.sort().values
        ):
            raise ValueError("Split variables must be disjoint and cover the parent scope")
        if self._use_multichannel_inputs(input_layer):
            pixels = self.image_shape[1] * self.image_shape[2]
            if torch.isin(left % pixels, right % pixels).any():
                raise ValueError("Shared RGB inputs require all channels of each pixel together")
        return left, right
