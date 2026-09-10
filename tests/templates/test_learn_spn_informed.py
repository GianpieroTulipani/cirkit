import itertools
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.symbolic.layers import (
    CategoricalLayer,
    HadamardLayer,
    MultichannelCategoricalLayer,
    SumLayer,
)
from cirkit.templates.learn_spn import LearnSPN
from cirkit.templates.learn_spn_informed import (
    InformedLearnSPN,
    categorical_mutual_information,
    local_image_pairs,
)
from cirkit.utils.scope import Scope


def independent_blocks() -> torch.Tensor:
    return torch.tensor([(a, a, b, b) for a, b in itertools.product(range(2), repeat=2)] * 20)


@pytest.mark.parametrize("alpha", [0.0, 0.03])
@pytest.mark.parametrize("pair_batch,sample_batch", [(1, 3), (2, 17), (32, 1000)])
def test_sparse_categorical_mi_reference(alpha: float, pair_batch: int, sample_batch: int) -> None:
    generator = torch.Generator().manual_seed(8)
    data = torch.randint(4, (41, 5), generator=generator)
    data[:, 1] = data[:, 0]
    data[:, 4] = 0
    pairs = torch.tensor([[0, 1], [1, 2], [0, 4]])
    rows = torch.arange(0, len(data), 2)
    expected = []
    for src, dst in pairs.tolist():
        counts = np.zeros((4, 4))
        for left, right in data[rows][:, [src, dst]].tolist():
            counts[left, right] += 1
        joint = (counts + alpha) / (len(rows) + 16 * alpha)
        outer = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
        positive = joint > 0
        expected.append(np.sum(joint[positive] * np.log(joint[positive] / outer[positive])))
    actual = categorical_mutual_information(
        data,
        pairs,
        rows=rows,
        num_categories=4,
        alpha=alpha,
        pair_batch_size=pair_batch,
        sample_batch_size=sample_batch,
    )
    torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 2, 3), (3, 2, 3)])
@pytest.mark.parametrize("radius", [0, 1, 3])
def test_local_pairs_match_spatial_neighborhood(shape: tuple[int, int, int], radius: int) -> None:
    pairs = local_image_pairs(shape, radius)
    _, height, width = shape
    expected = set()
    for src, dst in itertools.combinations(range(np.prod(shape)), 2):
        y1, x1 = divmod(src % (height * width), width)
        y2, x2 = divmod(dst % (height * width), width)
        if abs(y1 - y2) + abs(x1 - x2) <= radius:
            expected.add((src, dst))
    assert len(pairs) == len(expected)
    assert set(map(tuple, pairs.tolist())) == expected


def test_informed_split_recovers_independent_blocks() -> None:
    learner = InformedLearnSPN(
        image_shape=(1, 1, 4),
        num_categories=2,
        mi_alpha=0,
        mi_threshold=0.1,
        min_instances=1,
        min_cluster_size=2,
        max_depth=1,
        force_root_mixture=False,
    )
    circuit = learner.learn_spn(independent_blocks(), num_input_units=2, num_sum_units=3)
    (output,) = circuit.outputs
    (product,) = circuit.layer_inputs(output)
    assert isinstance(product, HadamardLayer)
    assert {circuit.layer_scope(child) for child in circuit.layer_inputs(product)} == {
        Scope([0, 1]),
        Scope([2, 3]),
    }
    assert circuit.is_smooth and circuit.is_decomposable


# Check that the subclass reuses the original implementation, including private helpers.
# pylint: disable-next=protected-access
def test_initialization_is_inherited_unchanged() -> None:
    assert InformedLearnSPN._estimate_parameters is LearnSPN._estimate_parameters
    assert InformedLearnSPN._make_input_param is LearnSPN._make_input_param
    assert InformedLearnSPN._make_sum_param_estimated is LearnSPN._make_sum_param_estimated
    assert InformedLearnSPN._cluster_instances is LearnSPN._cluster_instances
    learner = InformedLearnSPN(
        image_shape=(1, 1, 4),
        num_categories=2,
        alpha=0.7,
        noise_scale=0.15,
        adaptive_alpha=False,
    )
    with patch.object(
        learner, "_estimate_parameters", wraps=learner._estimate_parameters
    ) as estimate:
        circuit = learner.learn_spn(independent_blocks(), activation="sigmoid")
    estimate.assert_called_once()
    assert estimate.call_args.args[0] is circuit
    torch.testing.assert_close(estimate.call_args.args[1], independent_blocks())
    assert estimate.call_args.args[2] == "sigmoid"
    assert learner.alpha == 0.7 and learner.noise_scale == 0.15 and not learner.adaptive_alpha


def test_estimation_can_be_disabled() -> None:
    learner = InformedLearnSPN(image_shape=(1, 1, 4), num_categories=2)
    with patch.object(learner, "_estimate_parameters") as estimate:
        circuit = learner.learn_spn(independent_blocks(), use_estimated_weights=False)
    estimate.assert_not_called()
    assert circuit.is_smooth and circuit.is_decomposable


@pytest.mark.parametrize("fold,optimize", list(itertools.product([False, True], repeat=2)))
@pytest.mark.parametrize("estimated", [False, True])
def test_symbolic_circuit_compiles_and_normalizes(
    fold: bool, optimize: bool, estimated: bool
) -> None:
    data = torch.tensor([[0, 0, 0]] * 30 + [[1, 1, 1]] * 10)
    learner = InformedLearnSPN(
        image_shape=(1, 1, 3),
        num_categories=2,
        min_instances=1,
        min_cluster_size=2,
        max_depth=1,
    )
    symbolic = learner.learn_spn(
        data,
        num_input_units=2,
        num_sum_units=3,
        num_classes=2,
        use_estimated_weights=estimated,
    )
    assert symbolic.is_smooth and symbolic.is_decomposable
    assert any(isinstance(layer, SumLayer) and layer.arity == 2 for layer in symbolic.sum_layers)
    compiled = TorchCompiler(fold=fold, optimize=optimize, semiring="lse-sum").compile(symbolic)
    worlds = torch.tensor(list(itertools.product(range(2), repeat=3)))
    with torch.enable_grad():
        outputs = compiled(worlds)
        assert outputs.shape == (8, 1, 2)
        torch.testing.assert_close(outputs.exp().sum(dim=0), torch.ones(1, 2))
        (-outputs[0].mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in compiled.parameters())


def test_quad_initialization_matches_analytical_density() -> None:
    data = torch.tensor([[0, 0]] * 30 + [[1, 1]] * 10)
    learner = InformedLearnSPN(
        image_shape=(1, 1, 2),
        num_categories=2,
        min_instances=1,
        min_cluster_size=2,
        max_depth=1,
        noise_scale=0,
        alpha=0.5,
    )
    symbolic = learner.learn_spn(data, num_input_units=2, num_sum_units=3)
    compiled = TorchCompiler(semiring="sum-product", fold=True, optimize=True).compile(symbolic)
    worlds = torch.tensor(list(itertools.product(range(2), repeat=2)))
    expected = torch.zeros(4)
    for value, count in ((0, 30), (1, 10)):
        probs = torch.full((2,), 0.25)
        probs[value] += count
        probs /= count + 0.5
        expected += (count + 0.25) / 40.5 * probs[worlds[:, 0]] * probs[worlds[:, 1]]
    torch.testing.assert_close(compiled(worlds).flatten(), expected, atol=1e-7, rtol=1e-6)


def test_custom_variable_split() -> None:
    calls = []

    def split(
        features: torch.Tensor, rows: torch.Tensor, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(features.tolist())
        assert len(rows) == len(data)
        return features[:1], features[1:]

    learner = InformedLearnSPN(
        data_format="tabular",
        num_categories=2,
        min_instances=1,
        force_root_mixture=False,
        split_features=split,
    )
    circuit = learner.learn_spn(independent_blocks(), use_estimated_weights=False)
    assert calls == [[0, 1, 2, 3], [1, 2, 3], [2, 3]]
    assert circuit.is_smooth and circuit.is_decomposable


@pytest.mark.parametrize(
    "parts",
    [
        ([0, 1], [1, 2, 3]),
        ([0], [1]),
        ([], [0, 1, 2, 3]),
        ([0, 1], [2, 5]),
    ],
)
def test_invalid_custom_partitions(parts: tuple[list[int], list[int]]) -> None:
    learner = InformedLearnSPN(
        image_shape=(1, 1, 4),
        num_categories=2,
        min_instances=1,
        force_root_mixture=False,
        split_features=lambda *args: tuple(torch.tensor(p, dtype=torch.long) for p in parts),
    )
    with pytest.raises(ValueError, match="split|Split"):
        learner.learn_spn(independent_blocks())


@pytest.mark.parametrize("sharing", ["none", "full"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 1, 1), (3, 1, 2)])
def test_small_data_and_existing_input_sharing(sharing: str, shape: tuple[int, int, int]) -> None:
    data = torch.zeros((1, int(np.prod(shape))), dtype=torch.long)
    learner = InformedLearnSPN(image_shape=shape, num_categories=2, input_sharing=sharing)
    symbolic = learner.learn_spn(data, num_input_units=2, num_sum_units=3, num_classes=2)
    compiled = TorchCompiler(semiring="lse-sum", fold=True, optimize=True).compile(symbolic)
    assert torch.isfinite(compiled(data)).all()
    if sharing == "full" and shape[0] > 1:
        assert all(isinstance(leaf, MultichannelCategoricalLayer) for leaf in symbolic.inputs)
    else:
        assert all(isinstance(leaf, CategoricalLayer) for leaf in symbolic.inputs)


def test_rgb_splitting_keeps_pixel_channels_together() -> None:
    data = torch.tensor([[a, b, a, b, a, b] for a, b in itertools.product(range(2), repeat=2)] * 10)
    learner = InformedLearnSPN(
        image_shape=(3, 1, 2),
        input_sharing="full",
        num_categories=2,
        min_instances=1,
        force_root_mixture=False,
        mi_alpha=0,
        mi_threshold=0.1,
    )
    symbolic = learner.learn_spn(data, num_input_units=2, num_sum_units=3)
    assert {symbolic.layer_scope(leaf) for leaf in symbolic.inputs} == {
        Scope([0, 2, 4]),
        Scope([1, 3, 5]),
    }
    compiled = TorchCompiler(semiring="lse-sum", fold=True, optimize=True).compile(symbolic)
    worlds = torch.tensor(list(itertools.product(range(2), repeat=6)))
    torch.testing.assert_close(compiled(worlds).exp().sum(), torch.tensor(1.0))


def test_degenerate_clustering_terminates() -> None:
    learner = InformedLearnSPN(
        image_shape=(1, 1, 2),
        num_categories=2,
        min_instances=1,
        min_cluster_size=1,
    )
    circuit = learner.learn_spn(torch.zeros((20, 2), dtype=torch.long))
    assert all(layer.arity == 1 for layer in circuit.sum_layers)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid", "softplus", "none", "clamp"])
def test_quad_sum_activations(activation: str) -> None:
    learner = InformedLearnSPN(
        image_shape=(1, 1, 4),
        num_categories=2,
        noise_scale=0,
    )
    symbolic = learner.learn_spn(independent_blocks(), activation=activation, num_sum_units=3)
    compiled = TorchCompiler(semiring="sum-product", fold=True, optimize=True).compile(symbolic)
    assert torch.isfinite(compiled(independent_blocks())).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_mi_and_learning() -> None:
    data = independent_blocks()
    pairs = local_image_pairs((1, 1, 4), 3)
    cpu = categorical_mutual_information(data, pairs, num_categories=2)
    gpu = categorical_mutual_information(data.cuda(), pairs.cuda(), num_categories=2)
    torch.testing.assert_close(cpu, gpu.cpu())
    learner = InformedLearnSPN(
        image_shape=(1, 1, 4),
        num_categories=2,
        min_instances=1,
        min_cluster_size=2,
        max_depth=2,
    )
    symbolic = learner.learn_spn(data.cuda(), num_input_units=2, num_sum_units=3)
    compiled = TorchCompiler(semiring="lse-sum", fold=True, optimize=True).compile(symbolic).cuda()
    assert torch.isfinite(compiled(data.cuda())).all()
