"""Check raw input potentials against exact enumeration and projected training."""

import itertools

import pytest
import torch

from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.pipeline import PipelineContext
from cirkit.symbolic import functional
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, SumLayer
from cirkit.templates.data_modalities import image_data, tabular_data
from cirkit.templates.initialization import DataDrivenInitialization, initialize_parameters
from cirkit.templates.utils import (
    POSITIVE_PARAMETER_MIN,
    Parameterization,
    categorical_input_layer_factory,
    parameterization_to_factory,
)
from cirkit.utils.scope import Scope


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("sharing", ["none", "full"])
@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("fold,optimize", list(itertools.product([False, True], repeat=2)))
def test_raw_input_partition_gradients_and_marginals(channels, sharing, estimated, fold, optimize):
    factory = categorical_input_layer_factory(
        num_categories=2, num_channels=channels, sharing=sharing, activation="none"
    )
    leaves = [factory(Scope(range(pixel, 2 * channels, 2)), 2) for pixel in range(2)]
    product = HadamardLayer(2, arity=2)
    sum_param = Parameterization(activation="none", initialization="uniform")
    root = SumLayer(2, 1, weight_factory=parameterization_to_factory(sum_param))
    symbolic = Circuit([*leaves, product, root], {product: leaves, root: [product]}, [root])
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=2 * channels)))
    if estimated:
        initialize_parameters(
            symbolic,
            worlds,
            initialization=DataDrivenInitialization(noise_scale=0.4),
            sum_weight_param=sum_param,
            input_sharing=sharing,
            input_activation="none",
        )
    assert all(layer.probs is None and layer.logits is not None for layer in leaves)
    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=fold, optimize=optimize)
    circuit = ctx.compile(symbolic)
    partition = ctx.compile(functional.integrate(symbolic))
    marginal = ctx.compile(functional.integrate(symbolic, scope=Scope([0])))
    input_params = list(
        {
            id(p): p
            for layer in circuit.layers
            if isinstance(layer, TorchInputLayer)
            for p in layer.parameters()
        }.values()
    )
    assert sum(p.numel() for p in input_params) == 4 * channels * (1 if sharing == "full" else 2)
    # Deliberately change the input mass so an assumed unit integral cannot pass.
    for p in input_params:
        p.mul_(3)
    assert not torch.allclose(partition(), torch.zeros_like(partition()))

    with torch.enable_grad():
        optimizer = torch.optim.SGD(circuit.parameters(), lr=0.01)
        for _ in range(2):
            scores = circuit(worlds).flatten()
            log_z = partition().sum()
            enumerated_log_z = scores.logsumexp(dim=0)
            torch.testing.assert_close(log_z, enumerated_log_z)
            torch.testing.assert_close((scores - log_z).exp().sum(), torch.tensor(1.0))
            expected_marginal = scores.reshape(2, -1).logsumexp(dim=0).repeat(2)
            torch.testing.assert_close(marginal(worlds).flatten(), expected_marginal)
            grads = torch.autograd.grad(log_z, input_params, retain_graph=True)
            expected_grads = torch.autograd.grad(enumerated_log_z, input_params, retain_graph=True)
            for actual, expected in zip(grads, expected_grads):
                torch.testing.assert_close(actual, expected)
            optimizer.zero_grad()
            loss = -(scores[:3] - log_z).mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in input_params)
            optimizer.step()
            with torch.no_grad():
                for p in circuit.parameters():
                    p.clamp_(min=POSITIVE_PARAMETER_MIN)


@pytest.mark.parametrize("sharing", ["none", "full"])
@pytest.mark.parametrize("estimated", [False, True])
def test_rgb_raw_inputs_are_positive_before_first_forward(sharing, estimated):
    data = torch.zeros(2, 3, 1, 2, dtype=torch.uint8)
    symbolic = image_data(
        (3, 1, 2),
        data=data,
        input_layer="categorical",
        input_activation="none",
        input_sharing=sharing,
        num_input_units=2,
        num_sum_units=2,
        sum_product_layer="cp",
        initialization=DataDrivenInitialization(alpha=0.0, noise_scale=0.0) if estimated else None,
    )
    assert all(layer.probs is None for layer in symbolic.input_layers)
    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=True)
    circuit = ctx.compile(symbolic)
    partition = ctx.compile(functional.integrate(symbolic))
    # Includes unseen categories when alpha=0, which must still have a finite logarithm.
    with torch.enable_grad():
        nll = -(circuit(torch.full((2, 6), 255)) - partition()).mean()
        assert torch.isfinite(nll)
        nll.backward()
    params = {
        id(p): p
        for layer in circuit.layers
        if isinstance(layer, TorchInputLayer)
        for p in layer.parameters()
    }
    assert all((p >= POSITIVE_PARAMETER_MIN).all() for p in params.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())


@pytest.mark.parametrize("estimated", [False, True])
def test_tabular_raw_inputs_with_heterogeneous_categories(estimated):
    data = torch.tensor([[0, 0], [1, 2], [0, 1]])
    symbolic = tabular_data(
        data=data,
        input_layers=[{"name": "categorical", "args": {"num_categories": n}} for n in (2, 3)],
        input_activation="none",
        num_input_units=2,
        num_sum_units=2,
        sum_product_layer="cp",
        initialization=DataDrivenInitialization() if estimated else None,
    )
    ctx = PipelineContext(backend="torch", semiring="lse-sum")
    circuit = ctx.compile(symbolic)
    partition = ctx.compile(functional.integrate(symbolic))
    worlds = torch.tensor(list(itertools.product(range(2), range(3))))
    torch.testing.assert_close(circuit(worlds).flatten().logsumexp(0), partition().sum())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_activation": "unknown"},
        {"input_activation": "none", "input_layer": "gaussian"},
        {"input_activation": "none", "input_params": {}},
    ],
)
def test_invalid_image_input_activation(kwargs):
    options = dict(
        input_layer="categorical", num_input_units=1, num_sum_units=1, sum_product_layer="cp"
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        image_data((1, 1, 2), **options)
