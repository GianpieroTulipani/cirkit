"""Check that projected training clamps raw weights without changing softmax logits."""

import itertools

import pytest
import torch

from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.pipeline import PipelineContext
from cirkit.symbolic import functional
from cirkit.templates.data_modalities import image_data
from cirkit.templates.initialization import DataDrivenInitialization
from cirkit.templates.utils import POSITIVE_PARAMETER_MIN, Parameterization
from scripts.evaluate_image_circuit import make_nll_function, train_circuit


@pytest.mark.parametrize(
    "sum_activation,input_activation", list(itertools.product(["softmax", "clamp"], repeat=2))
)
@pytest.mark.parametrize("estimated", [False, True])
def test_training_projects_only_raw_parameters(
    sum_activation, input_activation, estimated, monkeypatch, tmp_path
):
    data = torch.tensor([[0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0]])
    symbolic = image_data(
        (3, 1, 2),
        input_layer="categorical",
        num_input_units=2,
        num_sum_units=2,
        sum_product_layer="cp",
        input_sharing="full",
        data=data,
        input_activation="none" if input_activation == "clamp" else "softmax",
        sum_weight_param=Parameterization(
            activation="none" if sum_activation == "clamp" else "softmax",
            initialization="uniform",
        ),
        initialization=DataDrivenInitialization() if estimated else None,
    )
    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=True)
    circuit = ctx.compile(symbolic)
    partition = ctx.compile(functional.integrate(symbolic))
    input_params = [
        p
        for layer in circuit.layers
        if isinstance(layer, TorchInputLayer)
        for p in layer.parameters()
    ]
    sum_params = [
        p
        for layer in circuit.layers
        if not isinstance(layer, TorchInputLayer)
        for p in layer.parameters()
    ]
    groups = [(input_params, input_activation), (sum_params, sum_activation)]
    for params, activation in groups:
        for p in params:
            p.fill_(0 if activation == "clamp" else -2)

    adam_step = torch.optim.Adam.step

    def step_outside_positive_domain(optimizer, *args, **kwargs):
        result = adam_step(optimizer, *args, **kwargs)
        # Simulate an update crossing zero, for both raw weights and unrestricted logits.
        with torch.no_grad():
            for p in circuit.parameters():
                p.fill_(-2)
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", step_outside_positive_domain)
    nll = make_nll_function(circuit, partition)

    def checked_nll(batch):
        for params, activation in groups:
            if activation == "clamp":
                assert all((p >= POSITIVE_PARAMETER_MIN).all() for p in params)
        loss = nll(batch)
        assert torch.isfinite(loss)
        return loss

    with torch.enable_grad():
        train_circuit(
            circuit,
            partition,
            [(data,)],
            [(data,)],
            sum_params=sum_params,
            input_params=input_params,
            activation=sum_activation,
            input_activation=input_activation,
            max_train_steps=2,
            lr=0.01,
            T_0=1,
            eta_min=0.0,
            weight_decay=0.0,
            device="cpu",
            save_path=tmp_path / "model.pt",
            validation_steps=1,
            delta=0.0,
            patience=3,
            num_dimensions=6,
            log_to_wandb=False,
            nll_fn=checked_nll,
        )
    for params, activation in groups:
        for p in params:
            expected = POSITIVE_PARAMETER_MIN if activation == "clamp" else -2.0
            torch.testing.assert_close(p, torch.full_like(p, expected), rtol=0, atol=0)
            assert p.grad is not None and torch.isfinite(p.grad).all()
