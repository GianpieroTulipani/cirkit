"""Check global raw-parameter projection and local parameter activations."""

import pytest
import torch

from cirkit.pipeline import PipelineContext
from cirkit.symbolic import functional
from cirkit.templates.data_modalities import image_data
from cirkit.templates.initialization import DataDrivenInitialization
from cirkit.templates.utils import POSITIVE_PARAMETER_MIN, Parameterization
from scripts.evaluate_image_circuit import build_parser, make_nll_function, train_circuit


@pytest.mark.parametrize("activation", ["none", "clamp", "softmax", "softplus", "sigmoid"])
@pytest.mark.parametrize("estimated", [False, True])
def test_training_projects_all_parameters_only_for_none(
    activation, estimated, monkeypatch, tmp_path
):
    data = torch.tensor([[0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0]])
    parameter_activation = "positive-clamp" if activation == "clamp" else activation
    symbolic = image_data(
        (3, 1, 2),
        input_layer="categorical",
        num_input_units=2,
        num_sum_units=2,
        sum_product_layer="cp",
        input_sharing="full",
        data=data,
        input_activation=parameter_activation,
        sum_weight_param=Parameterization(
            activation=parameter_activation,
            initialization="uniform",
        ),
        initialization=DataDrivenInitialization() if estimated else None,
    )
    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=True)
    circuit = ctx.compile(symbolic)
    partition = ctx.compile(functional.integrate(symbolic))
    parameters = list({id(p): p for p in circuit.parameters()}.values())
    with torch.no_grad():
        for parameter in parameters:
            parameter.fill_(-2)

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
        if activation == "none":
            assert all((p >= POSITIVE_PARAMETER_MIN).all() for p in parameters)
        loss = nll(batch)
        assert torch.isfinite(loss)
        return loss

    with torch.enable_grad():
        train_circuit(
            circuit,
            partition,
            [(data,)],
            [(data,)],
            activation=activation,
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
    for parameter in parameters:
        expected = POSITIVE_PARAMETER_MIN if activation == "none" else -2.0
        torch.testing.assert_close(parameter, torch.full_like(parameter, expected), rtol=0, atol=0)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_runner_exposes_one_activation_for_the_entire_circuit():
    parser = build_parser()
    for activation in ("none", "clamp", "softmax", "softplus", "sigmoid"):
        args = parser.parse_args(["--activation", activation])
        assert args.activation == activation
        assert not hasattr(args, "input_activation")
