"""Check image runner training, parameter activations, and region graph integration."""

import logging
import math
import re
import sys
from unittest.mock import patch

import pytest
import torch

from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.pipeline import PipelineContext
from cirkit.symbolic import functional
from cirkit.templates import data_modalities
from cirkit.templates.data_modalities import image_data
from cirkit.templates.initialization import DataDrivenInitialization
from cirkit.templates.utils import POSITIVE_PARAMETER_MIN, Parameterization
from scripts import evaluate_image_circuit
from scripts.evaluate_image_circuit import (
    TrainingConfig,
    build_parser,
    main,
    make_nll_function,
    make_pipeline_context,
    train_circuit,
)


@pytest.mark.parametrize("activation", ["none", "clamp", "softmax", "softplus", "sigmoid"])
@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("channels,sharing", [(1, "none"), (3, "full")])
def test_training_projects_only_sum_weights_for_none(
    activation, estimated, channels, sharing, monkeypatch, tmp_path
):
    data = torch.tensor([[0, 1], [1, 0]]).repeat(1, channels)
    parameter_activation = "positive-clamp" if activation == "clamp" else activation
    symbolic = image_data(
        (channels, 1, 2),
        input_layer="categorical",
        num_input_units=2,
        num_sum_units=2,
        sum_product_layer="cp",
        input_sharing=sharing,
        data=data,
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
    input_layers = [layer for layer in circuit.layers if isinstance(layer, TorchInputLayer)]
    input_ids = {id(p) for layer in input_layers for p in layer.parameters()}
    assert input_ids and any(id(p) not in input_ids for p in parameters)
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
        for p in parameters:
            if id(p) in input_ids:
                # Negative input logits are valid before and after optimizer updates.
                torch.testing.assert_close(p, torch.full_like(p, -2.0), rtol=0, atol=0)
            elif activation == "none":
                assert (p >= POSITIVE_PARAMETER_MIN).all()
        for layer in input_layers:
            probs = layer.probs()
            torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs.sum(-1)))
        loss = nll(batch)
        assert torch.isfinite(loss)
        return loss

    with torch.enable_grad():
        train_circuit(
            circuit,
            partition,
            [(data,)],
            [(data,)],
            device="cpu",
            config=TrainingConfig(
                max_steps=2,
                learning_rate=0.01,
                scheduler_period=1,
                minimum_learning_rate=0.0,
                weight_decay=0.0,
                save_path=tmp_path / "model.pt",
                validation_steps=1,
                min_delta=0.0,
                patience=3,
                num_dimensions=2 * channels,
                log_to_wandb=False,
                activation=activation,
            ),
            nll_fn=checked_nll,
        )
    for parameter in parameters:
        expected = (
            POSITIVE_PARAMETER_MIN
            if activation == "none" and id(parameter) not in input_ids
            else -2.0
        )
        torch.testing.assert_close(parameter, torch.full_like(parameter, expected), rtol=0, atol=0)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_runner_exposes_sum_activation_choices():
    parser = build_parser()
    assert not parser.parse_args([]).torch_compile
    assert parser.parse_args(["--torch-compile"]).torch_compile
    for activation in ("none", "clamp", "softmax", "softplus", "sigmoid"):
        args = parser.parse_args(["--activation", activation])
        assert args.activation == activation
        assert not hasattr(args, "input_activation")
        assert not hasattr(args, "compile_backend")


@pytest.mark.parametrize("enabled,backend", [(False, "torch"), (True, "torch-compile")])
def test_runner_selects_cirkit_backend(enabled, backend):
    with patch("scripts.evaluate_image_circuit.PipelineContext") as context_cls:
        assert make_pipeline_context(enabled) is context_cls.return_value
    context_cls.assert_called_once_with(
        backend=backend,
        semiring="lse-sum",
        fold=True,
        optimize=True,
    )


@pytest.mark.parametrize(
    "region_graph",
    [
        "quad-tree-2",
        "quad-tree-4",
        "quad-graph",
        "random-binary-tree",
        "poon-domingos",
        "chow-liu-tree",
        "linear-tree",
    ],
)
@pytest.mark.parametrize("channels,sharing,estimated", [(1, "full", False), (3, "none", True)])
def test_runner_trains_and_evaluates_image_region_graphs(
    region_graph, channels, sharing, estimated, monkeypatch, tmp_path, caplog
):
    train = torch.randint(0, 256, (6, channels, 2, 2), dtype=torch.uint8)
    torch.save(train, tmp_path / "train.pt")
    torch.save(torch.zeros(2, channels, 2, 2, dtype=torch.uint8), tmp_path / "valid.pt")
    torch.save(torch.full((2, channels, 2, 2), 255, dtype=torch.uint8), tmp_path / "test.pt")
    checkpoint = tmp_path / "model.pt"
    argv = [
        "evaluate_image_circuit",
        "--dataset",
        "tensor",
        "--root",
        str(tmp_path),
        "--rg",
        region_graph,
        "--input-sharing",
        sharing,
        "--inner-layer",
        "cp-t",
        "--activation",
        "none",
        "--k",
        "2",
        "--batch-size",
        "6",
        "--max-epochs",
        "1",
        "--validation-steps",
        "1",
        "--save-path",
        str(checkpoint),
    ]
    learned_tree = {}
    if region_graph == "chow-liu-tree":
        argv += ["--num-bins", "4", "--mi-chunk-size", "2"]
        chow_liu = data_modalities.ChowLiuTree

        def record_tree_data(**kwargs):
            learned_tree.update(kwargs)
            return chow_liu(**kwargs)

        monkeypatch.setattr(data_modalities, "ChowLiuTree", record_tree_data)
    if estimated:
        argv.append("--use-estimated-weights")
    nll_factory = evaluate_image_circuit.make_nll_function
    input_parameters = {}

    def check_softmax_inputs(circuit, partition, **kwargs):
        input_layers = [layer for layer in circuit.layers if isinstance(layer, TorchInputLayer)]
        for layer in input_layers:
            assert layer.probs is not None and layer.logits is None
            for p in layer.parameters():
                input_parameters[id(p)] = p
        assert any((p < 0).any() for p in input_parameters.values())
        nll = nll_factory(circuit, partition, **kwargs)

        def checked_nll(batch):
            for layer in input_layers:
                probs = layer.probs()
                torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs.sum(-1)))
            assert any((p < 0).any() for p in input_parameters.values())
            return nll(batch)

        return checked_nll

    monkeypatch.setattr(evaluate_image_circuit, "make_nll_function", check_softmax_inputs)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    caplog.set_level(logging.INFO, logger="scripts.evaluate_image_circuit")
    with torch.enable_grad():
        main()

    assert checkpoint.is_file()
    metrics = re.search(r"Test NLL: (\S+) \| bpd: (\S+)", caplog.text)
    assert metrics is not None
    assert all(math.isfinite(float(value)) for value in metrics.groups())
    if region_graph == "chow-liu-tree":
        torch.testing.assert_close(learned_tree["data"], train.flatten(1).long() // 64)
        assert learned_tree["num_categories"] == 4
        assert learned_tree["chunk_size"] == 2


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("--num-bins", "0", "--num-bins must be between 1 and 256"),
        ("--num-bins", "257", "--num-bins must be between 1 and 256"),
        ("--mi-chunk-size", "0", "--mi-chunk-size must be positive"),
        ("--mi-chunk-size", "-1", "--mi-chunk-size must be positive"),
    ],
)
def test_runner_rejects_invalid_chow_liu_options(option, value, message, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["evaluate_image_circuit", "--rg", "chow-liu-tree", option, value]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "region_graph", ["random-binary-tree", "poon-domingos", "chow-liu-tree", "linear-tree"]
)
def test_runner_rejects_unsupported_multichannel_sharing(
    region_graph, monkeypatch, tmp_path, capsys
):
    data = torch.zeros(2, 3, 2, 2, dtype=torch.uint8)
    torch.save(data, tmp_path / "train.pt")
    torch.save(data, tmp_path / "valid.pt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_image_circuit",
            "--dataset",
            "tensor",
            "--root",
            str(tmp_path),
            "--rg",
            region_graph,
            "--input-sharing",
            "full",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "use --input-sharing none with this RG" in capsys.readouterr().err
