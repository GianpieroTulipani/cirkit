"""Check image runner training, parameter activations, and region graph integration."""

import logging
import math
import re
import sys

import pytest
import torch

from cirkit.pipeline import PipelineContext
from cirkit.symbolic import functional
from cirkit.templates import data_modalities
from cirkit.templates.data_modalities import image_data
from cirkit.templates.initialization import DataDrivenInitialization
from cirkit.templates.utils import POSITIVE_PARAMETER_MIN, Parameterization
from scripts.evaluate_image_circuit import (
    _log_cuda_memory,
    build_parser,
    main,
    make_nll_function,
    train_circuit,
)


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_memory_logging_skips_cuda_on_cpu(device, monkeypatch, caplog):
    def unexpected_cuda_call(*args, **kwargs):
        pytest.fail("CPU memory logging must not call CUDA")

    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "memory_allocated",
        "memory_reserved",
        "mem_get_info",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, unexpected_cuda_call)
    caplog.set_level(logging.INFO, logger="scripts.evaluate_image_circuit")
    _log_cuda_memory("CPU training", device, reset_peaks=True)
    assert not caplog.records


@pytest.mark.parametrize("reset_peaks", [False, True])
def test_memory_logging_distinguishes_device_usage_and_allocator_peaks(
    reset_peaks, monkeypatch, caplog
):
    device = torch.device("cuda:1")
    calls = []
    # A temporary allocation has been released: current usage is below its peak.
    peaks = {"max_memory_allocated": 5, "max_memory_reserved": 6}

    def synchronize(actual_device):
        assert actual_device == device
        calls.append("synchronize")

    def reset(actual_device):
        assert actual_device == device
        calls.append("reset")
        peaks.update(max_memory_allocated=2, max_memory_reserved=3)

    def read_gib(value):
        def read(actual_device):
            assert actual_device == device
            return value * 2**30

        return read

    def memory_info(actual_device):
        assert actual_device == device
        return 10 * 2**30, 16 * 2**30

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "memory_allocated", read_gib(2))
    monkeypatch.setattr(torch.cuda, "memory_reserved", read_gib(3))
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_info)
    for name in peaks:
        monkeypatch.setattr(
            torch.cuda, name, lambda actual_device, name=name: read_gib(peaks[name])(actual_device)
        )

    caplog.set_level(logging.INFO, logger="scripts.evaluate_image_circuit")
    _log_cuda_memory("training checkpoint", device, reset_peaks=reset_peaks)

    assert calls == (["synchronize", "reset"] if reset_peaks else ["synchronize"])
    assert "allocated=2.00 GiB | reserved=3.00 GiB | reserved_unused=1.00 GiB" in caplog.text
    assert "device_used=6.00 GiB | device_free=10.00 GiB | device_total=16.00 GiB" in caplog.text
    assert f"peak_allocated={peaks['max_memory_allocated']:.2f} GiB" in caplog.text
    assert f"peak_reserved={peaks['max_memory_reserved']:.2f} GiB" in caplog.text


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
        activation=parameter_activation,
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
        "softmax",
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
