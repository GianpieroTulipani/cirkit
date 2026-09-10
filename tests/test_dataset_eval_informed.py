import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("estimated", [False, True])
def test_informed_eval_cli_trains_saves_and_evaluates(tmp_path: Path, estimated: bool) -> None:
    """Exercise the actual CLI with uint8 data, both initializers and both tensor layouts."""
    data = torch.tensor([[0, 0, 0, 0], [255, 255, 255, 255]] * 4, dtype=torch.uint8)
    if estimated:
        data = data.reshape(-1, 1, 2, 2)
        torch.save(data[:4], tmp_path / "valid.pt")
    torch.save(data, tmp_path / "train.pt")
    torch.save(data[:4], tmp_path / "test.pt")
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    command = [
        sys.executable,
        "-m",
        "cirkit.dataset_eval_informed",
        "--dataset",
        "tensor",
        "--root",
        str(tmp_path),
        "--image-shape",
        "1",
        "2",
        "2",
        "--valid-split",
        "0.25",
        "--k",
        "3",
        "--num-input-units",
        "2",
        "--batch-size",
        "4",
        "--max-epochs",
        "1",
        "--validation-steps",
        "1",
        "--structure-samples",
        "6",
        "--min-instances",
        "2",
        "--min-cluster-size",
        "2",
        "--max-depth",
        "2",
        "--local-radius",
        "1",
        "--mi-alpha",
        "0",
        "--mi-threshold",
        "0.1",
        "--pair-batch-size",
        "2",
        "--sample-batch-size",
        "3",
        "--estimation-device",
        "cpu",
        "--noise-scale",
        "0.1",
        "--alpha",
        "0.5",
        "--save-path",
        str(checkpoint),
    ]
    command += (
        ["--use-estimated-weights", "--adaptive-alpha"]
        if estimated
        else ["--activation", "softmax", "--use-mixing-weights", "--no-force-root-mixture"]
    )
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Using InformedLearnSPN" in output
    assert "Using 6 samples" in output
    assert "New best model" in output
    assert "Test NLL:" in output
    assert checkpoint.is_file()
    parameters = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert parameters and all(torch.isfinite(value).all() for value in parameters.values())
