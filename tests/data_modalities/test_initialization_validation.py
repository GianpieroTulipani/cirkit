"""Check categorical observations before Chow-Liu structure learning starts."""

from unittest.mock import patch

import pytest
import torch

from cirkit.templates.data_modalities import image_data, tabular_data
from cirkit.templates.initialization import DataDrivenInitialization


@pytest.mark.parametrize("modality", ["image", "tabular"])
@pytest.mark.parametrize(
    "data",
    [
        torch.tensor([[-1, 0], [0, 1]]),
        torch.tensor([[256, 0], [0, 1]]),
        torch.tensor([[0.5, 0], [0, 1]]),
        torch.tensor([[float("nan"), 0], [0, 1]]),
        torch.tensor([[float("inf"), 0], [0, 1]]),
        torch.ones(2, 2, dtype=torch.complex64),
        torch.empty(0, 2),
    ],
)
def test_invalid_initialization_data_precedes_chow_liu(modality, data):
    kwargs = {
        "region_graph": "chow-liu-tree",
        "data": data,
        "num_input_units": 2,
        "num_sum_units": 2,
        "sum_product_layer": "cp",
        "initialization": DataDrivenInitialization(),
    }
    with patch("cirkit.templates.data_modalities.ChowLiuTree") as learn_tree:
        with pytest.raises(ValueError):
            if modality == "image":
                image_data((1, 1, 2), input_layer="categorical", num_bins=8, **kwargs)
            else:
                tabular_data(
                    input_layers={"name": "categorical", "args": {"num_categories": 256}},
                    **kwargs,
                )
        learn_tree.assert_not_called()


@pytest.mark.parametrize(
    "data,counts",
    [
        (torch.tensor([[2, 0], [0, 1]]), [2, 3]),
        (torch.tensor([[0, 3], [1, 0]]), [2, 3]),
        (torch.tensor([[0, 1], [1, 0]]), [2]),
    ],
)
def test_feature_categories_are_checked_before_chow_liu(data, counts):
    with patch("cirkit.templates.data_modalities.ChowLiuTree") as learn_tree:
        with pytest.raises(ValueError, match="Observations in feature|Number of input layers"):
            tabular_data(
                region_graph="chow-liu-tree",
                data=data,
                input_layers=[
                    {"name": "categorical", "args": {"num_categories": count}} for count in counts
                ],
                num_input_units=2,
                num_sum_units=2,
                sum_product_layer="cp",
                initialization=DataDrivenInitialization(),
            )
        learn_tree.assert_not_called()


@pytest.mark.parametrize("modality", ["image", "tabular"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.float32])
def test_valid_integer_observations_reach_chow_liu(modality, dtype):
    kwargs = {
        "region_graph": "chow-liu-tree",
        "data": torch.tensor([[0, 0], [1, 255], [0, 255], [1, 0]], dtype=dtype),
        "num_input_units": 2,
        "num_sum_units": 2,
        "sum_product_layer": "cp",
        "initialization": DataDrivenInitialization(),
    }
    if modality == "image":
        circuit = image_data((1, 1, 2), input_layer="categorical", num_bins=8, **kwargs)
    else:
        circuit = tabular_data(
            input_layers={"name": "categorical", "args": {"num_categories": 256}},
            **kwargs,
        )
    assert set(circuit.scope) == {0, 1}
    assert circuit.is_smooth and circuit.is_decomposable
