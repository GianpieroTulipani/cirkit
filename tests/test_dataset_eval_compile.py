"""Exercise the optional compiled loss through training and checkpoint reload."""

import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

import cirkit.symbolic.functional as sf
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.dataset_eval import evaluate_circuit, make_nll_function, train_circuit
from tests.symbolic.test_utils import build_monotonic_structured_categorical_cpt_pc


def build_circuit():
    symbolic = build_monotonic_structured_categorical_cpt_pc()
    compiler = TorchCompiler(fold=True, optimize=True, semiring="lse-sum")
    return compiler.compile(symbolic), compiler.compile(sf.integrate(symbolic))


@torch.enable_grad()
def check_compiled_training_validation_and_checkpoint_reload(tmp_path):
    torch.compiler.reset()
    eager, eager_partition = build_circuit()
    compiled_model, compiled_partition = build_circuit()
    initial = {key: value.clone() for key, value in eager.state_dict().items()}
    compiled_model.load_state_dict(initial)
    loss_fn = make_nll_function(
        compiled_model, compiled_partition,
        use_compile=True, backend="aot_eager", fullgraph=True,
    )
    data = torch.randint(0, 2, (5, eager.num_variables))
    loader = DataLoader(TensorDataset(data), batch_size=3, shuffle=False)
    eager_loss = make_nll_function(eager, eager_partition)
    expected = eager_loss(data[:3])
    actual = loss_fn(data[:3])
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected, tuple(eager.parameters()), allow_unused=True)
    actual_grads = torch.autograd.grad(actual, tuple(compiled_model.parameters()), allow_unused=True)
    for ref, got in zip(expected_grads, actual_grads):
        if ref is None:
            assert got is None
        else:
            torch.testing.assert_close(got, ref)

    for name, model, partition, nll in (
        ("eager", eager, eager_partition, None),
        ("compiled", compiled_model, compiled_partition, loss_fn),
    ):
        train_circuit(
            model, partition, loader, loader,
            sum_params=[p for layer in model.layers if not isinstance(layer, TorchInputLayer)
                        for p in layer.parameters()],
            max_train_steps=3, lr=0.001, T_0=1, eta_min=0.0, weight_decay=0.0,
            device=torch.device("cpu"), save_path=tmp_path / f"{name}.pt",
            validation_steps=1, delta=0.0, patience=10,
            num_dimensions=model.num_variables, log_to_wandb=False,
            activation="clamp", nll_fn=nll,
        )

    assert any(not torch.equal(value, initial[key]) for key, value in eager.state_dict().items())
    for key, value in eager.state_dict().items():
        torch.testing.assert_close(compiled_model.state_dict()[key], value)
    eager_saved = torch.load(tmp_path / "eager.pt", weights_only=True)
    compiled_saved = torch.load(tmp_path / "compiled.pt", weights_only=True)
    assert compiled_saved.keys() == initial.keys()  # No _orig_mod checkpoint prefixes.
    for key in eager_saved:
        torch.testing.assert_close(compiled_saved[key], eager_saved[key])

    # Reuse the compiled callable after restoring parameters in the original module.
    compiled_model.load_state_dict(initial)
    eager.load_state_dict(initial)
    with torch.inference_mode():
        torch.testing.assert_close(loss_fn(data[:3]), eager_loss(data[:3]))
    compiled_model.load_state_dict(compiled_saved)
    eager.load_state_dict(eager_saved)
    calls = []

    def checked_test_loss(batch):
        loss = loss_fn(batch)
        torch.testing.assert_close(loss, eager_loss(batch))
        calls.append(len(batch))
        return loss

    evaluate_circuit(
        compiled_model, compiled_partition, loader, device=torch.device("cpu"),
        num_dimensions=compiled_model.num_variables, log_to_wandb=False,
        nll_fn=checked_test_loss,
    )
    assert calls == [3, 2]
    torch.compiler.reset()


class TestDatasetEvalCompile(unittest.TestCase):
    def test_training_validation_and_checkpoint_reload(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            check_compiled_training_validation_and_checkpoint_reload(Path(directory))

    def test_compile_mode_requires_inductor(self):
        with self.assertRaisesRegex(ValueError, "inductor"):
            make_nll_function(
                None, None, use_compile=True, backend="aot_eager", mode="max-autotune"
            )


if __name__ == "__main__":
    unittest.main()
