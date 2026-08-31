# MNIST differential regression probe

This harness compares the pre-RGB MNIST implementation at Git revision `83fc3c5` with the
current working tree. Each implementation runs in a fresh subprocess and the baseline is exported
to a temporary directory with `git archive`, so no historical source file is copied into the
current package and the repository metadata is not modified.

The comparison covers:

- raw MNIST data and deterministic train/validation/structure split hashes;
- symbolic circuit shape and parameter count;
- compiled circuit/partition parameter sharing;
- NLL, BPD and log-partition trajectories on a fixed test subset.

## Kaggle setup

Run these cells from the directory containing the cloned Cirkit repository:

```bash
cd cirkit
pip install -e . fast-pytorch-kmeans loguru torchvision
```

First validate the harness cheaply. This does not download MNIST:

```bash
python scripts/diagnostics/mnist_differential.py \
  --dataset synthetic \
  --k 8 \
  --steps 2 \
  --eval-every 1 \
  --structure-samples 64 \
  --estimated-mode both \
  --device cpu \
  --no-fail-on-diff
```

Then run the practical MNIST localization test. It compares random and data-estimated
initializations while keeping the run short enough for iteration:

```bash
python scripts/diagnostics/mnist_differential.py \
  --dataset mnist \
  --dataset-root datasets \
  --k 32 \
  --steps 250 \
  --eval-every 25 \
  --structure-samples 4096 \
  --estimated-mode both \
  --device cuda \
  --estimation-device cpu \
  --no-fail-on-diff
```

Finally, this is the closest differential counterpart of the MNIST command in
`replicating-pc-sota.ipynb`. A structure sample count of zero means the entire training split.

```bash
python scripts/diagnostics/mnist_differential.py \
  --dataset mnist \
  --dataset-root datasets \
  --k 256 \
  --steps 12000 \
  --eval-every 250 \
  --structure-samples 0 \
  --estimated-mode on \
  --device cuda \
  --estimation-device cpu \
  --bpd-atol 0.01 \
  --no-fail-on-diff
```

Results are written to `mnist-differential-results/`. `summary.json` contains the comparison;
the per-revision JSON files contain all data, model and trajectory diagnostics. Remove
`--no-fail-on-diff` when the command should return a non-zero exit code on a regression.

The long probe follows a fixed training trajectory; it deliberately does not implement validation
checkpointing or early stopping. Its purpose is to locate the first numerical difference between
the implementations, after which `dataset_eval.py` can be used for the full benchmark.

## Reading the result

- A data hash mismatch means the inputs or deterministic split are not equivalent.
- A model check mismatch before step 0 points to circuit construction, compilation, parameter
  count, or circuit/partition sharing.
- A BPD difference at step 0 points to initialization, forward evaluation, or normalization.
- Equal step-0 values that separate after training point to gradients, optimizer-visible
  parameters, clamping, or partition synchronization.
- A difference only in `estimated_init` localizes the problem to LearnSPN parameter estimation.

The generic evaluator stores its dataset as `uint8`, which is therefore the probe default. If that
run differs, repeat it with `--data-dtype long`. A difference that disappears in the second run
isolates the regression to the dtype/device boundary introduced by the generic loader.
