# Mixture of Pathways

_Studying brain-like processing pathways with heterogeneous mixture-of-experts models. 🧠_

## Setup

### Install dependencies

```bash
pip install -e .
```

## Model training

To train one of our standard multi-task models on all tasks in the [Mod-Cog task suite](https://github.com/mikailkhona/Mod_Cog), you may simply run our `run_experiment` script:

```bash
python -m cli.run_experiment
```

The command above will set your PyTorch device to `cuda` by default.
If you don't have access to a machine with an NVIDIA GPU, you have three options:

```bash
# Train using CPU
python -m cli.run_experiment --device cpu

# Train using MPS on macOS (experimental support)
python -m cli.run_experiment --device mps

# Train on Modal (https://modal.com)
python -m cli.run_experiment --modal
```

### Single-task models

To train a model on a single task, you must specify the task name.
For example:

```bash
python -m cli.run_experiment --task-id modcog/dm1
```

For a full list of task names, see Figure 14 in [our paper](https://arxiv.org/pdf/2506.02813), or [pathways/task.py](pathways/task.py#L52).

### Other model options

All hyperparameters are documented if you run:

```bash
python -m cli.run_experiment --help
```

A couple notable ones are:

- `--cost-based-loss-alpha`: $\alpha$ from Equation 2 in [our paper](https://arxiv.org/pdf/2506.02813).
- `--dropout-max-prob`: $\beta$ from Equation 3 in [our paper](https://arxiv.org/pdf/2506.02813).
- `--dropout-router-weight-threshold`: $\gamma$ from Equation 3 in [our paper](https://arxiv.org/pdf/2506.02813).
- `--layers` (`-l`): Comma-separated lists of expert sizes for each layer, with 0 denoting a skip connection. Should be passed multiple times to specify multiple layers. For example, `-l 0,32,64 -l 0,32,64 -l 0,32,64` is our default model with three layers, each of which has one skip connection, one expert with 32 hidden units, and one expert with 64 hidden units.
- `--num-epochs`: Number of epochs to train for. Defaults to 10.

More documentation will be provided soon!
If you have questions in the meantime, feel free to file an issue or get in touch directly.

## Lint

To make sure any code changes are compliant with our linting rules, please run `ruff` before you commit:

```bash
ruff check
```

## License

`pathways` is available under the CC-BY-4.0 license.
See the [LICENSE](/LICENSE.md) file for more details.
