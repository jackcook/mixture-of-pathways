import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from pathways.task import Task


@click.command()
@click.option(
    "--batch-size",
    type=int,
    default=128,
    help="Number of samples in each training batch.",
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a checkpoint directory to start training from. If not provided, a "
    "new run will be started.",
)
@click.option(
    "--cost-based-loss-alpha",
    type=float,
    default=1e-5,
    help="Tuning parameter to trade off between routing and action losses.",
)
@click.option(
    "--cost-based-loss-epsilon",
    type=float,
    default=1e-2,
    help="Small nonzero value to prevent division by zero in routing loss.",
)
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default="cuda")
@click.option(
    "--disable-fixation-loss",
    is_flag=True,
    default=False,
    help="Trains the model with task performance losses only, ignoring outputs during "
    "the fixation period.",
)
@click.option(
    "--disable-task-embedding-layer",
    is_flag=True,
    default=False,
    help="Trains the model without the task embedding layer.",
)
@click.option(
    "--disable-task-performance-scaling",
    is_flag=True,
    default=False,
    help="Disables the scaling of routing cost by task performance.",
)
@click.option(
    "--disable-wandb",
    is_flag=True,
    default=False,
    help="Disables logging of metrics to Weights & Biases.",
)
@click.option(
    "--dropout-max-prob",
    type=float,
    default=0.8,
    help="Dropout probability for experts with a routing weight of zero.",
)
@click.option(
    "--dropout-router-weight-threshold",
    type=float,
    default=0.1,
    help="Maximum routing weight at which experts will be dropped out. Dropout "
    "probability linearly decreases from dropout_max_prob to zero as the routing "
    "weight increases from zero to this threshold.",
)
@click.option(
    "--early-stopping-threshold",
    type=float,
    default=None,
    help="If loss is below this threshold, training will stop early.",
)
@click.option(
    "--ephemeral",
    is_flag=True,
    default=False,
    help="Train without saving any checkpoints or metrics.",
)
@click.option(
    "--expert-cost-exponent",
    type=float,
    default=2,
    help="Exponent to raise the expert size to when computing the routing cost.",
)
@click.option(
    "--intermediate-dim",
    type=int,
    default=64,
    help="Output dimension of all experts in all layers.",
)
@click.option(
    "--layers",
    "-l",
    type=str,
    multiple=True,
    default=("0,16,32", "0,16,32", "0,16,32"),
    help="Comma-separated lists of expert sizes for each layer. 0 denotes a skip "
    "connection.",
)
@click.option("--learning-rate", type=float, default=1e-2)
@click.option("--modal", is_flag=True, default=False, help="Run experiments on Modal.")
@click.option(
    "--num-epochs",
    type=int,
    default=10,
    help="Number of epochs to train for.",
)
@click.option(
    "--num-runs",
    type=int,
    default=1,
    help="Number of training runs to start. If --modal is provided, runs will be "
    "started in parallel on separate containers. Otherwise, runs will be started "
    "sequentially on the local machine.",
)
@click.option(
    "--num-steps",
    type=int,
    default=1000,
    help="Number of steps in each training epoch.",
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help="Train with the PyTorch profiler turned on. Traces can get large very "
    "quickly, so consider running this with --num-epochs 1 --num-steps 10.",
)
@click.option(
    "--router-dim",
    type=int,
    default=64,
    help="Hidden size of the router's GRU.",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Run ID to use for the experiment. If not provided, a run ID will be "
    "generated automatically.",
)
@click.option("--task-dim", type=int, default=16, help="Dimension of task embeddings.")
@click.option(
    "--task-id",
    type=str,
    default="modcog/all",
    help="Task to train the model on. modcog/all is the full set of 82 tasks, "
    "modcog/16 is a small set of 16 tasks that can be used for testing, and "
    "modcog/[name] (e.g. modcog/go, modcog/dm1) will specify a single task.",
)
def run_experiment(**kwargs: dict[str, Any]) -> None:
    """Run an experiment from the command line."""

    # Disable AMP if device is not CUDA
    if kwargs["device"] != "cuda":
        os.environ["MOP_DISABLE_AMP"] = "TRUE"

    # Ensure checkpoint exists
    if kwargs["checkpoint"] is not None and not Path(kwargs["checkpoint"]).exists():
        msg = f"Checkpoint {kwargs['checkpoint']} does not exist"
        raise FileNotFoundError(msg)

    # Set task-specific parameters
    task = Task.from_id(kwargs["task_id"])
    kwargs["input_dim"] = int(task.input_size)
    kwargs["output_dim"] = int(task.output_size)

    # Validate layer structures
    for layer in kwargs["layers"]:
        for expert_size in layer.split(","):
            if int(expert_size) < 0:
                msg = f"Expert size {expert_size} is negative"
                raise ValueError(msg)

    # Set default run ID
    run_id = kwargs["run_id"] or "_".join(
        layer_desc.replace(",", "-") for layer_desc in kwargs["layers"]
    )

    if not re.match(r"^\d{8}_\d{6}.*", run_id):
        run_id = f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{run_id}"

    # Create experiments
    enable_profiler = kwargs.pop("profile")
    num_runs = kwargs.pop("num_runs")

    if kwargs.pop("modal"):
        # Run on Modal
        import modal

        from pathways.modal_resources import app
        from pathways.modal_runner import run_experiment

        with modal.enable_output(), app.run(detach=True):
            for i in range(num_runs):
                kwargs["run_id"] = f"{run_id}.{i}" if num_runs > 1 else run_id
                run_experiment.spawn(**kwargs)
                print("Running experiment with run ID:", kwargs["run_id"])
    else:
        # Run locally
        from pathways.config import Config
        from pathways.experiment import Experiment

        for i in range(num_runs):
            kwargs["run_id"] = f"{run_id}.{i}" if num_runs > 1 else run_id
            config = Config(**kwargs)
            experiment = Experiment(config, task)

            if enable_profiler:
                import torch

                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as prof:
                    experiment.train()

                prof.export_chrome_trace("trace.json")
            else:
                experiment.train()

            print("Finished experiment with run ID:", kwargs["run_id"])


if __name__ == "__main__":
    run_experiment()
