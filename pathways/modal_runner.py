import subprocess
from pathlib import Path
from typing import Any

import modal

from .modal_resources import app, runs_volume, training_data_volume, wandb_secret

TIMEOUT = 24 * 60 * 60  # 1 day

experiment_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_pyproject(Path(__file__).parent.parent / "pyproject.toml")
)

with experiment_image.imports():
    from .config import Config
    from .experiment import Experiment
    from .task import Task

tensorboard_image = modal.Image.debian_slim().pip_install("tensorboard")


@app.function(
    image=experiment_image,
    secrets=[wandb_secret],
    gpu="T4",
    volumes={"/data": training_data_volume, "/runs": runs_volume},
    timeout=TIMEOUT,
)
def run_experiment(**kwargs: dict[str, Any]) -> None:
    """Run an experiment on Modal."""

    # Fetch latest model checkpoints
    runs_volume.reload()

    task = Task.from_id(kwargs["task_id"])
    config = Config(**kwargs)
    experiment = Experiment(config, task)
    experiment.train()


@app.function(
    image=tensorboard_image,
    volumes={"/runs": runs_volume},
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=6006)
def tensorboard() -> None:
    """Run TensorBoard on Modal."""
    subprocess.Popen(["tensorboard", "--logdir", "/runs", "--bind_all"])
