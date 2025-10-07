import itertools
from pathlib import Path

import click

from cli.run_experiment import run_experiment
from pathways.task import Task


@click.command()
@click.option(
    "--early-stopping-threshold",
    type=float,
    default=0.25,
    help="Loss threshold at which to stop training.",
)
@click.option(
    "--num-repeats",
    type=int,
    default=5,
    help="Number of times to repeat each task.",
)
@click.option(
    "--task-id",
    type=str,
    default="modcog/all",
    help="Group of tasks to compute difficulty for.",
)
@click.pass_context
def main(
    ctx: click.Context,
    early_stopping_threshold: float,
    num_repeats: int,
    task_id: str,
) -> None:
    """
    Run experiments that can be used to compute the difficulty of all modcog tasks.

    :param: early_stopping_threshold: The loss threshold at which to stop training.
    :param: num_repeats: The number of times to repeat each task.
    :param: task_id: The group of tasks to compute difficulty for.
    """

    env_names = Task.from_id(task_id).env_names

    for repeat, task_name in itertools.product(range(num_repeats), env_names):
        if Path("runs").glob(f"*{task_name}_{repeat}/*"):
            continue

        ctx.invoke(
            run_experiment,
            cost_based_loss_alpha=0,
            disable_fixation_loss=True,
            disable_wandb=True,
            early_stopping_threshold=early_stopping_threshold,
            layers=("64"),
            num_steps=100,
            num_epochs=100,
            run_id=f"{task_name}_{repeat}",
            task_id=f"modcog/{task_name}",
        )


if __name__ == "__main__":
    main()
