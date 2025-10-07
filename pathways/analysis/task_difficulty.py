import math
import statistics
from pathlib import Path

from tbparse import SummaryReader

task_difficulties_cache: dict[str, float] = {}


def get_task_difficulty(
    task_name: str,
    *,
    loss_threshold: float = 0.25,
    return_raw_steps: bool = False,
    runs_dir: str = "runs",
) -> float | list[float]:
    """
    Get the difficulty of a task, defined as the log of the median number of steps to
    reach a specified loss threshold over some number of runs.

    :param: task_name: The name of the task to get the difficulty for.
    :param: loss_threshold: The loss threshold to use to determine the difficulty of
        the task.
    :param: return_raw_steps: Whether to return the raw number of steps to reach the
        loss threshold.
    :param: runs_dir: The directory to look for runs in.
    :return: The difficulty of the task.
    """
    global task_difficulties_cache

    if task_name not in task_difficulties_cache:
        steps = []

        for run_path in Path(runs_dir).glob(f"*_{task_name}_*"):
            tb_path = str(next(run_path.glob("events.out.tfevents.*")))
            reader = SummaryReader(tb_path)

            steps.append(
                reader.scalars[
                    (reader.scalars.tag == "loss/final")
                    & (reader.scalars.value >= loss_threshold)
                ].step.max(),
            )

        if return_raw_steps:
            task_difficulties_cache[task_name] = steps
        else:
            task_difficulties_cache[task_name] = math.log(statistics.median(steps))

    return task_difficulties_cache[task_name]
