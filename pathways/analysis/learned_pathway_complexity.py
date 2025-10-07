from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from pathways.model import Model
from pathways.task import Task


def get_learned_pathway_complexities(
    model_or_pattern: str | Model,
    *,
    checkpoint: int | None = None,
    disable_progress_bar: bool = False,
    inference_disable_complex_experts: bool = False,
    inference_dropout_threshold: float | None = None,
    num_eval_batches: int = 100,
    pattern_max: int | None = None,
    runs_dir: str | None = None,
) -> pd.DataFrame:
    """
    Get the learned pathway complexities for a single model or a list of models.

    :param: model_or_pattern: The model or pattern of models to use.
    :param: checkpoint: The checkpoint of the model to use.
    :param: disable_progress_bar: Whether to disable the progress bar.
    :param: inference_disable_complex_experts: Whether to disable complex experts during
        inference.
    :param: inference_dropout_threshold: The dropout threshold to use during inference.
    :param: num_eval_batches: The number of batches to evaluate on.
    :param: pattern_max: The maximum number of models to use.
    :param: runs_dir: The directory to look for runs in.
    :return: A dataframe with the learned pathway complexities for each model and task.
    """

    task = None
    dataloader = None

    def get_single_model_lpcs(model: Model) -> pd.DataFrame:
        """
        Get the learned pathway complexities for a single model.

        :param: model: The model to use.
        :return: A dataframe with the learned pathway complexities for each task.
        """

        nonlocal task, dataloader

        if task is None:
            task = Task.from_id(model.config.task_id)

        if dataloader is None:
            dataloader = task.get_dataloader(
                num_samples=model.config.num_steps * model.config.batch_size,
                batch_size=model.config.batch_size,
                shuffle=True,
            )

        task_lpcs = {}
        task_perf = {}

        for batch_i, batch in enumerate(dataloader):
            if batch_i >= num_eval_batches:
                break

            inputs, labels = (
                batch["inputs"].to(model.device),
                batch["labels"].to(model.device).flatten(),
            )

            task_ids = inputs[:, :, 33:].argmax(dim=-1)
            fixation = inputs[:, :, 0]
            during_response_mask = fixation == 0

            action_pred, _, _, expert_usages = model(
                inputs,
                output_all_pathways=True,
                inference=True,
                inference_disable_complex_experts=inference_disable_complex_experts,
                inference_dropout_threshold=inference_dropout_threshold,
            )

            for i, task_name in enumerate(task.env_names):
                if task_name not in task_lpcs:
                    task_lpcs[task_name] = []

                complexity = 0
                task_i_expert_usages = [
                    x[((task_ids == i) & (fixation == 0)).cpu()] for x in expert_usages
                ]

                for j, router_output in enumerate(task_i_expert_usages):
                    layer_costs = [
                        int(size) ** model.config.expert_cost_exponent
                        for size in model.config.layers[j].split(",")
                    ]
                    complexity += (
                        np.einsum("ij,j->i", router_output, layer_costs).sum()
                        / router_output.shape[0]
                    )

                task_lpcs[task_name].append(complexity)

        for k, task_lpc_values in task_lpcs.items():
            task_lpcs[k] = sum(task_lpc_values) / len(task_lpc_values)
            fixation = inputs[:, :, 0]
            during_response_mask = fixation == 0

            for i, task_name in enumerate(task.env_names):
                task_perf[task_name] = (
                    (
                        labels[(during_response_mask & (task_ids == i)).cpu().flatten()]
                        == action_pred[
                            (during_response_mask & (task_ids == i)).cpu()
                        ].argmax(dim=-1)
                    )
                    .float()
                    .mean()
                    .item()
                )

        model_df = pd.DataFrame(
            [
                {
                    "task": task_name,
                    "lpc": task_lpcs[task_name],
                    "accuracy": task_perf[task_name],
                }
                for task_name in task.env_names
            ],
        )

        model_df["lpc_rank"] = model_df.sort_values(by="lpc", ascending=True).index
        return model_df

    if isinstance(model_or_pattern, str):
        model_or_pattern = sorted(Path(runs_dir).glob(model_or_pattern))

        if pattern_max is not None:
            model_or_pattern = [
                x.as_posix()
                for x in model_or_pattern
                if int(x.as_posix().split(".")[-1]) <= pattern_max
            ]
    else:
        model_or_pattern = [model_or_pattern]

    dfs = []

    for model_id in tqdm(model_or_pattern, disable=disable_progress_bar):
        model = Model.from_run(
            model_id.split("/")[-1],
            checkpoint=checkpoint,
            runs_dir=runs_dir,
        )
        model.eval()

        model_df = get_single_model_lpcs(model)
        model_df["model_id"] = model_id
        dfs.append(model_df)

    return pd.concat(dfs)
