import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from datasets import Array2D, Dataset, DatasetDict, Features, load_from_disk
from datasets.utils.logging import disable_progress_bar

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_dataset_path(name: str, num_samples: int) -> Path:
    """Get the path to a cached dataset for a given task and number of samples."""

    return (
        Path("/data" if Path("/data").exists() else Path.home())
        / ".pathways_cache"
        / f"{''.join(c for c in name if c.isalnum()).rstrip()}_{num_samples}"
    )


def get_dataset(
    name: str,
    task_id: str,
    num_samples: int,
    batch_fn: Callable,
) -> DatasetDict:
    """
    Get a HF dataset for a given task and number of samples. If the dataset
    does not exist, it will be built and cached.

    :param: name: The name of the dataset.
    :param: task_id: The id of the task to get the dataset for.
    :param: num_samples: The number of samples to get from the dataset.
    :param: batch_fn: A function that returns a new batch of data for the dataset.
    :return: A dataset for the given task and number of samples.
    """

    dataset_path = get_dataset_path(name, num_samples)

    if not dataset_path.exists():
        build_dataset(name, task_id, num_samples, batch_fn)
    else:
        disable_progress_bar()

    return load_from_disk(dataset_path).with_format("torch")


def build_dataset(
    name: str,
    task_id: str,
    num_samples: int,
    batch_fn: Callable,
) -> None:
    """
    Build a Hugging Face dataset for a given task and number of samples.

    :param: name: The name of the dataset to build.
    :param: task_id: The id of the task to build the dataset for.
    :param: num_samples: The number of samples that should be in the dataset.
    :param: batch_fn: A function that returns a new batch of data for the dataset.
    """

    dataset_path = get_dataset_path(name, num_samples)
    logger.info("Building dataset...")

    def gen() -> Iterator[dict[str, np.ndarray]]:
        dataset_size = 0

        while dataset_size < num_samples:
            inputs, labels = batch_fn()

            # Fix fixation periods, some tasks (e.g. modcog/dm1) have fixation
            # periods that seem to have a little bit of noise
            inputs[:, :, 0] = np.where(inputs[:, :, 0] > 0, 1, 0)

            for j in range(inputs.shape[1]):
                yield {
                    "inputs": inputs[:, j, :],
                    "labels": np.expand_dims(labels[:, j], axis=1),
                }

            dataset_size += inputs.shape[1]

    # modcog/all has an extra 82 dimensions that act as a one-hot encoding of
    # the current environment
    input_dim = (
        115 if task_id == "modcog/all" else (49 if task_id == "modcog/16" else 33)
    )

    if task_id.startswith("modcog"):
        seq_len = 350
    elif task_id.startswith("neurogym"):
        seq_len = 100
    else:
        msg = f"Unknown task: {task_id}"
        raise ValueError(msg)

    Dataset.from_generator(
        gen,
        features=Features(
            {
                "inputs": Array2D(shape=(seq_len, input_dim), dtype="float32"),
                "labels": Array2D(shape=(seq_len, 1), dtype="int64"),
            },
        ),
    ).save_to_disk(dataset_path)
