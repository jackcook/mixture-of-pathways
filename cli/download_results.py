from pathlib import Path

import click
import inquirer
from modal.volume import FileEntryType
from tqdm import tqdm

from pathways.modal_resources import runs_volume

LOCAL_RUNS_DIR = "runs"


@click.command()
@click.option("--include-all-checkpoints", is_flag=True, default=False)
@click.option("--include-latest-checkpoints", is_flag=True, default=False)
def download_results(  # noqa: PLR0912
    *,
    include_all_checkpoints: bool,
    include_latest_checkpoints: bool,
) -> None:
    """
    Download results from the runs volume on Modal.

    :param: include_all_checkpoints: Whether to download all checkpoints.
    :param: include_latest_checkpoints: Whether to download only the latest checkpoint.
    """

    run_volume_contents = runs_volume.listdir("/")
    run_ids = set()

    for file in run_volume_contents:
        if file.type != FileEntryType.DIRECTORY:
            continue

        run_id = file.path.split(".")[0]
        run_ids.add(run_id)

    # Show 10 most recent runs
    run_ids = sorted(run_ids, reverse=True)

    answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "run_ids",
                message="Select runs to download (↑/↓ to navigate, space to select, "
                "enter to confirm)",
                choices=run_ids,
            ),
        ],
        raise_keyboard_interrupt=True,
    )

    file_paths_to_download = []

    for run_id in answers["run_ids"]:
        run_folders = [
            item
            for item in run_volume_contents
            if run_id in item.path and item.type == FileEntryType.DIRECTORY
        ]

        for folder in run_folders:
            (Path(LOCAL_RUNS_DIR) / folder.path).mkdir(parents=True, exist_ok=True)

            latest_checkpoint_num = -1
            latest_checkpoint_path = None

            for file in runs_volume.listdir(folder.path):
                if file.type != FileEntryType.FILE:
                    continue

                if (
                    not include_all_checkpoints
                    and not include_latest_checkpoints
                    and "tfevents" not in file.path
                ):
                    continue

                if include_latest_checkpoints and file.path.endswith(".pth"):
                    if file.path.endswith("model.pth"):
                        file_paths_to_download.append(file.path)
                    else:
                        checkpoint_num = int(
                            file.path.split("/")[1]
                            .replace("model_ckpt", "")
                            .replace(".pth", ""),
                        )

                    if checkpoint_num > latest_checkpoint_num:
                        latest_checkpoint_num = checkpoint_num
                        latest_checkpoint_path = file.path
                else:
                    file_paths_to_download.append(file.path)

            if latest_checkpoint_path is not None:
                file_paths_to_download.append(latest_checkpoint_path)

    for file_path in tqdm(file_paths_to_download):
        local_path = Path(LOCAL_RUNS_DIR) / file_path

        with local_path.open("wb") as f:
            for chunk in runs_volume.read_file(file_path):
                f.write(chunk)


if __name__ == "__main__":
    download_results()
