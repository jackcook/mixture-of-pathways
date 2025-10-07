import click
import inquirer
from modal.volume import FileEntryType

from pathways.modal_resources import runs_volume


@click.command()
def delete_results() -> None:
    """Delete results from the runs volume on Modal."""

    run_volume_contents = runs_volume.listdir("/")
    run_ids = set()

    for file in run_volume_contents:
        if file.type != FileEntryType.DIRECTORY:
            continue

        run_id = file.path.split(".")[0]
        run_ids.add(run_id)

    run_ids = sorted(run_ids, reverse=True)

    answers = inquirer.prompt(
        [
            inquirer.Checkbox(
                "run_ids",
                message="Select runs to delete (↑/↓ to navigate, space to select, "
                "enter to confirm)",
                choices=run_ids,
            ),
        ],
        raise_keyboard_interrupt=True,
    )

    for run_id in answers["run_ids"]:
        run_folders = [
            item
            for item in run_volume_contents
            if run_id in item.path and item.type == FileEntryType.DIRECTORY
        ]

        for folder in run_folders:
            runs_volume.remove_file(folder.path, recursive=True)


if __name__ == "__main__":
    delete_results()
