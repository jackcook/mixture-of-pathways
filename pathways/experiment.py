import json
import logging
import os
import warnings
from pathlib import Path

import torch
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from .config import Config
from .model import Model
from .task import Task
from .utils import is_wandb_configured

MOP_DISABLE_AMP = os.getenv("MOP_DISABLE_AMP", "FALSE") == "TRUE"
RUNS_PATH = Path("/runs") if Path("/runs").exists() else Path("runs")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Experiment:
    """An experiment that can be used to train or evaluate a model on a task."""

    def __init__(
        self,
        model_or_config: Config | Model,
        task: Task,
    ) -> None:
        """
        Initialize an experiment.

        :param: model_or_config: The model or configuration to use for the experiment.
        :param: task: The task to train or evaluate the model on.
        """

        if isinstance(model_or_config, Model):
            self.model = model_or_config
            self.config = self.model.config
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device(model_or_config.device)
            self.config = model_or_config
            self.model = Model(model_or_config)

        self.task = task
        self.optimizer = AdamWScheduleFree(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        if is_wandb_configured() and not self.config.disable_wandb:
            import wandb

            self.run = wandb.init(
                project="pathways",
                id=self.config.run_id,
                config=self.config.to_dict(),
            )

        self.writer = SummaryWriter(RUNS_PATH / self.config.run_id)

    def save_metrics(self, data: dict, *, step: int) -> None:
        """Save metrics to the tensorboard writer and to wandb if it is configured."""

        if self.config.ephemeral:
            return

        if hasattr(self, "run"):
            self.run.log(data)

        for name, value in data.items():
            self.writer.add_scalar(name, value, step)

    def train(self) -> None:  # noqa: PLR0912, PLR0915
        """Train the model."""

        checkpoint_path = (
            Path(self.config.checkpoint)
            if self.config.checkpoint is not None
            else RUNS_PATH / self.config.run_id / "model.pth"
        )

        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, weights_only=False)
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.start_epoch = state_dict["epoch"]
        else:
            self.start_epoch = 0
            (RUNS_PATH / self.config.run_id).mkdir(parents=True, exist_ok=True)
            self.save(epoch=0)

        if self.start_epoch >= self.config.num_epochs:
            return

        # Start training
        before_response_criterion = nn.MSELoss()
        during_response_criterion = nn.CrossEntropyLoss()

        scaler = torch.GradScaler(enabled=not MOP_DISABLE_AMP)
        dataloader = self.task.get_dataloader(
            num_samples=self.config.num_steps * self.config.batch_size,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        for epoch in range(self.start_epoch, self.config.num_epochs):
            logger.info("Starting epoch %d", epoch)
            self.model.train()
            self.optimizer.train()
            stop_early = False

            with tqdm(total=self.config.num_steps) as pbar:
                for step, batch in enumerate(dataloader):
                    self.optimizer.zero_grad()

                    inputs, labels = (
                        batch["inputs"].to(self.device, dtype=torch.float16),
                        batch["labels"].to(self.device).flatten(),
                    )

                    with torch.autocast(
                        device_type=self.config.device,
                        dtype=torch.float16,
                        enabled=not MOP_DISABLE_AMP,
                    ):
                        # forward + backward + optimize
                        (outputs, task_ids, task_expert_usage_losses) = self.model(
                            inputs,
                        )

                        # Calculate separate losses for model outputs before and during
                        # response period
                        fixation = inputs[:, :, 0]
                        during_response_mask = fixation == 0

                        before_response_outputs = outputs[~during_response_mask]
                        before_response_labels = torch.zeros_like(
                            before_response_outputs,
                        )
                        before_response_labels[:, 0] = 1

                        before_response_loss = before_response_criterion(
                            before_response_outputs,
                            before_response_labels,
                        )

                        task_action_losses = {}

                        for i in range(self.model.num_tasks):
                            task_mask = task_ids == i
                            task_action_losses[i] = during_response_criterion(
                                outputs[during_response_mask & task_mask],
                                labels[(during_response_mask & task_mask).flatten()],
                            )

                        if self.config.disable_fixation_loss:
                            loss = torch.tensor(0.0, device=self.device)
                        else:
                            loss = before_response_loss

                        # Cost-based routing loss
                        for k in task_expert_usage_losses:
                            if torch.isnan(task_action_losses[k]):
                                warnings.warn(
                                    f"Task action loss for task {k} is NaN. Consider "
                                    "increasing your batch size to ensure samples from "
                                    "all tasks are included in each batch.",
                                    stacklevel=2,
                                )
                                continue

                            loss = loss + task_action_losses[k]

                            if self.config.cost_based_loss_alpha > 0:
                                task_loss_numerator = (
                                    self.config.cost_based_loss_alpha
                                    * task_expert_usage_losses[k]
                                )
                                task_loss_denominator = (
                                    self.config.cost_based_loss_epsilon
                                    + task_action_losses[k]
                                )

                                if self.config.disable_task_performance_scaling:
                                    loss = loss + task_loss_numerator
                                else:
                                    loss = (
                                        loss
                                        + task_loss_numerator / task_loss_denominator
                                    )

                    routing_losses = {
                        f"loss/routing_cost/{self.task.env_names[k]}": loss.item()
                        for k, loss in task_expert_usage_losses.items()
                    }

                    task_action_losses = {
                        f"loss/action/{self.task.env_names[k]}": task_action_losses[
                            k
                        ].item()
                        for k in task_action_losses
                    }

                    self.save_metrics(
                        {
                            "loss/before_response": before_response_loss.item(),
                            "loss/during_response": sum(task_action_losses.values()),
                            "loss/final": loss.item(),
                            "loss/routing_cost": sum(
                                task_expert_usage_losses.values(),
                            ),
                            **routing_losses,
                            **task_action_losses,
                        },
                        step=epoch * self.config.num_steps + step,
                    )

                    pbar.set_description(f"Loss: {loss.item():.2f}")
                    pbar.update(1)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if (
                        self.config.early_stopping_threshold is not None
                        and loss.item() < self.config.early_stopping_threshold
                    ):
                        stop_early = True
                        break

            self.test(epoch=epoch + 1, log_results=True)
            self.save(epoch=epoch + 1)

            if stop_early:
                break

    def test(
        self,
        *,
        epoch: int | None = None,
        log_results: bool = False,
        dropout_threshold: float | None = None,
    ) -> float:
        """
        Evaluate the model on the test set.

        :param: epoch: The current epoch.
        :param: log_results: Whether to log the results.
        :param: dropout_threshold: The threshold for the dropout probability.
        :return: The average performance over the test set.
        """

        self.model.eval()
        self.optimizer.eval()

        perf = 0

        dataloader = self.task.get_dataloader(
            num_samples=self.config.num_steps * self.config.batch_size,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        num_eval_batches = 50

        for i, batch in enumerate(dataloader):
            if i >= num_eval_batches:
                break

            inputs, labels = (
                batch["inputs"].to(self.device, dtype=torch.float16),
                batch["labels"].to(self.device).flatten(),
            )

            action_pred, _, _ = self.model(
                inputs,
                inference=epoch is None,
                inference_dropout_threshold=dropout_threshold,
            )

            fixation = inputs[:, :, 0]
            during_response_mask = fixation == 0
            during_response_outputs = action_pred[during_response_mask].argmax(dim=-1)
            during_response_perf = (
                labels[during_response_mask.flatten()] == during_response_outputs
            )

            perf += during_response_perf.sum().item() / during_response_perf.nelement()

        perf /= num_eval_batches

        if epoch is not None:
            self.save_metrics(
                {"test/perf": perf},
                step=epoch * self.config.num_steps,
            )

        if log_results:
            logger.info(
                "Average performance in %d batches: %f",
                num_eval_batches,
                perf,
            )

        return perf

    def save(self, epoch: int | None = None) -> None:
        """Save the trained model and its configuration."""

        if self.config.ephemeral:
            return

        # Save model
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
        }

        if epoch is not None:
            torch.save(
                checkpoint,
                RUNS_PATH / self.config.run_id / f"model_ckpt{epoch}.pth",
            )

        torch.save(
            checkpoint,
            RUNS_PATH / self.config.run_id / "model.pth",
        )

        logger.info("Saved model to %s", RUNS_PATH / self.config.run_id / "model.pth")

        # Save config
        with (RUNS_PATH / self.config.run_id / "config.json").open("w") as f:
            json.dump(
                self.config.to_dict(),
                f,
                indent=4,
            )
