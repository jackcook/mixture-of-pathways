import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import Config


class Expert(nn.Module):
    """A GRU-based module that processes the input and returns an output."""

    def __init__(
        self,
        config: Config,
        hidden_dim: int,
    ) -> None:
        """
        Initialize an expert. An expert is a GRU-based module that processes the input
        and returns an output.

        :param: config: The configuration for the model.
        :param: hidden_dim: The dimension of the hidden state of the GRU.
        """

        super().__init__()

        self.identity = hidden_dim == 0

        if not self.identity:
            self.rnn = nn.GRU(config.intermediate_dim, hidden_dim)
            self.batchnorm = nn.BatchNorm1d(hidden_dim)
            self.relu = nn.ReLU()
            self.output_layer = nn.Linear(hidden_dim, config.intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the expert.

        :param: x: Input tensor of dimension (batch_size, seq_len, input_dim).
        :return: Output tensor of dimension (batch_size, seq_len, intermediate_dim).
        """

        if self.identity:
            return x
        x, _ = self.rnn(x)

        if x.var() == 0:
            warnings.warn(
                f"Expert input of shape {x.shape} has no variance, constant value "
                f"{x.mean()}",
                stacklevel=2,
            )
        else:
            x = self.batchnorm(x)

        x = self.relu(x)
        return self.output_layer(x)


class CostBasedRouter(nn.Module):
    """
    A cost-based router, which uses a GRU to route the input to the experts based on
    the active task and the previous layer's output.
    """

    def __init__(
        self,
        config: Config,
        expert_dims: list[int],
        num_tasks: int,
    ) -> None:
        """
        Initialize a cost-based router. A cost-based router is a router that uses a
        cost-based routing algorithm to route the input to the experts.

        :param: config: The configuration for the model.
        :param: expert_dims: The dimensions of the experts in this router.
        :param: num_tasks: The number of tasks the model is being trained on.
        """

        super().__init__()

        self.config = config
        self.expert_dims = expert_dims
        self.num_tasks = num_tasks

        self.rnn = nn.GRU(config.intermediate_dim, config.router_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(config.router_dim, len(expert_dims))

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(
        self,
        prev_layer_output: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        inference: bool = False,
        inference_dropout_threshold: float | None = None,
        inference_disable_complex_experts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        """
        Forward pass of the cost-based router.

        :param: prev_layer_output: The output of the previous layer in the model.
        :param: task_ids: Tensor of dimension (batch_size, seq_len) with the IDs of the
            active task at each timestep.
        :param: inference: Whether to run the router in inference mode.
        :param: inference_dropout_threshold: If set when running in inference mode,
            experts with a routing weight below this threshold will be disabled.
        :param: inference_disable_complex_experts: If set to true when running in
            inference mode, the most complex expert in each layer will be disabled.
        :return: A tuple containing the raw router output, the router output, the
            indices of the experts to activate, and the task expert usage losses.
        """

        logits, _ = self.rnn(prev_layer_output)
        logits = self.relu(logits)
        logits = self.output_layer(logits)
        indices = torch.zeros(
            (logits.shape[0], logits.shape[1], len(self.expert_dims)),
            dtype=torch.long,
            device=logits.device,
        )

        for i in range(len(self.expert_dims)):
            indices[:, :, i] = i

        raw_router_output = F.softmax(logits, dim=-1)
        router_output = raw_router_output.clone()

        # Handle dropout
        if inference:
            if inference_dropout_threshold is not None:
                # Disable experts with low weight
                indices[router_output < inference_dropout_threshold] = -1
                logits[router_output < inference_dropout_threshold] = float("-inf")
                router_output = F.softmax(logits, dim=-1)

            if inference_disable_complex_experts:
                indices[:, :, -1] = -1
                logits[:, :, -1] = float("-inf")
                router_output = F.softmax(logits, dim=-1)
        elif (
            self.config.dropout_max_prob is not None
            and self.config.dropout_router_weight_threshold is not None
        ):
            # Random dropout
            mask = (router_output < self.config.dropout_router_weight_threshold) & (
                torch.rand_like(router_output)
                < (
                    self.config.dropout_max_prob
                    - (
                        self.config.dropout_max_prob
                        / self.config.dropout_router_weight_threshold
                    )
                    * router_output
                )
            )

            logits = torch.where(mask, float("-inf"), logits)
            indices = torch.where(mask, -1, indices)
            router_output = F.softmax(logits, dim=-1)

        if len(self.expert_dims) > 1:
            routing_costs = torch.tensor(
                [
                    expert_dim**self.config.expert_cost_exponent
                    for expert_dim in self.expert_dims
                ],
                dtype=logits.dtype,
                device=logits.device,
            )

            task_expert_usage_losses = {}

            # Separate expert usage loss for each task
            expert_usage_loss = torch.einsum(
                "ijk,k->ij",
                raw_router_output,
                routing_costs,
            )
            for i in range(self.num_tasks):
                task_mask = task_ids == i
                task_expert_usage_losses[i] = (
                    expert_usage_loss[task_mask].sum() / task_mask.sum()
                )
        else:
            task_expert_usage_losses = None

        return (raw_router_output, router_output, indices, task_expert_usage_losses)


class HeterogeneousMoE(nn.Module):
    """
    A heterogeneous mixture of experts (HMoE), consisting of a cost-based router and
    several experts of different sizes.
    """

    def __init__(
        self,
        config: Config,
        expert_dims: list[int],
        num_tasks: int,
    ) -> None:
        """
        Initialize a heterogeneous mixture of experts (HMoE). An HMoE consists of a
        cost-based router and several experts of different sizes.

        :param: config: The configuration for the model.
        :param: expert_dims: The dimensions of the experts in this HMoE.
        :param: num_tasks: The number of tasks the model is being trained on.
        """

        super().__init__()

        self.router = CostBasedRouter(
            config,
            expert_dims,
            num_tasks,
        )
        self.experts = nn.ModuleList(
            [Expert(config, expert_dim) for expert_dim in expert_dims],
        )

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        inference: bool = False,
        inference_dropout_threshold: float | None = None,
        inference_disable_complex_experts: bool = False,
        output_activations: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the heterogeneous mixture of experts (HMoE).

        :param: x: Input tensor of dimension (batch_size, seq_len, input_dim).
        :param: task_ids: Tensor of dimension (batch_size, seq_len) with the IDs of the
            active task at each timestep.
        :param: inference: Whether to run the HMoE in inference mode.
        :param: inference_dropout_threshold: If set when running in inference mode,
            experts with a routing weight below this threshold will be disabled.
        :param: inference_disable_complex_experts: If set to true when running in
            inference mode, the most complex expert in each layer will be disabled.
        :param: output_activations: Whether to output the activations of the experts.
        :return: A tuple containing the output of the HMoE, the router logits, task
            expert usage losses, and expert activations if output_activations is set.
        """

        (raw_router_output, router_output, indices, task_expert_usage_losses) = (
            self.router(
                x,
                task_ids,
                inference=inference,
                inference_dropout_threshold=inference_dropout_threshold,
                inference_disable_complex_experts=inference_disable_complex_experts,
            )
        )
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_router_output = router_output.view(-1, router_output.size(-1))

        if output_activations:
            expert_activations = []

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if not flat_mask.any():
                continue

            expert_input = flat_x[flat_mask]
            expert_output = expert(expert_input)

            # Extract and apply gating scores
            gating_scores = flat_router_output[flat_mask, i]
            weighting_output = torch.einsum("i,ij->ij", gating_scores, expert_output)

            if output_activations:
                expert_activations.append(weighting_output.cpu().detach().numpy())

            # Update final output
            final_output[expert_mask] += weighting_output.squeeze(1)

        output = (final_output, raw_router_output, task_expert_usage_losses)

        if output_activations:
            output += (expert_activations,)

        return output


class Block(nn.Module):
    """
    A block of the model, consisting of a heterogeneous mixture of experts (HMoE) and a
    layer norm.
    """

    def __init__(
        self,
        config: Config,
        expert_dims: list[int],
        num_tasks: int,
    ) -> None:
        """
        Initialize a block of the model. A block consists of a heterogeneous mixture of
        experts (HMoE) and a layer norm.

        :param: config: The configuration for the model.
        :param: expert_dims: The dimensions of the experts in this block.
        :param: num_tasks: The number of tasks the model is being trained on.
        """

        super().__init__()
        self.hmoe = HeterogeneousMoE(
            config,
            expert_dims,
            num_tasks,
        )
        self.ln = nn.LayerNorm(config.intermediate_dim)

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        inference: bool = False,
        inference_dropout_threshold: float | None = None,
        inference_disable_complex_experts: bool = False,
        output_activations: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the block.

        :param: x: Input tensor of dimension (batch_size, seq_len, input_dim).
        :param: task_ids: Tensor of dimension (batch_size, seq_len) with the IDs of the
            active task at each timestep.
        :param: inference: Whether to run the model in inference mode.
        :param: inference_dropout_threshold: If set when running in inference mode,
            experts with a routing weight below this threshold will be disabled.
        :param: inference_disable_complex_experts: If set to true when running in
            inference mode, the most complex expert in each layer will be disabled.
        :param: output_activations: Whether to output the activations of the experts.
        :return: A tuple containing the output of the block, the router logits, task
            expert usage losses, and expert activations if output_activations is set.
        """
        moe_outputs = self.hmoe(
            x,
            task_ids,
            inference=inference,
            inference_dropout_threshold=inference_dropout_threshold,
            inference_disable_complex_experts=inference_disable_complex_experts,
            output_activations=output_activations,
        )

        # Add post-layer norm
        return (
            self.ln(x + moe_outputs[0]),
            *moe_outputs[1:],
        )


class Model(nn.Module):
    """The model consisting of layers of heterogeneous mixtures of experts."""

    @classmethod
    def from_run(
        cls,
        run_id: str,
        *,
        runs_dir: str = "runs",
        checkpoint: int | None = None,
        **kwargs: dict[str, Any],
    ) -> "Model":
        """
        Load a model from a saved run.

        :param: run_id: The id of the run.
        :param: runs_dir: The directory in which this run is contained.
        :param: checkpoint: The checkpoint number to load. If None, the latest
            checkpoint will be loaded.
        :return: The loaded model.
        """

        run_path = Path(runs_dir) / run_id
        config = Config.from_path(run_path / "config.json", **kwargs)
        model = cls(config)

        model_path = None

        if checkpoint is None:
            if (run_path / "model.pth").exists():
                model_path = run_path / "model.pth"
            else:
                model_paths = run_path.glob("model_*.pth")

                if len(model_paths) == 0:
                    msg = f"No model checkpoints found at {run_path}"
                    raise FileNotFoundError(msg)

                latest_checkpoint_num = max(
                    [
                        int(path.split("_")[-1].split(".")[0].replace("ckpt", ""))
                        for path in model_paths
                    ],
                )
                model_path = run_path / f"model_ckpt{latest_checkpoint_num}.pth"
        else:
            model_path = run_path / f"model_ckpt{checkpoint}.pth"

        if model_path is None:
            msg = f"No model checkpoint found at {run_path}"
            raise FileNotFoundError(msg)

        state_dict = torch.load(
            model_path,
            weights_only=False,
            map_location="cpu" if config.device == "cpu" else None,
        )

        if "model" in state_dict:
            state_dict = state_dict["model"]

        # Migrate old models: Older models used the name "sparse_moe" for the HMoE
        # layer, but this has since been renamed to "hmoe".
        if any(k for k in state_dict if "sparse_moe" in k):
            old_state_dict = state_dict.copy()

            for k, v in old_state_dict.items():
                if "sparse_moe" in k:
                    state_dict[k.replace("sparse_moe", "hmoe")] = v
                    del state_dict[k]

        model.load_state_dict(state_dict)
        return model

    def __init__(
        self,
        config: Config,
    ) -> None:
        """
        Initialize the model.

        :param: config: The configuration for the model.
        """

        super().__init__()

        self.config = config
        self.num_tasks = 82 if config.task_id == "modcog/all" else 1
        self.device = torch.device(config.device)

        # Initialize model components
        if self.num_tasks > 1 and not self.config.disable_task_embedding_layer:
            self.embedding = nn.Embedding(
                num_embeddings=self.num_tasks,
                embedding_dim=self.config.task_dim,
            )
            self.input_layer = nn.Linear(
                self.config.input_dim - self.num_tasks + self.config.task_dim,
                self.config.intermediate_dim,
            )
        else:
            self.input_layer = nn.Linear(
                self.config.input_dim,
                self.config.intermediate_dim,
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.config,
                    [int(expert_size) for expert_size in layer.split(",")],
                    self.num_tasks,
                )
                for layer in self.config.layers
            ],
        )
        self.output_layer = nn.Linear(
            self.config.intermediate_dim,
            self.config.output_dim,
        )
        self.to(self.config.device)

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def forward(
        self,
        x: torch.Tensor,
        *,
        inference: bool = False,
        inference_dropout_threshold: float | None = None,
        inference_disable_complex_experts: bool = False,
        output_activations: bool = False,
        output_all_pathways: bool = False,
    ) -> tuple:
        """
        Forward pass of the model.

        :param: x: Input tensor.
        :param: inference: Whether to run in inference mode.
        :param: inference_dropout_threshold: If set when running in inference mode,
            experts with a routing weight below this threshold will be disabled.
        :param: inference_disable_complex_experts: If set to true when running in
            inference mode, the most complex expert in each layer will be disabled.
        :param: output_activations: Whether to output the activations of the experts.
        :param: output_all_pathways: Whether to output the usages of all pathways.
        """

        task_ids = (
            x[:, :, 33:].argmax(dim=-1)
            if self.num_tasks > 1
            else torch.zeros(x.shape[0], x.shape[1], device=self.device)
        )

        if self.num_tasks > 1 and not self.config.disable_task_embedding_layer:
            x = torch.cat([x[:, :, :33], self.embedding(task_ids)], dim=-1)

        x = self.input_layer(x)

        total_task_expert_usage_losses = {
            i: torch.tensor(0.0, device=self.device) for i in range(self.num_tasks)
        }
        expert_usages = []
        expert_activations = []

        for block in self.blocks:
            block_outputs = block(
                x,
                task_ids,
                inference=inference,
                inference_dropout_threshold=inference_dropout_threshold,
                inference_disable_complex_experts=inference_disable_complex_experts,
                output_activations=output_activations,
            )

            x = block_outputs[0]
            router_logits = block_outputs[1]
            task_expert_usage_losses = block_outputs[2]

            if output_activations:
                block_activations = block_outputs[4]

            if task_expert_usage_losses is not None:
                for k in task_expert_usage_losses:
                    total_task_expert_usage_losses[k] = (
                        total_task_expert_usage_losses[k] + task_expert_usage_losses[k]
                    )

            if output_all_pathways:
                expert_usages.append(router_logits.cpu().detach().numpy())

            if output_activations:
                expert_activations.append(block_activations)

        x = self.output_layer(x)
        output = (x, task_ids, total_task_expert_usage_losses)

        if output_all_pathways:
            output += (expert_usages,)

        if output_activations:
            output += (expert_activations,)

        return output
