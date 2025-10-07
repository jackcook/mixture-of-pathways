import json
from pathlib import Path
from typing import Any, Literal, NamedTuple

MIGRATION_CONFIG = {
    "cost_based_entropy_scale": None,
    "cost_based_loss_beta": "cost_based_loss_epsilon",
    "disable_task_embedding_layer": False,
    "disable_wandb": True,
    "flat_expert_knockout_prob": None,
    "remove_fixation_loss": "disable_fixation_loss",
    "remove_loss_normalization": "disable_task_performance_scaling",
    "router_dim": 64,
    "routing_weight_noise": None,
    "task_filter": None,
    "within_expert_dropout_prob": None,
}


class Config(NamedTuple):
    """A configuration for an experiment."""

    batch_size: int
    checkpoint: str | None
    cost_based_loss_alpha: float
    cost_based_loss_epsilon: float
    device: Literal["cpu", "cuda", "mps"]
    disable_fixation_loss: bool
    disable_task_embedding_layer: bool
    disable_task_performance_scaling: bool
    disable_wandb: bool
    dropout_max_prob: float | None
    dropout_router_weight_threshold: float | None
    early_stopping_threshold: float | None
    ephemeral: bool
    expert_cost_exponent: float
    input_dim: int
    intermediate_dim: int
    layers: list[str]
    learning_rate: float
    num_epochs: int
    num_steps: int
    output_dim: int
    router_dim: int
    run_id: str
    task_dim: int
    task_id: str

    @classmethod
    def from_dict(cls, d: dict, *, migrate: bool = True) -> "Config":
        """Load a config from a dictionary."""

        if migrate:
            for k, v in MIGRATION_CONFIG.items():
                if k in d:
                    if v is None:
                        del d[k]
                    elif isinstance(v, str):
                        d[v] = d[k]
                        del d[k]
                # TODO(jack): This prevents us from adding keys to the config that
                # should have a string value.
                elif v is not None and not isinstance(v, str):
                    d[k] = v

        return cls(**d)

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        migrate: bool = True,
        **kwargs: dict[str, Any],
    ) -> "Config":
        """Load a config from a JSON file."""
        return cls.from_dict(
            {**json.loads(Path(path).read_text()), **kwargs},
            migrate=migrate,
        )

    def to_dict(self) -> dict[str, Any]:
        """Get the config as a dictionary."""
        return self._asdict()
