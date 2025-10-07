import modal

# Apps
app = modal.App("pathways")

# Secrets
wandb_secret = modal.Secret.from_name("wandb-secret")

# Volumes
runs_volume = modal.Volume.from_name("pathways-runs", create_if_missing=True)
training_data_volume = modal.Volume.from_name(
    "pathways-training-data",
    create_if_missing=True,
)
