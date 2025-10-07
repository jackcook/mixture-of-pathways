import importlib
import netrc
import os
from pathlib import Path


def is_wandb_configured() -> bool:
    """Return True if wandb is configured, False otherwise."""
    if importlib.util.find_spec("wandb") is None:
        return False

    if os.environ.get("WANDB_API_KEY") is not None:
        return True

    netrc_path = Path.home() / ".netrc"

    if not netrc_path.exists():
        return False

    auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    return bool(auth)
