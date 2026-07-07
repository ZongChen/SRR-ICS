"""Optional Weights & Biases integration.

The training code should run without wandb unless the user explicitly enables
logging. This module keeps wandb as an optional dependency while preserving the
small ``wandb.init`` / ``wandb.log`` call surface used in the project.
"""

try:
    import wandb as _wandb
except ImportError:  # pragma: no cover - depends on the local environment
    _wandb = None


def _require_wandb():
    if _wandb is None:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb`, "
            "or run without `--wandb_enabled`."
        )
    return _wandb


def init(*args, **kwargs):
    return _require_wandb().init(*args, **kwargs)


def log(*args, **kwargs):
    return _require_wandb().log(*args, **kwargs)


def __getattr__(name):
    return getattr(_require_wandb(), name)
