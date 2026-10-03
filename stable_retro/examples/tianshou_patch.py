import tianshou.data.utils.converter as converter
import torch
import numpy as np
from typing import Any

# MPS does not support float64, so downcast float64 arrays/tensors to float32 when converting
_original_to_torch = converter.to_torch


def to_torch(
    x: Any,
    dtype: torch.dtype | None = None,
    device: str | int | torch.device = "cpu",
):
    """Return an object without np.ndarray."""
    if dtype in (np.float64, torch.float64):
        dtype = torch.float32
    if dtype is None and isinstance(x, (np.ndarray, torch.Tensor)) and x.dtype in (np.float64, torch.float64):
        dtype = torch.float32
    return _original_to_torch(x, dtype, device)


converter.to_torch = to_torch


def load_policy_weights(policy: torch.nn.Module, path: str, device) -> dict | None:
    """Load policy weights from a checkpoint saved by tianshou 0.5 or 2.x.

    Accepts a tianshou 0.5 RainbowPolicy state dict (``model.*``, ``model_old.*``, ``support``),
    a tianshou 2.x algorithm state dict (``policy.*``, ``model_old.*``, ``_optimizers``), or a
    periodic checkpoint wrapping either of these.

    Returns the full algorithm state dict when the checkpoint has one (tianshou 2.x format), so the
    caller can also restore the optimizer and target network; otherwise None.
    """
    state_dict = torch.load(path, map_location=device)
    for key in ("algorithm_state_dict", "policy_state_dict"):
        if key in state_dict:
            state_dict = state_dict[key]

    if any(k.startswith("policy.") for k in state_dict):
        policy.load_state_dict({k.removeprefix("policy."): v for k, v in state_dict.items() if k.startswith("policy.")})
        return state_dict

    # Legacy tianshou 0.5 RainbowPolicy: drop the target network, the algorithm recreates it from the policy
    policy.load_state_dict({k: v for k, v in state_dict.items() if not k.startswith("model_old.")})
    return None
