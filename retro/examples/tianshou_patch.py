import tianshou.data.utils.converter as converter
import torch
import numpy as np
from typing import Any, Optional, Union
from copy import deepcopy
from numbers import Number
from tianshou.data.batch import Batch, _parse_value

def to_torch(
    x: Any,
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Union[Batch, torch.Tensor]:
    """Return an object without np.ndarray."""
    if dtype == np.float64:
        dtype = np.float32
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x, copy=True) if isinstance(x, dict) else deepcopy(x)
        x.to_torch(dtype, device)
        return x
    elif isinstance(x, (list, tuple)):
        return to_torch(_parse_value(x), dtype, device)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")
    
converter.to_torch = to_torch