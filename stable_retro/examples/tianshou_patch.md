`tianshou_patch.py` is imported by the tianshou example scripts and provides:

- A patch for `tianshou.data.utils.converter.to_torch` that downcasts float64 arrays/tensors to
  float32, since the MPS backend does not support float64.
- `load_policy_weights(policy, path, device)`, which loads a `C51Policy` from either checkpoint format:
  - tianshou 0.5 `RainbowPolicy` state dicts (`model.*`, `model_old.*`, `support`), as saved by older runs
  - tianshou 2.x `RainbowDQN` algorithm state dicts (`policy.*`, `model_old.*`, `_optimizers`)
  - periodic `checkpoint_epoch_*.pth` files wrapping either of these

  It returns the full algorithm state dict for the 2.x format so training can also restore the
  optimizer and target network on `--resume`.
