# Need for Speed Carbon (GBA): self-driving agent

A pixel-based driving policy for *Need for Speed Carbon: Own the City* (GBA), trained with
tianshou PPO to drive tracks it has never seen.

## Results

Evaluated on 33 races on 7 held-out tracks that were never used for training
(deterministic policy, finish rate / wins):

| model | Easy | Normal | Hard | wins |
| --- | --- | --- | --- | --- |
| `models/nfs_dagger_v2.pth` (imitation of `RacingExpert`) | 64% | 64% | 55% | 0 |
| `models/nfs_ppo_v6.pth` (PPO) | 45% | 27% | 36% | 0 |

The agent follows unseen tracks but does not yet beat the AI. The RAM-based `RacingExpert`
(pure pursuit on the AI racing line, braking before corners, steering around opponents)
finishes ~90% of races but is still ~10% slower than the AI and wins only 6 of 147 races.

Findings about the game: the car can corner as hard as the AI but carries ~25% less speed
through the sharpest turns; steering ramps up while a direction is held and snaps back to
zero on release (0x202c644), so short repeated presses steer best; handbrake drifts, nitro
(SELECT), faster cars (the AI scales with your car), catch-up and blocking opponents did not help.

## Usage

```bash
# watch the agent drive a held-out track (or --video out.mp4 to record)
uv run python -m stable_retro.examples.nfs_drive --checkpoint stable_retro/examples/models/nfs_dagger_v2.pth --state Circuit.Parkside.Fwd.Hard

# evaluate finish / win rates on held-out tracks
uv run python -m stable_retro.examples.nfs_eval --checkpoint stable_retro/examples/models/nfs_dagger_v2.pth --split test

# train (PPO), optionally starting from imitation-learned weights
uv run python -m stable_retro.examples.nfs_dagger --out tb_logs_tianshou/dagger
uv run python -m stable_retro.examples.nfs_ppo_tianshou --init tb_logs_tianshou/dagger/policy.pth --lr 1e-4
```

## Pieces

- `nfs_env.py`: race env (frame skip 4, 4x 80x120 grayscale frames, 11 left/right symmetric
  actions). Reward = progress along the AI racing line + overtakes + lap/finish bonuses
  - wrong-way / crash / stuck penalties. Augmentation: random mirroring, brightness;
  mid-race restarts from snapshots.
- `nfs_ppo_tianshou.py`: tianshou 2.x PPO with a Nature CNN encoder shared by actor and critic.
- `nfs_expert.py`: RAM-based pure-pursuit teacher; `nfs_dagger.py`: imitation (DAgger).
- `nfs_eval.py` / `nfs_bench.py`: finish/win rates and pace vs. the leading AI car.
- Data (`stable_retro/data/stable/NeedForSpeedCarbon-GBA-v0/`): 147 race start states
  `<Event>.<Track>.<Fwd|Rev>.<Easy|Normal|Hard>` (catch-up off), `states_meta.json`,
  `racing_lines.npz` (AI path per state). Held-out tracks are listed in `nfs_env.HELDOUT_TRACKS`.

## RAM notes

See the docstrings of `nfs_env.py` and `nfs_expert.py`. Notably the `dist` counter
(0x202c6b6) is an odometer that also increases when driving the wrong way, and the `reverse`
variable in `data.json` (0x3005a84) is the game's wrong-way flag (red X icon).
