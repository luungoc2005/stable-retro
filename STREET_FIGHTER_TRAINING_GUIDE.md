# Street Fighter II Training Improvements Guide

## Problems Identified

Your agent was experiencing three main issues:
1. **Too defensive** - Preferred staying back rather than engaging
2. **Poor blocking** - Didn't learn defensive techniques
3. **Can't handle reverse side** - Performed poorly when opponent was on the left

## Solutions Implemented

### 1. Improved Reward Function (`script_aggressive.lua`)

**Key Changes:**

- **Reduced damage penalty coefficient** (3.0 → 1.5): Agent is now willing to trade damage for offense
- **Aggression bonuses**:
  - Landing hits gives 0.5x damage dealt bonus
  - Combo system: 30-frame window with 0.3 bonus for consecutive hits
  - Proximity rewards: 0.15 for being very close (<40px), 0.05 for medium range (<80px)
- **Distance management**:
  - Rewards closing distance (0.002 per pixel)
  - Penalizes excessive retreating (-0.1 if >1.5x initial distance)
- **Defense bonus**: Small reward (0.02) for sustained blocking/avoiding damage when enemy is close
- **Coward penalty**: Discourages staying too far away

### 2. Improved Flip Wrapper (`StreetFighterFlipEnvWrapper`)

**Key Changes:**

- **Consistent perspective**: Agent always sees itself on the left side
- **Data augmentation**: Optional `always_flip` parameter randomly flips observations during training
- **Better action mapping**: Properly handles LEFT/RIGHT button swapping in all scenarios
- **Fixed reset behavior**: Correctly initializes flip state

### 3. Updated Hyperparameters

**Key Changes:**

- **n_steps**: 64 → 128 (better sample efficiency)
- **gamma**: 0.995 → 0.99 (focus on immediate rewards/aggression)
- **ent_coef**: 0.005 → 0.01 (more exploration of aggressive strategies)

## How to Use

### Option 1: Use Aggressive Scenario (Recommended)

```bash
python retro/examples/ppo.py \
    --game StreetFighterIISpecialChampionEdition-Genesis \
    --scenario scenario_aggressive.json \
    --flip-augment
```

### Option 2: Replace Default Scenario

If you want this to be the default behavior, rename the files:

```bash
cd retro/data/stable/StreetFighterIISpecialChampionEdition-Genesis/
mv scenario.json scenario_old.json
mv scenario_aggressive.json scenario.json
mv script.lua script_old.lua
mv script_aggressive.lua script.lua
```

Then train normally:

```bash
python retro/examples/ppo.py \
    --game StreetFighterIISpecialChampionEdition-Genesis \
    --flip-augment
```

### Option 3: Compare Both

Train two agents side-by-side to compare:

```bash
# Original defensive agent
python retro/examples/ppo.py \
    --game StreetFighterIISpecialChampionEdition-Genesis

# New aggressive agent
python retro/examples/ppo.py \
    --game StreetFighterIISpecialChampionEdition-Genesis \
    --scenario scenario_aggressive.json \
    --flip-augment
```

## Additional Training Tips

### 1. Action Bias for Specific Behaviors

Encourage specific moves during training:

```bash
# Encourage forward movement and attacks (example)
python retro/examples/ppo.py \
    --game StreetFighterIISpecialChampionEdition-Genesis \
    --scenario scenario_aggressive.json \
    --flip-augment \
    --action-bias "0 0 0 0.05 0 0 0.1 0.1 0.1 0 0 0"
```

The action bias corresponds to button order. Check button meanings with `env.buttons`.

### 2. Resume Training

If training is interrupted:

```bash
python retro/examples/ppo.py \
    --game StreetFighterIISpecialChampionEdition-Genesis \
    --scenario scenario_aggressive.json \
    --flip-augment \
    --resume
```

### 3. Use IMPALA CNN

For potentially better feature extraction:

```bash
python retro/examples/ppo.py \
    --game StreetFighterIISpecialChampionEdition-Genesis \
    --scenario scenario_aggressive.json \
    --flip-augment \
    --cnn impala
```

## Expected Improvements

With these changes, you should see:

1. **More aggressive play**: Agent actively seeks to close distance and land hits
2. **Better combos**: Consecutive hit bonus encourages follow-up attacks
3. **Improved blocking**: Small rewards for successful defense
4. **Direction-agnostic**: Equal performance regardless of which side the agent is on
5. **Better exploration**: Higher entropy coefficient encourages trying different strategies

## Monitoring Progress

Watch TensorBoard for these metrics:

```bash
tensorboard --logdir tb_logs
```

Key metrics to monitor:
- **Episode reward**: Should increase steadily
- **Episode length**: Should stabilize (not ending too quickly)
- **Value loss**: Should decrease over time
- **Entropy**: Should remain positive (exploration)

## Troubleshooting

### Agent still too defensive?

- Increase proximity bonuses in `script_aggressive.lua` (lines 52-57)
- Increase coward penalty (line 76)
- Further reduce `reward_coeff` (line 8)

### Agent too reckless?

- Increase defense bonus (line 72)
- Increase `reward_coeff` slightly
- Add health preservation bonus

### Poor performance on reverse side?

- Ensure `--flip-augment` flag is used
- Check that `StreetFighterFlipEnvWrapper` is in `GAME_WRAPPERS`
- Verify observations are being flipped correctly (add debug prints)

## File Summary

- **script_aggressive.lua**: New reward function with aggression bonuses
- **scenario_aggressive.json**: Scenario config pointing to new script
- **wrappers.py**: Improved `StreetFighterFlipEnvWrapper` with augmentation
- **ppo.py**: Updated with flip-augment flag and improved hyperparameters

## Next Steps

1. Start training with the aggressive scenario and flip augmentation
2. Monitor for 5-10M steps to see if behavior improves
3. If needed, fine-tune reward coefficients based on observed behavior
4. Consider curriculum learning: start with easier opponents, gradually increase difficulty
