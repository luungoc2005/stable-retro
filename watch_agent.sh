#!/bin/bash
# Script to watch a trained Street Fighter 2 agent

CHECKPOINT_DIR="tb_logs_tianshou/ppo-StreetFighterIISpecialChampionEdition-Genesis"

echo "================================================"
echo "Street Fighter 2 - Watch Trained Agent"
echo "================================================"
echo ""
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

if [ ! -f "$CHECKPOINT_DIR/policy.pth" ]; then
    echo "Error: No trained policy found at $CHECKPOINT_DIR/policy.pth"
    echo "Please train the agent first."
    exit 1
fi

echo "Loading trained agent and starting demo..."
echo ""

uv run python stable_retro/examples/sf_rainbow_tianshou.py \
    --watch \
    --checkpoint "$CHECKPOINT_DIR" \
    --training-num 1 \
    --device mps

echo ""
echo "Demo finished!"
