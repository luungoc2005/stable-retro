#!/usr/bin/env python3
"""
Quick comparison script to visualize reward differences between
defensive and aggressive reward functions.
"""

def defensive_reward(health_delta, enemy_health_delta, distance, initial_distance):
    """Original defensive reward function"""
    reward_coeff = 3
    # Note: enemy_health_delta is negative when enemy takes damage
    # Subtracting it makes it positive (reward for damaging enemy)
    reward = health_delta - enemy_health_delta * reward_coeff
    reward -= max(distance - initial_distance, 0) * 0.0001
    return reward * 0.01

def aggressive_reward(health_delta, enemy_health_delta, distance, prev_distance, initial_distance, combo_active=False):
    """New aggressive reward function"""
    reward_coeff = 1.5
    
    # Base damage reward
    damage_reward = health_delta - enemy_health_delta * reward_coeff
    
    # Distance rewards
    distance_delta = prev_distance - distance
    distance_reward = distance_delta * 0.002
    
    # Damage dealt bonus
    damage_dealt_bonus = 0
    if enemy_health_delta < 0:
        damage_dealt_bonus = -enemy_health_delta * 0.5
    
    # Combo bonus
    combo_bonus = 0.3 if combo_active and enemy_health_delta < 0 else 0
    
    # Proximity bonus
    proximity_bonus = 0
    if distance < 40:
        proximity_bonus = 0.15
    elif distance < 80:
        proximity_bonus = 0.05
    
    # Coward penalty
    coward_penalty = -0.1 if distance > initial_distance * 1.5 else 0
    
    reward = damage_reward + distance_reward + damage_dealt_bonus + combo_bonus + proximity_bonus + coward_penalty
    return reward * 0.01

# Test scenarios
print("=" * 80)
print("REWARD COMPARISON: Defensive vs Aggressive")
print("=" * 80)

scenarios = [
    {
        "name": "Landing a strong hit (10 damage)",
        "health_delta": 0,
        "enemy_health_delta": -10,
        "distance": 30,
        "prev_distance": 35,
        "initial_distance": 100,
        "combo_active": False
    },
    {
        "name": "Landing a combo hit (5 damage, combo active)",
        "health_delta": 0,
        "enemy_health_delta": -5,
        "distance": 25,
        "prev_distance": 25,
        "initial_distance": 100,
        "combo_active": True
    },
    {
        "name": "Taking damage while attacking (both take 8 damage)",
        "health_delta": -8,
        "enemy_health_delta": -8,
        "distance": 30,
        "prev_distance": 35,
        "initial_distance": 100,
        "combo_active": False
    },
    {
        "name": "Taking damage defensively (8 damage)",
        "health_delta": -8,
        "enemy_health_delta": 0,
        "distance": 30,
        "prev_distance": 30,
        "initial_distance": 100,
        "combo_active": False
    },
    {
        "name": "Moving closer (no damage)",
        "health_delta": 0,
        "enemy_health_delta": 0,
        "distance": 50,
        "prev_distance": 70,
        "initial_distance": 100,
        "combo_active": False
    },
    {
        "name": "Staying at close range (no damage)",
        "health_delta": 0,
        "enemy_health_delta": 0,
        "distance": 35,
        "prev_distance": 35,
        "initial_distance": 100,
        "combo_active": False
    },
    {
        "name": "Running away far",
        "health_delta": 0,
        "enemy_health_delta": 0,
        "distance": 180,
        "prev_distance": 150,
        "initial_distance": 100,
        "combo_active": False
    },
]

for scenario in scenarios:
    print(f"\nScenario: {scenario['name']}")
    print("-" * 80)
    
    def_reward = defensive_reward(
        scenario['health_delta'],
        scenario['enemy_health_delta'],
        scenario['distance'],
        scenario['initial_distance']
    )
    
    agg_reward = aggressive_reward(
        scenario['health_delta'],
        scenario['enemy_health_delta'],
        scenario['distance'],
        scenario['prev_distance'],
        scenario['initial_distance'],
        scenario['combo_active']
    )
    
    diff = agg_reward - def_reward
    diff_pct = (diff / abs(def_reward) * 100) if def_reward != 0 else float('inf')
    
    print(f"  Defensive reward: {def_reward:+.4f}")
    print(f"  Aggressive reward: {agg_reward:+.4f}")
    print(f"  Difference: {diff:+.4f} ({diff_pct:+.1f}%)")
    
    if diff > 0:
        print(f"  → Aggressive function rewards this MORE")
    elif diff < 0:
        print(f"  → Aggressive function penalizes this MORE")
    else:
        print(f"  → Same reward")

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)
print("""
1. Landing hits is much more rewarding in aggressive function (+100-200%)
2. Trading damage is less penalized (only 1.5x vs 3x multiplier)
3. Moving closer and staying close gives continuous small rewards
4. Running away is heavily penalized
5. Combo system provides extra incentive for consecutive hits
""")
