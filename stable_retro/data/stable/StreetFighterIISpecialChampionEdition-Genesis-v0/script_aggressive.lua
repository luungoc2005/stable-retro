-- Improved reward function for more aggressive play and better learning
prev_health = 0
prev_enemy_health = 0
full_hp = -1
full_enemy_hp = -1
initial_distance = -1
reward_coeff = 1.5  -- Reduced from 3 to encourage trading damage
normalize_coeff = 0.01
combo_timer = 0
prev_distance = -1

-- Track blocking behavior
blocking_bonus_timer = 0
prev_agent_y = -1

function custom_reward ()
  if full_hp == -1 then
    full_hp = data.health
  end
  if full_enemy_hp == -1 then
    full_enemy_hp = data.enemy_health
  end

  local health_delta = data.health - prev_health
  local enemy_health_delta = data.enemy_health - prev_enemy_health

  distance = math.abs(data.enemy_x - data.agent_x)
  if initial_distance == -1 then
    initial_distance = distance
  end
  if prev_distance == -1 then
    prev_distance = distance
  end

  local reward = 0
  
  -- Terminal rewards (end of round)
  if data.health < 0 then
    reward = -(full_hp ^ ((data.enemy_health + 1) / (full_hp + 1))) * 0.2
  elseif data.enemy_health < 0 then
    reward = (full_hp ^ ((data.health + 1) / (full_enemy_hp + 1))) * reward_coeff * 0.2
  else
    -- Normal gameplay rewards
    
    -- 1. Damage dealt/received with more balanced coefficients
    local damage_reward = health_delta - enemy_health_delta * reward_coeff
    
    -- 2. Encourage aggressive closing distance
    local distance_delta = prev_distance - distance
    local distance_reward = distance_delta * 0.002  -- Reward for closing distance
    
    -- 3. Large bonus for dealing damage (encourage aggression)
    local damage_dealt_bonus = 0
    if enemy_health_delta < 0 then
      damage_dealt_bonus = -enemy_health_delta * 0.5  -- Big bonus for landing hits
      combo_timer = 30  -- Start combo window
    end
    
    -- 4. Combo bonus (reward for consecutive hits)
    local combo_bonus = 0
    if combo_timer > 0 then
      combo_timer = combo_timer - 1
      if enemy_health_delta < 0 then
        combo_bonus = 0.3  -- Extra reward during combo window
      end
    end
    
    -- 5. Proximity bonus (reward being close to enemy)
    local proximity_bonus = 0
    if distance < 40 then  -- Very close
      proximity_bonus = 0.15
    elseif distance < 80 then  -- Medium range
      proximity_bonus = 0.05
    end
    
    -- 6. Blocking/defense detection bonus
    -- If agent takes no damage while enemy is close, small reward
    local defense_bonus = 0
    if health_delta == 0 and distance < 60 and data.enemy_health == prev_enemy_health then
      blocking_bonus_timer = blocking_bonus_timer + 1
      if blocking_bonus_timer > 5 then  -- Sustained defense
        defense_bonus = 0.02
      end
    else
      blocking_bonus_timer = 0
    end
    
    -- 7. Penalty for staying too far away
    local coward_penalty = 0
    if distance > initial_distance * 1.5 then
      coward_penalty = -0.1
    end
    
    reward = damage_reward + distance_reward + damage_dealt_bonus + combo_bonus + proximity_bonus + defense_bonus + coward_penalty
  end

  prev_health = data.health
  prev_enemy_health = data.enemy_health
  prev_distance = distance
  if prev_agent_y == -1 then
    prev_agent_y = data.agent_y
  end

  return reward * normalize_coeff
end
