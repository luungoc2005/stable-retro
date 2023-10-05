prev_health = 0
prev_enemy_health = 0
full_hp = -1
full_enemy_hp = -1
initial_distance = -1
reward_coeff = 3
overal_coeff = 0.01

function custom_reward ()
  if full_hp == -1 then
    full_hp = data.health
  end
  if full_enemy_hp == -1 then
    full_enemy_hp = data.enemy_health
  end

  local health_delta = data.health - prev_health
  local enemy_health_delta = data.enemy_health - prev_enemy_health

  prev_health = data.health
  prev_enemy_health = data.enemy_health

  distance = math.abs(data.enemy_x - data.agent_x)
  if initial_distance == -1 then
    initial_distance = distance
  end

  local reward = 0
  if data.health < 0 then
    reward = -(full_hp ^ ((data.enemy_health + 1) / (full_hp + 1))) * 0.01
  elseif data.enemy_health < 0 then
    reward = (full_hp ^ ((data.health + 1) / (full_enemy_hp + 1))) * 0.01 * reward_coeff
  else
    reward = health_delta - enemy_health_delta * reward_coeff - math.max(distance - initial_distance, 0) * 0.05
  end

  return reward * overal_coeff
end