prev_speed = 0
prev_score = 0
speed_score = 3
-- speed_multiplier = 1
offroad_penalty = 1
normalize_coeff = 0.01

function custom_reward()
  local speed_delta = data.speed - prev_speed
  local score_delta = data.score - prev_score

  prev_speed = data.speed
  prev_score = data.score

  local reward = 0.1 * speed_delta + 0.001 * score_delta
  -- score not increasing - means we're offroad
  if score_delta <= 0 and data.speed > 0 then
    reward = reward - offroad_penalty
  end

  if speed_delta > 0 then
    reward = reward + speed_score
  elseif speed_delta < 0 then
    reward = reward - speed_score
  end

  return reward * normalize_coeff
end