prev_speed = 0
reward_coeff = .01
neg_speed_penalty = 3

function speed_reward()
  local current_speed = data.speed1 * 100 + data.speed2 * 10 + data.speed3
  local speed_delta = current_speed - prev_speed
  prev_speed = current_speed

  if current_speed == 0 then
    -- heavy penalty on hitting an obstacle
    return -1
  end

  if speed_delta < -20 then
    -- heavy penalty for slowing down too much
    speed_delta = speed_delta * neg_speed_penalty
  end

  if data.reverse > 0 then
    return -math.max(10, speed_delta) * reward_coeff * neg_speed_penalty
  else
    return speed_delta * reward_coeff
  end
end