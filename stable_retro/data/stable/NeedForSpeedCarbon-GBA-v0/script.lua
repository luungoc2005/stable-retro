prev_speed = 0
step_count = 0
stuck_count = 0
max_stuck_frames = 60  -- ~1 second at 60fps before escalating penalty

function speed_reward()
  local current_speed = data.speed1 * 100 + data.speed2 * 10 + data.speed3
  local speed_delta = current_speed - prev_speed
  prev_speed = current_speed
  step_count = step_count + 1

  -- Track how long the car has been stuck at 0 speed
  if current_speed == 0 then
    stuck_count = stuck_count + 1
  else
    stuck_count = 0
  end

  -- Heavy penalty for reversing
  if data.reverse > 0 then
    return -0.3
  end

  -- Penalty when stopped (crashed / stuck)
  if current_speed == 0 then
    if stuck_count > max_stuck_frames then
      return -0.5  -- escalating penalty for being stuck a long time
    end
    return -0.2
  end

  -- === Main reward: proportional to current speed ===
  -- KEY FIX: reward maintaining high speed, not just speed changes.
  -- Typical speeds: 100-300. At speed 250 -> 0.0125 per frame.
  -- Over 4-frame skip (summed): ~0.05 per agent step.
  -- With gamma=0.99, V ~ 0.05 / 0.01 = 5, fits in [-10, 10].
  local speed_bonus = current_speed * 0.00005

  -- === Secondary: small speed change component ===
  local delta_bonus = 0
  if speed_delta < -15 then
    -- Penalty for sharp deceleration (hit wall, missed turn)
    delta_bonus = speed_delta * 0.001
  elseif speed_delta > 5 then
    -- Small bonus for accelerating
    delta_bonus = speed_delta * 0.0003
  end

  return speed_bonus + delta_bonus
end