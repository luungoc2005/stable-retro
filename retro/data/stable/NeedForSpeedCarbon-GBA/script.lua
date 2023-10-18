prev_speed = 0

function speed_reward()
  local current_speed = data.speed1 * 100 + data.speed2 * 10 + data.speed3
  local speed_delta = current_speed - prev_speed
  prev_speed = current_speed

  if data.reverse > 0 then
    return -prev_speed
  else
    return prev_speed
  end
end