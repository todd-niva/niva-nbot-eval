#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=/tmp/optA_moveit.log
PID_FILE=/tmp/optA_moveit.pid

USE_FAKE=${USE_FAKE:-true}

# Ensure ROS environment is available inside script (for background nohup)
set +u
[ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash
[ -f /ros2_ws/install/setup.bash ] && . /ros2_ws/install/setup.bash
set -u

pkill -f ur_robotiq_moveit.launch.py || true
rm -f "$LOG_FILE" "$PID_FILE" || true

nohup ros2 launch ur_robotiq_moveit_config ur_robotiq_moveit.launch.py \
  use_fake_hardware:=${USE_FAKE} \
  >"${LOG_FILE}" 2>&1 &

echo $! > "$PID_FILE"

READY=0
for i in $(seq 1 60); do
  if grep -Eq "PlanningSceneMonitor|ROBOT_DESCRIPTION" "$LOG_FILE"; then
    READY=1
    break
  fi
  sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
  echo "[start_moveit] MoveIt not ready in time. Tail of log:"
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

echo "[start_moveit] Ready. PID=$(cat "$PID_FILE")"
exit 0

