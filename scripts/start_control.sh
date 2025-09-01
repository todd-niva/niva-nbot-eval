#!/usr/bin/env bash
set -euo pipefail

UR_TYPE=${UR_TYPE:-ur10e}
ROBOT_IP=${ROBOT_IP:-0.0.0.0}
USE_FAKE=${USE_FAKE:-true}
FAKE_SENSOR_CMDS=${FAKE_SENSOR_CMDS:-true}
HEADLESS=${HEADLESS:-true}
LAUNCH_RVIZ=${LAUNCH_RVIZ:-false}
INITIAL_CTRL=${INITIAL_CTRL:-joint_trajectory_controller}
SPAWNER_TIMEOUT=${SPAWNER_TIMEOUT:-20}

LOG_FILE=/tmp/optA_control.log
PID_FILE=/tmp/optA_control.pid

# Ensure ROS environment is available inside script (for background nohup)
set +u
[ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash
[ -f /ros2_ws/install/setup.bash ] && . /ros2_ws/install/setup.bash
set -u

echo "[start_control] Stopping any previous instances..."
pkill -f ur_robotiq_control.launch.py || true
rm -f "$LOG_FILE" "$PID_FILE" || true

echo "[start_control] Launching control (fake=$USE_FAKE, ur_type=$UR_TYPE)..."
nohup ros2 launch ur_robotiq_description ur_robotiq_control.launch.py \
  ur_type:=${UR_TYPE} \
  robot_ip:=${ROBOT_IP} \
  use_fake_hardware:=${USE_FAKE} \
  fake_sensor_commands:=${FAKE_SENSOR_CMDS} \
  headless_mode:=${HEADLESS} \
  launch_rviz:=${LAUNCH_RVIZ} \
  initial_joint_controller:=${INITIAL_CTRL} \
  controller_spawner_timeout:=${SPAWNER_TIMEOUT} \
  >"${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

# Wait for readiness
echo "[start_control] Waiting for joint_state_broadcaster activation..."
READY=0
for i in $(seq 1 40); do
  if grep -q "Configured and activated joint_state_broadcaster" "$LOG_FILE"; then
    READY=1
    break
  fi
  sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
  echo "[start_control] Control not ready in time. Tail of log:"
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

echo "[start_control] Ready. PID=$(cat "$PID_FILE")"
exit 0



