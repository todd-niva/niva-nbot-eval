#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=${1:-/tmp/optA_control.log}

echo "[health] Checking control manager and joint_states..."
python3 - <<'PY'
import time
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers
from sensor_msgs.msg import JointState

class Health(Node):
    def __init__(self):
        super().__init__('health_check')
        self.cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self.js_ok = False
        self.create_subscription(JointState, '/joint_states', self._cb, 10)
    def _cb(self, msg: JointState):
        if msg.name:
            self.js_ok = True

rclpy.init()
node = Health()
try:
    ok = node.cli.wait_for_service(timeout_sec=5.0)
    print(f"cm_service_available={ok}")
    start = time.time()
    while time.time() - start < 3.0 and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"joint_states_received={node.js_ok}")
finally:
    node.destroy_node()
    rclpy.shutdown()
PY

if ! grep -q "Configured and activated joint_state_broadcaster" "$LOG_FILE"; then
  echo "[health] joint_state_broadcaster not confirmed active in logs"
  exit 2
fi

echo "[health] OK"

