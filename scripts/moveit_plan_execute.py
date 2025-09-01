#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, RobotState
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes


class MoveItPlanExecute(Node):
    def __init__(self) -> None:
        super().__init__("moveit_plan_execute_cli")
        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.execute_action_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.latest_joint_state: JointState | None = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        if msg.name and len(msg.name) == len(msg.position):
            self.latest_joint_state = msg

    def wait_for_current_state(self, timeout_sec: float = 3.0) -> JointState | None:
        start = time.time()
        while time.time() - start < timeout_sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_joint_state is not None:
                return self.latest_joint_state
        return None

    def plan_and_execute(self) -> bool:
        if not self.plan_client.wait_for_service(timeout_sec=6.0):
            self.get_logger().error("Plan service not available: /plan_kinematic_path")
            return False
        if not self.execute_action_client.wait_for_server(timeout_sec=6.0):
            self.get_logger().error("Execute action not available: /execute_trajectory")
            return False

        js = self.wait_for_current_state(3.0)
        if js is None:
            self.get_logger().error("No joint states received")
            return False

        start_state = RobotState()
        start_state.joint_state = js

        joint_delta_map = {"shoulder_pan_joint": 0.2}
        target_positions = {}
        for name, pos in zip(js.name, js.position):
            target_positions[name] = pos + joint_delta_map.get(name, 0.0)

        joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        goal_constraints = Constraints()
        for joint_name in joint_names:
            if joint_name not in target_positions:
                self.get_logger().error(f"Missing joint in current state: {joint_name}")
                return False
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(target_positions[joint_name])
            jc.tolerance_above = 0.03
            jc.tolerance_below = 0.03
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)

        request = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = "ur_manipulator"
        mpr.allowed_planning_time = 5.0
        mpr.num_planning_attempts = 3
        mpr.start_state = start_state
        mpr.goal_constraints.append(goal_constraints)
        request.motion_plan_request = mpr

        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Planning failed or timed out")
            return False
        resp = future.result()
        if resp.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"Planning error code: {resp.motion_plan_response.error_code.val}")
            return False
        traj = resp.motion_plan_response.trajectory
        if not traj.joint_trajectory.points:
            self.get_logger().error("Empty trajectory in response")
            return False

        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = traj
        send_future = self.execute_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=8.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Execute goal not accepted")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=12.0)
        res = result_future.result()
        return bool(res and res.result.error_code.val == MoveItErrorCodes.SUCCESS)


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = MoveItPlanExecute()
    try:
        ok = node.plan_and_execute()
        print(f"moveit_plan_execute_ok={ok}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)


