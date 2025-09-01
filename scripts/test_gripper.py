#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory


class GripperActionClient(Node):
    def __init__(self) -> None:
        super().__init__("gripper_action_client")
        self._client = ActionClient(
            self, FollowJointTrajectory,
            "/robotiq_gripper_joint_trajectory_controller/follow_joint_trajectory",
        )

    def send_position(self, position: float, timeout_sec: float = 6.0) -> bool:
        if not self._client.wait_for_server(timeout_sec=4.0):
            print("server_unavailable")
            return False
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ["finger_joint"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.time_from_start.sec = 1
        traj.points = [pt]
        goal.trajectory = traj
        goal_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=timeout_sec)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            print("goal_rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        result = result_future.result()
        print(f"result_ok={bool(result)}")
        return bool(result)


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = GripperActionClient()
    try:
        print("=== open ===")
        node.send_position(0.8)
        time.sleep(0.5)
        print("=== close ===")
        node.send_position(0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
