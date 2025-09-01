import math
import sys
import time
from typing import List, Optional


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from sensor_msgs.msg import JointState

    JOINT_ORDER = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    class JTClient(Node):
        def __init__(self) -> None:
            super().__init__("jt_sender")
            self._js: Optional[JointState] = None
            self.create_subscription(JointState, "/joint_states", self._on_js, 10)
            self._ac = ActionClient(self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory")

        def _on_js(self, msg: JointState) -> None:
            if not msg.name or not msg.position:
                return
            self._js = msg

        def wait_for_js(self, timeout_sec: float) -> Optional[JointState]:
            end = time.time() + timeout_sec
            while rclpy.ok() and time.time() < end and self._js is None:
                rclpy.spin_once(self, timeout_sec=0.1)
            return self._js

        def wait_for_server(self, timeout_sec: float) -> bool:
            return self._ac.wait_for_server(timeout_sec=timeout_sec)

        def send_trajectory(self, names: List[str], start: List[float]) -> bool:
            delta = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
            target = [s + d for s, d in zip(start, delta)]
            jt = JointTrajectory()
            jt.joint_names = names
            pt1 = JointTrajectoryPoint(positions=target)
            pt1.time_from_start.sec = 2
            pt2 = JointTrajectoryPoint(positions=start)
            pt2.time_from_start.sec = 4
            jt.points = [pt1, pt2]
            goal = FollowJointTrajectory.Goal(trajectory=jt)
            send_future = self._ac.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
            if not send_future.done():
                self.get_logger().error("send_goal timeout")
                return False
            goal_handle = send_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("goal rejected")
                return False
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=20.0)
            return result_future.done() and result_future.result().status == 4  # STATUS_SUCCEEDED

    rclpy.init()
    node = JTClient()
    try:
        if not node.wait_for_server(timeout_sec=8.0):
            print("action_server_available=False")
            return
        js = node.wait_for_js(timeout_sec=5.0)
        if js is None:
            print("joint_states_available=False")
            return
        idx = [js.name.index(n) for n in JOINT_ORDER]
        start = [float(js.position[i]) for i in idx]
        ok = node.send_trajectory(JOINT_ORDER, start)
        print(f"trajectory_ok={ok}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error={exc}")
        sys.exit(1)


