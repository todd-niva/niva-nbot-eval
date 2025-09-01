import time
from typing import List


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class TrajPub(Node):
        def __init__(self) -> None:
            super().__init__("traj_pub")
            self.arm_pub = self.create_publisher(
                JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
            )
            self.grip_pub = self.create_publisher(
                JointTrajectory, "/robotiq_gripper_joint_trajectory_controller/joint_trajectory", 10
            )

        def send_arm(self, positions: List[float], seconds: float) -> None:
            msg = JointTrajectory()
            msg.joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            pt = JointTrajectoryPoint()
            pt.positions = list(map(float, positions))
            pt.time_from_start.sec = int(seconds)
            pt.time_from_start.nanosec = int((seconds - int(seconds)) * 1e9)
            msg.points = [pt]
            self.arm_pub.publish(msg)

        def send_grip(self, finger_position: float, seconds: float) -> None:
            msg = JointTrajectory()
            msg.joint_names = ["finger_joint"]
            pt = JointTrajectoryPoint()
            pt.positions = [float(finger_position)]
            pt.time_from_start.sec = int(seconds)
            pt.time_from_start.nanosec = int((seconds - int(seconds)) * 1e9)
            msg.points = [pt]
            self.grip_pub.publish(msg)

    rclpy.init()
    node = TrajPub()
    try:
        # Small motion target
        arm_goal = [0.1, -1.2, 1.5, -0.6, -1.57, 0.0]
        grip_goal = 0.6
        end = time.time() + 2.5
        while time.time() < end:
            node.send_arm(arm_goal, 2.0)
            node.send_grip(grip_goal, 1.0)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
