import time
from typing import List


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class JointCommandPublisher(Node):
        def __init__(self) -> None:
            super().__init__("isaac_joint_command_publisher")
            # Isaac Sim bridge subscribers
            self.cmd_pub = self.create_publisher(JointState, "/isaac_joint_commands", 10)
            self.grip_pub = self.create_publisher(JointState, "/isaac_joint_gripper", 10)

        def publish_targets(self, arm_positions: List[float], gripper_position: float) -> None:
            now = self.get_clock().now().to_msg()

            arm = JointState()
            arm.header.stamp = now
            arm.name = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            arm.position = list(map(float, arm_positions))
            self.cmd_pub.publish(arm)

            grip = JointState()
            grip.header.stamp = now
            grip.name = ["finger_joint"]
            grip.position = [float(gripper_position)]
            self.grip_pub.publish(grip)

    rclpy.init()
    node = JointCommandPublisher()
    try:
        # Small safe offset target
        target_positions = [0.2, -1.1, 1.6, -0.4, -1.57, 0.1]
        target_grip = 0.7  # slightly open
        end_time = time.time() + 4.0
        while time.time() < end_time:
            node.publish_targets(target_positions, target_grip)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(1.0 / 30.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


