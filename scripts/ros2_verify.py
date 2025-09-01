import sys
import time
from typing import Optional


def main() -> None:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from controller_manager_msgs.srv import ListControllers
    from sensor_msgs.msg import JointState

    class Verifier(Node):
        def __init__(self) -> None:
            super().__init__("ros2_verifier")
            self.joint_states_received: bool = False
            self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
            self.cli = self.create_client(ListControllers, "/controller_manager/list_controllers")

        def _js_cb(self, msg: JointState) -> None:  # noqa: ARG002
            self.joint_states_received = True

        def wait_for_cm(self, timeout_sec: float) -> bool:
            return self.cli.wait_for_service(timeout_sec=timeout_sec)

        def list_controllers(self) -> Optional[ListControllers.Response]:
            req = ListControllers.Request()
            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not future.done():
                return None
            return future.result()

    rclpy.init()
    node = Verifier()
    try:
        ok = node.wait_for_cm(timeout_sec=5.0)
        print(f"cm_service_available={ok}")
        if ok:
            resp = node.list_controllers()
            if resp is None:
                print("controllers: timeout")
            else:
                names = [f"{c.name}:{c.state}" for c in resp.controller]
                print("controllers=" + ",".join(names))
        # Spin briefly to see if joint_states arrive
        start = time.time()
        while time.time() - start < 3.0 and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
        print(f"joint_states_received={node.joint_states_received}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error={exc}")
        sys.exit(1)



