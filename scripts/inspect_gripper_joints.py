from isaacsim.simulation_app import SimulationApp

sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import omni
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
import time

create_new_stage()
robot_usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
add_reference_to_stage(robot_usd_path, "/World/UR10e_Robotiq_2F_140")

usd_ctx = omni.usd.get_context()
stage = None
for _ in range(300):
    stage = usd_ctx.get_stage()
    if stage is not None:
        break
    sim_app.update()
    time.sleep(0.02)

if stage:
    print("=== All gripper/finger related prims ===")
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if 'gripper' in name or 'finger' in name:
            print(f"Found: {prim.GetPath().pathString} - {prim.GetTypeName()}")
    
    print("\n=== All joints ===")
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsRevoluteJoint" or prim.GetTypeName() == "PhysicsPrismaticJoint":
            print(f"Joint: {prim.GetPath().pathString} - {prim.GetTypeName()}")

sim_app.close()
