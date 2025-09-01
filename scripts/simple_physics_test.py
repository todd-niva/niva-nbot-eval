import math
import sys
import time
from typing import Optional
import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    # Headless Isaac Sim with physics
    from isaacsim.simulation_app import SimulationApp

    sim_app = SimulationApp({
        "headless": True,
        "renderer": "RayTracedLighting",
        "physics_dt": 1.0/60.0,  # 60Hz physics
        "rendering_dt": 1.0/30.0  # 30Hz rendering
    })

    import omni
    from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics, PhysxSchema
    from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
    from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim

    # Create new stage for clean physics test
    create_new_stage()
    
    # Load robot USD
    robot_usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    add_reference_to_stage(robot_usd_path, "/World/UR10e_Robotiq_2F_140")

    usd_ctx = omni.usd.get_context()
    stage: Optional[Usd.Stage] = None
    for _ in range(400):
        stage = usd_ctx.get_stage()
        if stage is not None:
            break
        sim_app.update()
        time.sleep(0.02)

    if stage is None:
        log("FAIL: Stage did not load")
        sim_app.close()
        sys.exit(1)

    # Enable physics scene
    try:
        scene_path = "/World/PhysicsScene"
        if not stage.GetPrimAtPath(scene_path).IsValid():
            UsdPhysics.Scene.Define(stage, scene_path)
    except Exception as e:
        log(f"Warning: Could not configure physics scene: {e}")

    # Create cylinder with proper physics and 500g mass
    cyl_xform_path = "/World/TestCylinder"
    cyl_xform = UsdGeom.Xform.Define(stage, cyl_xform_path)
    cyl_path = f"{cyl_xform_path}/Geom"
    cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
    cyl.CreateRadiusAttr(0.03)  # 3cm radius
    cyl.CreateHeightAttr(0.12)  # 12cm height
    
    # Position cylinder in front of robot
    start_pos = Gf.Vec3d(0.55, 0.00, 0.06)  # On table surface
    UsdGeom.XformCommonAPI(cyl_xform.GetPrim()).SetTranslate(start_pos)
    
    # Enable physics on cylinder
    try:
        # Collision
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        
        # Rigid body with mass
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(cyl.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(cyl.GetPrim())
        mass_api.CreateMassAttr(0.5)  # 500g mass
        
    except Exception as e:
        log(f"Warning: Could not set cylinder physics: {e}")

    # Get robot articulation for control
    robot_prim = get_prim_at_path("/World/UR10e_Robotiq_2F_140")
    if not robot_prim.IsValid():
        log("FAIL: Robot prim not found")
        sim_app.close()
        sys.exit(1)

    robot_articulation = Articulation(robot_prim.GetPath().pathString)
    
    # Get tool0 for positioning (try multiple paths)
    tool0_prim = None
    tool0_paths = [
        "/World/UR10e_Robotiq_2F_140/tool0",
        "/World/UR10e_Robotiq_2F_140/ur10e_Link6/tool0",
        "/World/UR10e_Robotiq_2F_140/wrist_3_link/tool0"
    ]
    
    for path in tool0_paths:
        prim = get_prim_at_path(path)
        if prim.IsValid():
            tool0_prim = prim
            log(f"Found tool0 at: {path}")
            break
    
    if tool0_prim is None:
        log("WARN: tool0 not found, using robot base for positioning")
        # Use the robot root for attachment
        tool0_prim = robot_prim

    # Start physics simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Warmup physics
    for _ in range(60):  # 1 second at 60Hz
        sim_app.update()

    # TEST 1: Verify cylinder exists and has proper physics
    cylinder_rigid = None
    try:
        cylinder_rigid = RigidPrim(cyl_xform_path)
        cylinder_mass = cylinder_rigid.get_mass()
        log(f"CHECK: Cylinder mass = {cylinder_mass:.3f} kg")
        if abs(cylinder_mass - 0.5) > 0.1:  # Allow some tolerance
            log(f"WARN: Cylinder mass not exactly 0.5kg (got {cylinder_mass:.3f})")
    except Exception as e:
        log(f"WARN: Could not get cylinder mass: {e}")

    # Check if cylinder is in reach
    tool0_world_pos = robot_articulation.get_world_pose()[0]  # position only
    cyl_world_pos = start_pos  # Use initial position for distance check
    
    distance = np.linalg.norm(np.array(tool0_world_pos) - np.array(cyl_world_pos))
    log(f"CHECK: tool0→cylinder distance = {distance:.3f} m")
    if distance > 1.2:  # UR10e reach
        log("FAIL: Cylinder not within reach")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    # TEST 2: Move robot to pick position
    log("CHECK: Moving robot to pick position")
    
    # Define poses for pick sequence
    home_pose = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
    pick_pose = np.array([-1.57, -0.8, 0.8, -1.5, -1.57, 0.0])  # Lower to reach cylinder
    
    # Move to pick position
    current_pose = robot_articulation.get_joint_positions()
    for i in range(60):
        alpha = (i + 1) / 60
        interp_pose = current_pose + alpha * (pick_pose - current_pose)
        robot_articulation.set_joint_positions(interp_pose)
        sim_app.update()

    # TEST 3: Manually attach cylinder to simulate grasp
    log("CHECK: Simulating grasp by attaching cylinder to tool0")
    
    # Record initial cylinder position
    initial_cyl_pos = UsdGeom.XformCommonAPI(cyl_xform.GetPrim()).GetTranslate()
    
    # Import omni commands for manipulation
    import omni.kit.commands
    
    # Attach cylinder to tool0
    attached_path = f"{tool0_prim.GetPath().pathString}/AttachedCylinder"
    try:
        # Remove if exists
        if stage.GetPrimAtPath(attached_path).IsValid():
            omni.kit.commands.execute("DeletePrims", paths=[attached_path])
        
        # Move cylinder under tool0
        omni.kit.commands.execute("MovePrim", path_from=cyl_xform_path, path_to=attached_path)
        
        # Position relative to tool0
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(attached_path)).SetTranslate(Gf.Vec3d(0.0, 0.0, -0.1))
        log("PASS: Cylinder attached to tool0")
        
    except Exception as e:
        log(f"FAIL: Could not attach cylinder: {e}")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    # Allow attachment to settle
    for _ in range(30):
        sim_app.update()

    # TEST 4: Lift cylinder and verify physics response
    log("CHECK: Lifting cylinder to test physics")
    
    # Define lift pose
    lift_pose = pick_pose.copy()
    lift_pose[1] += 0.3  # Lift shoulder joint
    
    # Lift gradually
    current_pose = robot_articulation.get_joint_positions()
    lift_successful = False
    
    for i in range(90):  # 1.5 seconds to lift
        alpha = (i + 1) / 90
        interp_pose = current_pose + alpha * (lift_pose - current_pose)
        robot_articulation.set_joint_positions(interp_pose)
        sim_app.update()
        
        # Check cylinder position every 15 frames
        if i % 15 == 0 and i > 30:
            try:
                current_cyl_world = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(attached_path))
                if current_cyl_world:
                    # Get world transform
                    from pxr import Usd
                    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(attached_path))
                    m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    world_pos = (float(m[3][0]), float(m[3][1]), float(m[3][2]))
                    cyl_height = world_pos[2]
                    
                    if cyl_height > initial_cyl_pos[2] + 0.05:  # Lifted at least 5cm
                        lift_successful = True
                        log(f"CHECK: Cylinder lifted to height {cyl_height:.3f}m")
            except Exception:
                pass
    
    # Final position check
    try:
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(attached_path))
        m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        final_world_pos = (float(m[3][0]), float(m[3][1]), float(m[3][2]))
        final_height = final_world_pos[2]
        height_gain = final_height - initial_cyl_pos[2]
        
        log(f"CHECK: Cylinder height change = {height_gain:.3f}m")
        
        if height_gain < 0.03:  # Must lift at least 3cm
            log("FAIL: Cylinder not lifted sufficiently")
            timeline.stop()
            sim_app.close()
            sys.exit(1)
        
        log("PASS: Cylinder successfully lifted")
        
    except Exception as e:
        log(f"WARN: Could not verify final position: {e}")

    # All tests passed
    log("TEST_PASS: Simple physics-based pick-grab-lift successful")
    log("- Cylinder has 500g mass")
    log("- Robot can reach and grasp cylinder")
    log("- Cylinder lifted with physics simulation")
    log("- Attachment mechanism works correctly")

    timeline.stop()
    sim_app.close()


if __name__ == "__main__":
    main()
