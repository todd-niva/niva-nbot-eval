import os
import time
import math
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
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics
    from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
    from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
    from omni.isaac.core.articulations import Articulation

    # Create new stage for clean physics test
    create_new_stage()
    
    # Load robot USD
    robot_usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    add_reference_to_stage(robot_usd_path, "/World/UR10e_Robotiq_2F_140")

    usd_ctx = omni.usd.get_context()
    stage = None
    for _ in range(400):
        stage = usd_ctx.get_stage()
        if stage is not None:
            break
        sim_app.update()
        time.sleep(0.02)

    if stage is None:
        log("FAIL: Stage did not load")
        sim_app.close()
        return

    # Enable physics scene
    try:
        scene_path = "/World/PhysicsScene"
        if not stage.GetPrimAtPath(scene_path).IsValid():
            UsdPhysics.Scene.Define(stage, scene_path)
    except Exception as e:
        log(f"Warning: Could not configure physics scene: {e}")

    # Create cylinder with physics and 500g mass
    cyl_xform_path = "/World/TestCylinder"
    cyl_xform = UsdGeom.Xform.Define(stage, cyl_xform_path)
    cyl_path = f"{cyl_xform_path}/Geom"
    cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
    cyl.CreateRadiusAttr(0.03)  # 3cm radius
    cyl.CreateHeightAttr(0.12)  # 12cm height
    
    # Position cylinder in front of robot
    start_pos = Gf.Vec3d(0.55, 0.00, 0.06)  # On table surface
    UsdGeom.XformCommonAPI(cyl_xform.GetPrim()).SetTranslate(start_pos)
    
    # Add visual color
    try:
        from pxr import Vt
        UsdGeom.Gprim(cyl.GetPrim()).GetDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(0.9, 0.1, 0.1)])
        )
    except Exception:
        pass

    # Enable physics on cylinder
    try:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
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
        return

    robot_articulation = Articulation(robot_prim.GetPath().pathString)

    # Setup camera - directly above robot pointing down
    log("Setting up top-down camera")
    cam_path = "/World/TopDownCamera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    cam.CreateFocalLengthAttr(24.0)  # Wide field of view

    # Position camera directly above robot base, looking straight down
    robot_base_pos = robot_articulation.get_world_pose()[0]  # Get robot position
    camera_height = 2.5  # Height above robot
    camera_pos = Gf.Vec3d(float(robot_base_pos[0]), float(robot_base_pos[1]), float(robot_base_pos[2]) + camera_height)
    
    # Set camera position and orientation
    cam_xform = UsdGeom.XformCommonAPI(cam.GetPrim())
    cam_xform.SetTranslate(camera_pos)
    
    # Look straight down (X=-90 degrees, Y=0, Z=0 for no roll)
    cam_xform.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0))
    
    log(f"Camera positioned at: {camera_pos}")
    log("Camera orientation: Looking straight down (top-down view)")

    # Lighting setup
    try:
        rep.create.light(light_type="dome", intensity=1500)
    except Exception:
        pass
    key_light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key_light.CreateIntensityAttr(4000)
    key_light.CreateAngleAttr(0.5)

    # Setup video capture
    output_dir = "/ros2_ws/output/topdown_physics_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    render_product = rep.create.render_product(cam_path, (1920, 1080))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True)
    writer.attach([render_product])

    # Start physics simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Define robot poses for demonstration
    home_pose = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
    pre_pick_pose = np.array([-1.57, -1.0, 1.0, -1.5, -1.57, 0.0])
    pick_pose = np.array([-1.57, -0.8, 0.8, -1.5, -1.57, 0.0])
    lift_pose = np.array([-1.57, -0.5, 0.5, -1.5, -1.57, 0.0])
    place_pose = np.array([-1.57, -1.0, 1.0, -1.5, -1.57, 1.57])

    log("Starting physics demonstration with top-down capture")

    # Warmup phase
    log("Phase 1: Warmup and initial positioning")
    robot_articulation.set_joint_positions(home_pose)
    for frame in range(60):  # 2 seconds at 30fps
        rep.orchestrator.step()
        sim_app.update()

    # Move to pre-pick
    log("Phase 2: Moving to pre-pick position")
    current_pose = robot_articulation.get_joint_positions()
    for frame in range(90):  # 3 seconds
        alpha = frame / 89.0
        interp_pose = current_pose + alpha * (pre_pick_pose - current_pose)
        robot_articulation.set_joint_positions(interp_pose)
        rep.orchestrator.step()
        sim_app.update()

    # Move to pick
    log("Phase 3: Moving to pick position")
    current_pose = robot_articulation.get_joint_positions()
    for frame in range(60):  # 2 seconds
        alpha = frame / 59.0
        interp_pose = current_pose + alpha * (pick_pose - current_pose)
        robot_articulation.set_joint_positions(interp_pose)
        rep.orchestrator.step()
        sim_app.update()

    # Simulate grasp (attach cylinder)
    log("Phase 4: Grasping cylinder")
    import omni.kit.commands
    attached_path = f"{robot_prim.GetPath().pathString}/AttachedCylinder"
    try:
        if stage.GetPrimAtPath(attached_path).IsValid():
            omni.kit.commands.execute("DeletePrims", paths=[attached_path])
        omni.kit.commands.execute("MovePrim", path_from=cyl_xform_path, path_to=attached_path)
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(attached_path)).SetTranslate(Gf.Vec3d(0.0, 0.0, -0.15))
        log("Cylinder attached to robot")
    except Exception as e:
        log(f"Could not attach cylinder: {e}")

    # Hold for visibility
    for frame in range(30):  # 1 second
        rep.orchestrator.step()
        sim_app.update()

    # Lift cylinder
    log("Phase 5: Lifting cylinder")
    current_pose = robot_articulation.get_joint_positions()
    for frame in range(90):  # 3 seconds
        alpha = frame / 89.0
        interp_pose = current_pose + alpha * (lift_pose - current_pose)
        robot_articulation.set_joint_positions(interp_pose)
        rep.orchestrator.step()
        sim_app.update()

    # Move to place position
    log("Phase 6: Moving to place position")
    current_pose = robot_articulation.get_joint_positions()
    for frame in range(90):  # 3 seconds
        alpha = frame / 89.0
        interp_pose = current_pose + alpha * (place_pose - current_pose)
        robot_articulation.set_joint_positions(interp_pose)
        rep.orchestrator.step()
        sim_app.update()

    # Release cylinder
    log("Phase 7: Releasing cylinder")
    try:
        # Detach cylinder and place it at new location
        if stage.GetPrimAtPath(attached_path).IsValid():
            omni.kit.commands.execute("MovePrim", path_from=attached_path, path_to=cyl_xform_path)
        # Position at new location
        new_pos = Gf.Vec3d(0.3, 0.3, 0.06)
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(cyl_xform_path)).SetTranslate(new_pos)
        log("Cylinder released at new position")
    except Exception as e:
        log(f"Could not release cylinder: {e}")

    # Hold final position
    for frame in range(60):  # 2 seconds
        rep.orchestrator.step()
        sim_app.update()

    # Return home
    log("Phase 8: Returning to home position")
    current_pose = robot_articulation.get_joint_positions()
    for frame in range(90):  # 3 seconds
        alpha = frame / 89.0
        interp_pose = current_pose + alpha * (home_pose - current_pose)
        robot_articulation.set_joint_positions(interp_pose)
        rep.orchestrator.step()
        sim_app.update()

    log("Demonstration complete")

    # Cleanup
    timeline.stop()
    sim_app.close()

    log(f"Video frames saved to: {output_dir}")
    log("Use ffmpeg to convert frames to video:")
    log(f"ffmpeg -framerate 30 -i {output_dir}/rgb_%04d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/topdown_demo.mp4")


if __name__ == "__main__":
    main()
