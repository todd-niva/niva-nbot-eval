import os
import time
import math
from typing import Optional


def main() -> None:
    # Headless Isaac Sim
    from isaacsim.simulation_app import SimulationApp  # type: ignore

    sim_app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting",
        }
    )

    import omni
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, UsdLux, Gf

    usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    output_dir = "/ros2_ws/output/topic_based_pickplace1"
    os.makedirs(output_dir, exist_ok=True)

    # Open stage
    usd_ctx = omni.usd.get_context()
    usd_ctx.open_stage(usd_path)
    stage: Optional[Usd.Stage] = None
    for _ in range(400):
        stage = usd_ctx.get_stage()
        if stage is not None:
            break
        sim_app.update()
        time.sleep(0.02)

    # Lighting
    try:
        rep.create.light(light_type="dome", intensity=2500)
    except Exception:
        pass
    key = UsdLux.DistantLight.Define(stage, "/World/PickPlaceKeyLight")
    key.CreateIntensityAttr(6500)

    # Camera (wide view)
    cam_path = "/World/VideoCamera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 200.0))
    try:
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateVerticalApertureAttr(24.0)
        cam.CreateFocalLengthAttr(18.0)
    except Exception:
        pass
    cam.AddTranslateOp().Set(Gf.Vec3d(4.2, 2.2, 2.7))
    cam.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, -28.0))

    # Cylinder prop
    cyl_path = "/World/PropCylinder"
    cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
    cyl.CreateRadiusAttr(0.03)
    cyl.CreateHeightAttr(0.12)
    cyl_xform = UsdGeom.Xformable(cyl.GetPrim())
    # Place near robot workspace (adjust as needed)
    cyl_xform.AddTranslateOp().Set(Gf.Vec3d(0.6, 0.0, 0.06))

    # Render product & BasicWriter (frame sequence)
    render_product = rep.create.render_product(cam_path, (1920, 1080))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True)
    writer.attach([render_product])

    # Dynamic control for gripper joints and end-effector link pose
    dc = None
    gripper_dofs = []
    eef_body = None
    try:
        import omni.isaac.dynamic_control as dynamic_control
        dc = dynamic_control.acquire_dynamic_control_interface()

        # Try to find a body named 'tool0' or something with 'gripper'/'robotiq'
        candidates = ["tool0", "robotiq", "gripper", "robotiq_140_base_link"]
        scene = dc.get_physics_scene()
        if scene:
            # brute-force over bodies
            for i in range(dc.get_bodies_count()):
                b = dc.get_body(i)
                name = (dc.get_body_name(b) or "").lower()
                if any(c in name for c in candidates):
                    eef_body = b
                    break

        # Try to collect gripper DOFs as before
        if scene:
            for i in range(dc.get_articulations_count()):
                art = dc.get_articulation(i)
                dof_count = dc.get_articulation_dof_count(art)
                for j in range(dof_count):
                    dof = dc.get_articulation_dof(art, j)
                    name = dc.get_dof_name(dof) or ""
                    lname = name.lower()
                    if "robotiq" in lname or "grip" in lname or "finger" in lname:
                        gripper_dofs.append(dof)
    except Exception:
        pass

    # Start timeline
    try:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
    except Exception:
        pass

    # Phase timings
    total_frames = 300
    open_frames = 40
    pre_attach_frames = 60
    carry_frames = 120
    place_frames = 60
    return_frames = 20

    # Targets
    place_target = Gf.Vec3d(0.4, 0.3, 0.06)
    start_pos = Gf.Vec3d(0.6, 0.0, 0.06)

    def set_gripper(amount: float) -> None:
        if not (dc and gripper_dofs):
            return
        for dof in gripper_dofs:
            try:
                dc.set_dof_position(dof, float(amount))
            except Exception:
                pass

    # Warm up
    for _ in range(10):
        rep.orchestrator.step()
        sim_app.update()

    # Open gripper
    for f in range(open_frames):
        set_gripper(0.015)
        rep.orchestrator.step()
        sim_app.update()

    # Move cylinder slowly toward gripper until attach
    attach_frame = open_frames + pre_attach_frames
    for f in range(pre_attach_frames):
        alpha = (f + 1) / pre_attach_frames
        # If we know eef pose, lerp cyl toward it; else small lift
        if dc and eef_body:
            pose = dc.get_rigid_body_pose(eef_body)
            target = Gf.Vec3d(pose.p.x, pose.p.y, pose.p.z)
            cur = start_pos
            new = cur * (1.0 - alpha) + target * alpha
            UsdGeom.XformCommonAPI(cyl.GetPrim()).SetTranslate(new)
        else:
            UsdGeom.XformCommonAPI(cyl.GetPrim()).SetTranslate(
                start_pos + Gf.Vec3d(0.0, 0.0, 0.002 * f)
            )
        rep.orchestrator.step()
        sim_app.update()

    # Close gripper to 'attach'
    for f in range(20):
        set_gripper(0.002)
        rep.orchestrator.step()
        sim_app.update()

    # Carry: follow eef while closed
    for f in range(carry_frames):
        if dc and eef_body:
            pose = dc.get_rigid_body_pose(eef_body)
            offset = Gf.Vec3d(0.0, 0.0, -0.10)  # slight offset below toolframe
            new = Gf.Vec3d(pose.p.x, pose.p.y, pose.p.z) + offset
            UsdGeom.XformCommonAPI(cyl.GetPrim()).SetTranslate(new)
        rep.orchestrator.step()
        sim_app.update()

    # Move to place target over place_frames
    # If eef available, ignore; otherwise lerp cylinder directly
    cur_pos = UsdGeom.XformCommonAPI(cyl.GetPrim()).GetTranslateAttr().Get()
    cur_vec = Gf.Vec3d(cur_pos[0], cur_pos[1], cur_pos[2])
    for f in range(place_frames):
        alpha = (f + 1) / place_frames
        new = cur_vec * (1.0 - alpha) + place_target * alpha
        UsdGeom.XformCommonAPI(cyl.GetPrim()).SetTranslate(new)
        rep.orchestrator.step()
        sim_app.update()

    # Open gripper to 'release'
    for f in range(20):
        set_gripper(0.015)
        rep.orchestrator.step()
        sim_app.update()

    # Return: optionally move back toward start (on table)
    pos = UsdGeom.XformCommonAPI(cyl.GetPrim()).GetTranslateAttr().Get()
    pos_vec = Gf.Vec3d(pos[0], pos[1], pos[2])
    for f in range(return_frames):
        alpha = (f + 1) / return_frames
        new = pos_vec * (1.0 - alpha) + start_pos * alpha
        UsdGeom.XformCommonAPI(cyl.GetPrim()).SetTranslate(new)
        rep.orchestrator.step()
        sim_app.update()

    try:
        timeline.stop()
    except Exception:
        pass
    sim_app.close()


if __name__ == "__main__":
    main()


