import os
import time
import math


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
    from pxr import UsdGeom, UsdLux, Gf, Usd, UsdPhysics

    usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    output_dir = "/ros2_ws/output/topic_based_video3"
    os.makedirs(output_dir, exist_ok=True)

    # Open stage
    usd_ctx = omni.usd.get_context()
    usd_ctx.open_stage(usd_path)
    stage: Usd.Stage | None = None
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
    key = UsdLux.DistantLight.Define(stage, "/World/VideoKeyLight")
    key.CreateIntensityAttr(6000)

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
    cam.AddTranslateOp().Set(Gf.Vec3d(4.0, 2.0, 2.6))
    cam.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, -28.0))

    # Render product & BasicWriter (frame sequence)
    render_product = rep.create.render_product(cam_path, (1920, 1080))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True)
    writer.attach([render_product])

    # Prepare dynamic control to animate gripper DOFs if available
    dc = None
    gripper_dofs = []
    try:
        import omni.isaac.dynamic_control as dynamic_control
        dc = dynamic_control.acquire_dynamic_control_interface()

        # Find articulation roots in the stage
        articulation_roots: list[Usd.Prim] = []
        for prim in stage.Traverse():
            if UsdPhysics.ArticulationRootAPI(prim):
                if UsdPhysics.ArticulationRootAPI(prim).IsApplied():
                    articulation_roots.append(prim)

        # Collect DOFs that likely belong to the Robotiq gripper
        for root in articulation_roots:
            art = dc.get_articulation(root.GetPath().pathString)
            if not art:
                continue
            dof_count = dc.get_articulation_dof_count(art)
            for i in range(dof_count):
                dof = dc.get_articulation_dof(art, i)
                name = dc.get_dof_name(dof) or ""
                lname = name.lower()
                if "robotiq" in lname or "grip" in lname or "finger" in lname:
                    gripper_dofs.append(dof)
    except Exception:
        gripper_dofs = []

    # Start physics/timeline so DC updates are applied
    try:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
    except Exception:
        pass

    # Warm up
    for _ in range(20):
        rep.orchestrator.step()
        sim_app.update()

    # Animate gripper open/close over ~9 seconds
    total_frames = 270  # ~9 seconds at ~30fps
    t = 0.0
    for frame in range(total_frames):
        # Simple sinusoidal open/close between 0.0 and ~0.015 m (or radian equivalent)
        val = 0.0075 * (1.0 - math.cos(2.0 * math.pi * (frame / total_frames)))  # 0..0.015
        if dc and gripper_dofs:
            for dof in gripper_dofs:
                try:
                    # Try setting position directly; if joint is prismatic this opens/close linearly
                    dc.set_dof_position(dof, float(val))
                except Exception:
                    pass
        rep.orchestrator.step()
        sim_app.update()

    try:
        timeline.stop()
    except Exception:
        pass
    sim_app.close()


if __name__ == "__main__":
    main()


