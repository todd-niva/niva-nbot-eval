import os
import time


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
    from pxr import UsdGeom, UsdLux, Gf

    usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    output_dir = "/ros2_ws/output/topic_based_video1"
    os.makedirs(output_dir, exist_ok=True)

    # Open stage
    usd_ctx = omni.usd.get_context()
    usd_ctx.open_stage(usd_path)
    stage = None
    for _ in range(400):
        stage = usd_ctx.get_stage()
        if stage is not None:
            break
        sim_app.update()
        time.sleep(0.02)

    # Lighting for consistent exposure
    try:
        rep.create.light(light_type="dome", intensity=2500)
    except Exception:
        pass
    key = UsdLux.DistantLight.Define(stage, "/World/VideoKeyLight")
    key.CreateIntensityAttr(6000)

    # Camera with wider FOV and pulled-back framing to capture full robot
    cam_path = "/World/VideoCamera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 200.0))
    # Use full-frame aperture and a short focal length for a much wider view
    try:
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateVerticalApertureAttr(24.0)
        cam.CreateFocalLengthAttr(18.0)
    except Exception:
        pass
    # Pull back and raise camera; angle down and yaw to frame whole scene
    cam.AddTranslateOp().Set(Gf.Vec3d(4.0, 2.0, 2.6))
    cam.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, -28.0))

    # Render product & BasicWriter (frame sequence)
    render_product = rep.create.render_product(cam_path, (1920, 1080))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True)
    writer.attach([render_product])

    # Let sim warm up a little
    for _ in range(30):
        rep.orchestrator.step()
        sim_app.update()

    # Record ~8 seconds at ~30 fps (~240 frames)
    frames = 240
    for _ in range(frames):
        rep.orchestrator.step()
        sim_app.update()

    sim_app.close()


if __name__ == "__main__":
    main()


