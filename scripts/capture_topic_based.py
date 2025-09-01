import os
import time


def main() -> None:
    # Configure Isaac Sim headless renderer
    try:
        # Isaac Sim 4.5+ preferred import
        from isaacsim.simulation_app import SimulationApp  # type: ignore
    except Exception:
        # Fallback for older builds
        from omni.isaac.kit import SimulationApp  # type: ignore

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
    output_dir = "/ros2_ws/output/topic_based_capture1"
    os.makedirs(output_dir, exist_ok=True)

    # Open the stage and wait until fully loaded
    usd_ctx = omni.usd.get_context()
    usd_ctx.open_stage(usd_path)
    stage = None
    for _ in range(300):
        stage = usd_ctx.get_stage()
        if stage is not None:
            break
        sim_app.update()
        time.sleep(0.02)

    # Add stable lighting to avoid blown-out or black captures
    try:
        rep.create.light(light_type="dome", intensity=2000)
    except Exception:
        pass

    key = UsdLux.DistantLight.Define(stage, "/World/CaptureKeyLight")
    key.CreateIntensityAttr(5000)

    # Ensure a camera exists and is well-positioned
    cam_path = "/World/CaptureCamera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    # Previous camera that produced a valid image
    cam.AddTranslateOp().Set(Gf.Vec3d(1.8, 0.9, 1.7))
    cam.AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 0.0, -30.0))

    # Bind render product and writer
    render_product = rep.create.render_product(cam_path, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True)
    writer.attach([render_product])

    # Ensure orchestrator knows which product to render
    try:
        rep.orchestrator.set_render_product(render_product)
    except Exception:
        pass

    # Advance frames explicitly to flush a stable capture
    for _ in range(30):
        rep.orchestrator.step()
        sim_app.update()
    time.sleep(0.5)

    sim_app.close()


if __name__ == "__main__":
    main()


