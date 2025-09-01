import os
import time
from typing import Optional


def main() -> None:
    try:
        from isaacsim.simulation_app import SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp  # type: ignore

    sim_app = SimulationApp({
        "headless": True,
        "renderer": "RayTracedLighting",
    })

    import omni
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, UsdLux, Gf

    # Enable ROS 2 bridge extensions if available
    try:
        app = omni.kit.app.get_app()
        ext_mgr = app.get_extension_manager()
        for ext_name in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"):
            try:
                if not ext_mgr.is_extension_enabled(ext_name):
                    ext_mgr.set_extension_enabled_immediate(ext_name, True)
            except Exception:
                pass
        for _ in range(150):
            app.update(); time.sleep(0.02)
    except Exception:
        pass

    usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    out_dir = os.environ.get("ROS_BRIDGE_CAPTURE_OUT", "/ros2_ws/output/ros_bridge_capture1")
    os.makedirs(out_dir, exist_ok=True)

    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)

    stage: Optional[Usd.Stage] = None
    for _ in range(600):
        stage = ctx.get_stage()
        if stage is not None:
            break
        sim_app.update(); time.sleep(0.02)
    if stage is None:
        print("ERROR: Stage did not load", flush=True)
        sim_app.close(); return

    # Lighting and camera
    try:
        rep.create.light(light_type="dome", intensity=2000)
    except Exception:
        pass
    UsdLux.DistantLight.Define(stage, "/World/Key").CreateIntensityAttr(5000)

    cam_path = "/World/ROSBridgeCaptureCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 200.0))
    # Ground-level camera: looking horizontally at robot base
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], True)
        robot_prim = stage.GetPrimAtPath("/World/UR10e_Robotiq_2F_140")
        if not robot_prim.IsValid():
            robot_prim = stage.GetPrimAtPath("/World")
        wb = bbox_cache.ComputeWorldBound(robot_prim)
        box = wb.GetBox()
        center = (box.GetMin() + box.GetMax()) * 0.5
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        # Position camera at ground level, looking horizontally at robot
        camera_distance = 2.5
        camera_height = 0.8  # Slightly above ground for better view
        xform = UsdGeom.XformCommonAPI(cam.GetPrim())
        xform.SetTranslate(Gf.Vec3d(cx + camera_distance, cy - camera_distance, camera_height))
        # Look toward robot center: slight downward tilt to see base
        xform.SetRotate(Gf.Vec3f(-10.0, 0.0, 45.0))
    except Exception:
        xform = UsdGeom.XformCommonAPI(cam.GetPrim())
        xform.SetTranslate(Gf.Vec3d(2.5, -2.5, 0.8))
        xform.SetRotate(Gf.Vec3f(-10.0, 0.0, 45.0))
    try:
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateVerticalApertureAttr(24.0)
        cam.CreateFocalLengthAttr(20.0)
    except Exception:
        pass

    render_product = rep.create.render_product(cam_path, (1920, 1080))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=out_dir, rgb=True)
    writer.attach([render_product])
    try:
        rep.orchestrator.set_render_product(render_product)
    except Exception:
        pass

    # Start timeline so ROS bridge and graphs process
    try:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
    except Exception:
        timeline = None

    # Warmup and capture for ~10 seconds
    total_frames = 300
    for _ in range(total_frames):
        rep.orchestrator.step()
        sim_app.update()

    # Mark done
    try:
        with open(os.path.join(out_dir, "_done.txt"), "w", encoding="utf-8") as f:
            f.write("done\n")
    except Exception:
        pass

    try:
        if timeline:
            timeline.stop()
    except Exception:
        pass
    sim_app.close()


if __name__ == "__main__":
    main()


