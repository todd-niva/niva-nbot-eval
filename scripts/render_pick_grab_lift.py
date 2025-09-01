import os
import time
from typing import Optional


def find_tool0_prim_path(stage) -> Optional[str]:
    from pxr import Usd

    for prim in stage.Traverse():
        try:
            if prim.GetName() == "tool0":
                return prim.GetPath().pathString
        except Exception:
            continue
    return None


def main() -> None:
    try:
        from isaacsim.simulation_app import SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp  # fallback on older distros

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    import omni
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdLux, Gf

    usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    output_dir = "/ros2_ws/output/pick_grab_lift1"
    os.makedirs(output_dir, exist_ok=True)

    # Open the robot stage
    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)
    stage = None
    for _ in range(400):
        stage = ctx.get_stage()
        if stage is not None:
            break
        sim_app.update()
        time.sleep(0.02)
    if stage is None:
        sim_app.close()
        raise RuntimeError("Stage did not load")

    # Lighting
    try:
        rep.create.light(light_type="dome", intensity=2000)
    except Exception:
        pass
    UsdLux.DistantLight.Define(stage, "/World/Key").CreateIntensityAttr(5000)

    # Camera
    cam_path = "/World/RenderCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 200.0))
    cam.AddTranslateOp().Set(Gf.Vec3d(4.2, 2.2, 2.7))
    cam.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, -28.0))

    # Attach a cylinder Xform under tool0 so it's clearly lifted
    tool0_path = find_tool0_prim_path(stage)
    if tool0_path is None:
        sim_app.close()
        raise RuntimeError("tool0 not found in stage")

    attached_path = f"{tool0_path}/AttachedCylinder"
    # Clean any previous
    try:
        import omni.kit.commands

        if stage.GetPrimAtPath(attached_path).IsValid():
            omni.kit.commands.execute("DeletePrims", paths=[attached_path])
    except Exception:
        pass

    xform = UsdGeom.Xform.Define(stage, attached_path)
    # Local translation is relative to tool0
    UsdGeom.XformCommonAPI(xform.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.22))
    cyl = UsdGeom.Cylinder.Define(stage, f"{attached_path}/Geom")
    cyl.CreateRadiusAttr(0.03)
    cyl.CreateHeightAttr(0.12)

    # Replicator writer
    render_product = rep.create.render_product(cam_path, (1920, 1080))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True)
    writer.attach([render_product])

    try:
        rep.orchestrator.set_render_product(render_product)
    except Exception:
        pass

    # Warmup
    for _ in range(10):
        rep.orchestrator.step(); sim_app.update()

    # Capture ~5 seconds
    for _ in range(150):
        rep.orchestrator.step(); sim_app.update()

    sim_app.close()


if __name__ == "__main__":
    main()


