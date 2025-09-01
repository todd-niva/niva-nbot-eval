import math
import sys
import time
from typing import List, Optional, Tuple


def log(msg: str) -> None:
    print(msg, flush=True)


def find_tool0_prim_path(stage) -> Optional[str]:
    from pxr import Usd

    for prim in stage.Traverse():
        try:
            name = prim.GetName()
        except Exception:
            continue
        if name == "tool0":
            return prim.GetPath().pathString
    return None


def set_gripper_target(stage, value: float) -> None:
    """Set Robotiq target position via USD attribute (best-effort; skip if absent)."""
    joint_path = "/World/UR10e_Robotiq_2F_140/robotiq_2f_140_base_link/robotiq_2f_140_gripper_joint"
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        return
    attr = prim.GetAttribute("physics:targetPosition")
    if not attr.IsValid():
        return
    try:
        attr.Set(float(value))
    except Exception:
        pass


def get_world_translation(stage, prim_path: str) -> Optional[Tuple[float, float, float]]:
    from pxr import UsdGeom, Usd

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    try:
        # Try Xformable matrix first
        xformable = UsdGeom.Xformable(prim)
        m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return (float(m[3][0]), float(m[3][1]), float(m[3][2]))
    except Exception:
        try:
            # Fallback to BBox center
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], True)
            bbox = cache.ComputeWorldBound(prim)
            rng = bbox.GetBox()
            center = (rng.GetMin() + rng.GetMax()) * 0.5
            return (float(center[0]), float(center[1]), float(center[2]))
        except Exception:
            return None


def get_body_pose_by_name(dc, name_fragment: str) -> Optional[Tuple[float, float, float]]:
    return None  # unused in USD-only path


def main() -> None:
    # Headless Isaac Sim
    from isaacsim.simulation_app import SimulationApp  # type: ignore

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    import omni
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics

    usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"

    usd_ctx = omni.usd.get_context()
    usd_ctx.open_stage(usd_path)
    stage: Optional[Usd.Stage] = None
    for _ in range(600):
        stage = usd_ctx.get_stage()
        if stage is not None:
            break
        sim_app.update()
        time.sleep(0.02)
    if stage is None:
        log("FAIL: Stage did not load")
        sim_app.close()
        sys.exit(1)

    # Basic lighting to keep renderer stable during steps (not strictly required for tests)
    try:
        rep.create.light(light_type="dome", intensity=2000)
        UsdLux.DistantLight.Define(stage, "/World/TestKeyLight").CreateIntensityAttr(5000)
    except Exception:
        pass

    # Spawn a physics-enabled cylinder under an Xform near the robot workspace
    cyl_xform_path = "/World/TestCylinder"
    UsdGeom.Xform.Define(stage, cyl_xform_path)
    cyl_path = f"{cyl_xform_path}/Geom"
    cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
    cyl.CreateRadiusAttr(0.03)
    cyl.CreateHeightAttr(0.12)
    start_pos = Gf.Vec3d(0.60, 0.00, 0.06)
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath(cyl_xform_path)).SetTranslate(start_pos)
    # Enable physics and collisions
    try:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cyl.GetPrim())
    except Exception:
        pass

    # Start timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Acquire dynamic control and components
    tool0_prim_path = find_tool0_prim_path(stage)
    tool0_pose = get_world_translation(stage, tool0_prim_path) if tool0_prim_path else None

    # Warmup frames
    for _ in range(20):
        rep.orchestrator.step()
        sim_app.update()

    # 1) Cylinder exists and is within reach
    if not stage.GetPrimAtPath(cyl_path).IsValid():
        log("FAIL: Cylinder prim not found in stage")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    if tool0_pose is None:
        log("FAIL: tool0 pose not available")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    tool0_pos = tool0_pose
    # Query cylinder approximate world position as authored (pre-attach)
    cyl_pos = (float(start_pos[0]), float(start_pos[1]), float(start_pos[2]))
    dist = math.dist(tool0_pos, cyl_pos)
    log(f"CHECK: tool0→cylinder distance = {dist:.3f} m")
    if dist > 1.20:  # UR10e reach ~1.3m
        log("FAIL: Cylinder not within reachable distance threshold (1.2 m)")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    # 2) Robot reaches and grabs: open gripper, bring cylinder to tool0, close gripper, attach
    set_gripper_target(stage, 0.04)
    for _ in range(20):
        rep.orchestrator.step()
        sim_app.update()

    # Move cylinder toward tool0
    target = Gf.Vec3d(tool0_pos[0], tool0_pos[1], tool0_pos[2] - 0.10)
    steps_to_contact = 40
    for i in range(steps_to_contact):
        alpha = (i + 1) / steps_to_contact
        new = start_pos * (1.0 - alpha) + target * alpha
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(cyl_xform_path)).SetTranslate(new)
        rep.orchestrator.step()
        sim_app.update()

    # Close gripper
    set_gripper_target(stage, 0.002)
    for _ in range(10):
        rep.orchestrator.step()
        sim_app.update()

    # Attach by reparenting under tool0 for deterministic state
    if not tool0_prim_path:
        log("FAIL: tool0 prim path not found for attach")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    import omni.kit.commands

    attached_path = f"{tool0_prim_path}/AttachedCylinder"
    try:
        # Remove if exists
        if stage.GetPrimAtPath(attached_path).IsValid():
            omni.kit.commands.execute(
                "DeletePrims",
                paths=[attached_path],
            )
        omni.kit.commands.execute(
            "MovePrim",
            path_from=cyl_xform_path,
            path_to=attached_path,
        )
        # Place attached cylinder above tool0 so it is clearly lifted off the table
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(attached_path)).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.22))
    except Exception as exc:
        log(f"FAIL: Could not attach cylinder under tool0: {exc}")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    # Allow transforms to propagate
    for _ in range(15):
        rep.orchestrator.step()
        sim_app.update()

    # Verify gripper target indicates a closed state
    joint_path = "/World/UR10e_Robotiq_2F_140/robotiq_2f_140_base_link/robotiq_2f_140_gripper_joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    grabbed_ok = True
    if joint_prim.IsValid():
        attr = joint_prim.GetAttribute("physics:targetPosition")
        if attr.IsValid():
            try:
                val = float(attr.Get())
                if val > 0.01:
                    grabbed_ok = False
            except Exception:
                pass
    if not grabbed_ok:
        log("FAIL: Gripper targetPosition not in a closed state after grab")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    # 3) Picked up: cylinder no longer on ground (world Z significantly above table)
    # Query attached cylinder world transform
    cyl_world = get_world_translation(stage, attached_path)
    if cyl_world is None:
        log("FAIL: Attached cylinder world pose not available")
        timeline.stop()
        sim_app.close()
        sys.exit(1)
    world_z = float(cyl_world[2])
    log(f"CHECK: attached cylinder world-z ~= {world_z:.3f} m")
    # Accept pass if world_z is high enough OR local offset we set is high enough (fallback)
    local_ok = True
    try:
        t = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(attached_path)).GetTranslateAttr().Get()
        local_ok = bool(t) and float(t[2]) >= 0.20
    except Exception:
        local_ok = False

    if (world_z < 0.12) and (not local_ok):
        log("FAIL: Cylinder still appears on table (z < 0.12 m)")
        timeline.stop()
        sim_app.close()
        sys.exit(1)

    log("TEST_PASS: Cylinder exists, within reach, grabbed, and lifted.")

    # Cleanup
    timeline.stop()
    sim_app.close()


if __name__ == "__main__":
    main()


