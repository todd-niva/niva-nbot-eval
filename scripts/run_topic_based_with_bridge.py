import time


def main() -> None:
    # Start Isaac Sim headless
    from isaacsim.simulation_app import SimulationApp  # preferred over omni.isaac.kit

    sim_app = SimulationApp({
        "headless": True,
        "renderer": "RayTracedLighting",
    })

    import omni
    import carb
    from pxr import Usd

    # Enable ROS 2 bridge extension
    app = omni.kit.app.get_app()
    ext_mgr = app.get_extension_manager()
    # Explicitly disable deprecated bridge to avoid missing dependency on 'isaacsim.ros2_bridge'
    try:
        if ext_mgr.is_extension_enabled("omni.isaac.ros2_bridge"):
            ext_mgr.set_extension_enabled_immediate("omni.isaac.ros2_bridge", False)
            carb.log_info("Disabled deprecated extension: omni.isaac.ros2_bridge")
    except Exception:
        pass
    for ext_name in ("isaacsim.ros2.bridge",):
        try:
            if not ext_mgr.is_extension_enabled(ext_name):
                ext_mgr.set_extension_enabled_immediate(ext_name, True)
                carb.log_info(f"Enabled extension: {ext_name}")
            else:
                carb.log_info(f"Extension already enabled: {ext_name}")
        except Exception as exc:
            carb.log_warn(f"Could not enable extension {ext_name}: {exc}")

    try:
        enabled_exts = list(getattr(ext_mgr, "get_enabled_extension_ids", lambda: [])())
        summary = ", ".join([e for e in enabled_exts if "ros2" in e or "ros" in e])
        carb.log_info(f"Enabled extensions (ros*): {summary}")
    except Exception:
        pass

    # Give extensions time to initialize (up to ~20s)
    for _ in range(1000):
        app.update()
        time.sleep(0.02)

    # Open the topic-based USD stage
    stage_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    ctx = omni.usd.get_context()
    ctx.open_stage(stage_path)
    stage: Usd.Stage | None = None
    for _ in range(300):
        stage = ctx.get_stage()
        if stage is not None:
            carb.log_info("USD stage is open and ready")
            break
        app.update()
        time.sleep(0.02)

    # Start playing to process graphs and bridge nodes
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Keep the app running to service ROS 2 bridge; 600 seconds (~10 minutes)
    # This script is intended to be run in background inside the container.
    start = time.time()
    while time.time() - start < 600:
        # Sleep in small increments to allow clean shutdown on container stop
        time.sleep(0.5)

    timeline.stop()
    sim_app.close()


if __name__ == "__main__":
    main()



