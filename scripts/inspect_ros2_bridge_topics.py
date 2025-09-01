import time
from typing import Optional


def main() -> None:
    from isaacsim.simulation_app import SimulationApp  # type: ignore

    sim = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    import omni
    from pxr import Usd
    import omni.graph.core as og

    # Ensure ROS 2 bridge extensions are enabled so node types resolve
    try:
        app = omni.kit.app.get_app()
        ext_mgr = app.get_extension_manager()
        for ext_name in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"):
            try:
                if not ext_mgr.is_extension_enabled(ext_name):
                    ext_mgr.set_extension_enabled_immediate(ext_name, True)
            except Exception:
                pass
        # Let extensions initialize
        for _ in range(150):
            app.update(); time.sleep(0.02)
    except Exception:
        pass

    usd_path = "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)

    stage: Optional[Usd.Stage] = None
    for _ in range(600):
        stage = ctx.get_stage()
        if stage is not None:
            break
        sim.update(); time.sleep(0.02)
    if stage is None:
        print("ERROR: Stage failed to load", flush=True)
        sim.close(); return

    # Gather all graphs and ROS2 nodes
    try:
        graphs = og.get_all_graphs()
    except Exception as exc:
        print(f"ERROR: Could not list OmniGraphs: {exc}", flush=True)
        sim.close(); return

    print("FOUND_GRAPHS:", len(graphs), flush=True)

    for graph in graphs:
        try:
            graph_path = graph.get_path()
        except Exception:
            graph_path = "<unknown>"
        print(f"GRAPH: {graph_path}", flush=True)
        try:
            node_iter = graph.get_nodes()
        except Exception:
            continue
        for node in node_iter:
            try:
                node_path = node.get_prim_path()
                type_name = node.get_type_name()
            except Exception:
                continue
            if "ROS2" not in type_name and "ArticulationController" not in type_name and "Action" not in type_name:
                continue

            print(f" NODE: {node_path} TYPE: {type_name}", flush=True)
            # Print known topic attributes if present
            for attr_name in ("inputs:topicName", "inputs:topic", "outputs:topicName"):
                try:
                    attr = node.get_attribute(attr_name)
                    if attr is not None:
                        val = attr.get_value()
                        print(f"  {attr_name} = {val}", flush=True)
                except Exception:
                    pass
            # For articulation controller/topic mappers also print joint data if available
            for extra in ("inputs:jointNames", "inputs:jointPaths", "inputs:enablePosition", "inputs:enableVelocity"):
                try:
                    attr = node.get_attribute(extra)
                    if attr is not None:
                        val = attr.get_value()
                        print(f"  {extra} = {val}", flush=True)
                except Exception:
                    pass

    sim.close()


if __name__ == "__main__":
    main()


