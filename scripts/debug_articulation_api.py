#!/usr/bin/env python3
"""
Debug Articulation API - Find correct methods for Isaac Sim 4.5
"""

import time

def log(msg: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def main():
    log("🔍 DEBUGGING ARTICULATION API FOR ISAAC SIM 4.5")
    
    from isaacsim.simulation_app import SimulationApp
    sim_config = {
        "headless": True,
        "renderer": "RayTracedLighting",
    }
    sim_app = SimulationApp(sim_config)
    
    import omni
    from pxr import UsdGeom, UsdPhysics
    from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation
    
    try:
        create_new_stage()
        add_reference_to_stage(
            "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd",
            "/World/UR10e_Robotiq_2F_140"
        )
        
        usd_ctx = omni.usd.get_context()
        stage = None
        for _ in range(400):
            stage = usd_ctx.get_stage()
            if stage is not None:
                break
            sim_app.update()
            time.sleep(0.02)
        
        world = World(stage_units_in_meters=1.0)
        robot_path = "/World/UR10e_Robotiq_2F_140"
        robot_articulation = SingleArticulation(robot_path)
        
        world.reset()
        for _ in range(60):
            world.step(render=False)
        
        robot_articulation.initialize()
        
        log("✅ Robot articulation initialized")
        log(f"📋 Available attributes and methods:")
        
        # Get all attributes and methods
        attrs = [attr for attr in dir(robot_articulation) if not attr.startswith('_')]
        for attr in sorted(attrs):
            try:
                value = getattr(robot_articulation, attr)
                if callable(value):
                    log(f"   {attr}() - method")
                else:
                    log(f"   {attr} - property: {type(value)}")
            except:
                log(f"   {attr} - property (error accessing)")
        
        # Test specific methods
        log("\n🧪 Testing specific methods:")
        
        # Try different joint methods
        joint_methods = [
            'joint_names', 'get_joint_names', 'dof_names', 'get_dof_names',
            'get_joint_positions', 'get_applied_joint_efforts', 'get_joint_velocities',
            'set_joint_positions', 'num_dof', 'dof_count'
        ]
        
        for method in joint_methods:
            try:
                if hasattr(robot_articulation, method):
                    attr = getattr(robot_articulation, method)
                    if callable(attr):
                        try:
                            result = attr()
                            log(f"   ✅ {method}(): {type(result)} - {len(result) if hasattr(result, '__len__') else 'N/A'}")
                        except Exception as e:
                            log(f"   ⚠️  {method}(): Error - {e}")
                    else:
                        log(f"   ✅ {method}: {type(attr)} - {len(attr) if hasattr(attr, '__len__') else 'N/A'}")
                else:
                    log(f"   ❌ {method}: Not available")
            except Exception as e:
                log(f"   ❌ {method}: Exception - {e}")
        
        log("\n🎯 DEBUGGING COMPLETE")
        
    except Exception as e:
        log(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'world' in locals():
                world.stop()
            sim_app.close()
        except Exception as e:
            log(f"Cleanup warning: {e}")

if __name__ == "__main__":
    main()
