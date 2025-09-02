#!/usr/bin/env python3

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from isaacsim import SimulationApp

# Launch Isaac Sim with specific rendering settings to avoid white images
simulation_app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080,
    "physics_gpu": 0,
    "physics_dt": 1.0/240.0,
    "rendering_dt": 1.0/60.0,
    "renderer": "RayTracedLighting",
    "anti_aliasing": 3,
    "multi_gpu": False,
})

import sys
sys.path.append('/ros2_ws/scripts')
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import open_stage
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
from scene_complexity_manager import SceneComplexityManager
from realistic_baseline_controller import RealisticBaselineController
from realistic_baseline_evaluator import RealisticBaselineEvaluator

def create_enhanced_lighting(stage):
    """Create strong, directional lighting to avoid white images"""
    print("🔆 Setting up enhanced lighting to prevent white images")
    
    # Remove or tone down the dome light if it exists
    dome_light_path = "/World/DomeLight"
    dome_prim = stage.GetPrimAtPath(dome_light_path)
    if dome_prim.IsValid():
        dome_light = UsdLux.DomeLight(dome_prim)
        # Significantly reduce dome light intensity
        dome_light.CreateIntensityAttr(50.0)  # Very low intensity
        dome_light.CreateColorAttr(Gf.Vec3f(0.8, 0.8, 0.9))  # Subtle blue tint
        print(f"   - Reduced DomeLight intensity to 50")
    
    # Create strong directional key light
    key_light_path = "/World/DirectionalKeyLight"
    key_light = UsdLux.DirectionalLight.Define(stage, key_light_path)
    key_light.CreateIntensityAttr(2000.0)  # Strong intensity
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))  # Warm white
    key_xform = UsdGeom.Xformable(key_light)
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))  # Angled from above
    
    # Create fill light to reduce harsh shadows
    fill_light_path = "/World/DirectionalFillLight"
    fill_light = UsdLux.DirectionalLight.Define(stage, fill_light_path)
    fill_light.CreateIntensityAttr(800.0)  # Medium intensity
    fill_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))  # Cool white
    fill_xform = UsdGeom.Xformable(fill_light)
    fill_xform.AddRotateXYZOp().Set(Gf.Vec3f(-30, -60, 0))  # From opposite side
    
    # Create rim light for depth
    rim_light_path = "/World/DirectionalRimLight"
    rim_light = UsdLux.DirectionalLight.Define(stage, rim_light_path)
    rim_light.CreateIntensityAttr(600.0)  # Lower intensity
    rim_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.9))  # Slightly warm
    rim_xform = UsdGeom.Xformable(rim_light)
    rim_xform.AddRotateXYZOp().Set(Gf.Vec3f(-70, 180, 0))  # From behind
    
    print(f"   - Created 3 directional lights: Key (2000), Fill (800), Rim (600)")

def create_additional_cameras(stage):
    """Create multiple camera viewpoints"""
    cameras_created = []
    
    # Top-down camera
    top_camera_path = "/World/TopDownCamera"
    top_camera_prim = UsdGeom.Camera.Define(stage, top_camera_path)
    top_camera_prim.CreateFocalLengthAttr(35.0)
    top_camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
    
    # Position directly above the workspace
    top_xform = UsdGeom.Xformable(top_camera_prim)
    top_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.5))  # 2.5m above origin
    top_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))  # Look straight down
    
    cameras_created.append({
        'name': 'TopDownCamera',
        'path': top_camera_path,
        'description': 'Top-down workspace view'
    })
    
    # Angled perspective camera
    persp_camera_path = "/World/PerspectiveCamera"
    persp_camera_prim = UsdGeom.Camera.Define(stage, persp_camera_path)
    persp_camera_prim.CreateFocalLengthAttr(50.0)
    persp_camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
    
    # Position at an angle to see robot and workspace
    persp_xform = UsdGeom.Xformable(persp_camera_prim)
    persp_xform.AddTranslateOp().Set(Gf.Vec3d(1.8, 1.8, 1.2))  # Angled position
    persp_xform.AddRotateXYZOp().Set(Gf.Vec3f(-25, 45, 0))  # Look toward workspace
    
    cameras_created.append({
        'name': 'PerspectiveCamera',
        'path': persp_camera_path,
        'description': 'Angled perspective view'
    })
    
    # Side view camera
    side_camera_path = "/World/SideCamera"
    side_camera_prim = UsdGeom.Camera.Define(stage, side_camera_path)
    side_camera_prim.CreateFocalLengthAttr(50.0)
    side_camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
    
    # Position to side of robot
    side_xform = UsdGeom.Xformable(side_camera_prim)
    side_xform.AddTranslateOp().Set(Gf.Vec3d(2.5, 0.0, 1.0))  # Side position
    side_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))  # Look toward robot
    
    cameras_created.append({
        'name': 'SideCamera',
        'path': side_camera_path,
        'description': 'Side view of robot'
    })
    
    print(f"📷 Created {len(cameras_created)} additional cameras")
    return cameras_created

def run_baseline_with_multi_camera():
    """Run baseline evaluation with multiple camera views"""
    print("🎬 BASELINE EVALUATION WITH MULTI-CAMERA CAPTURE")
    
    # Create output directory
    output_dir = Path("/ros2_ws/output/baseline_multi_camera")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize world
    world = World(stage_units_in_meters=1.0)
    
    # Load the updated scene with eval_camera
    open_stage("/ros2_ws/assets/evaluation_scene.usd")
    
    # Initialize world
    world.reset()
    
    print(f"📥 Scene loaded with stage: {world.stage}")
    
    # Set up enhanced lighting to prevent white images
    create_enhanced_lighting(world.stage)
    
    # Create additional cameras
    additional_cameras = create_additional_cameras(world.stage)
    
    # Find all cameras in scene
    all_cameras = []
    for prim in world.stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            all_cameras.append({
                'path': str(prim.GetPath()),
                'name': prim.GetName()
            })
    
    print(f"📷 Total cameras found: {len(all_cameras)}")
    for cam in all_cameras:
        print(f"   - {cam['name']} at {cam['path']}")
    
    # Set up scene complexity manager and controller
    scene_manager = SceneComplexityManager(world.stage)
    controller = RealisticBaselineController()
    evaluator = RealisticBaselineEvaluator()
    
    # Set up Level 1 complexity (simple scene)
    print("🎯 Setting up Level 1 complexity")
    scene_manager.setup_complexity_level(1)
    
    # Let simulation settle
    print("⏳ Letting simulation settle...")
    for _ in range(60):  # 1 second
        world.step(render=True)
    
    # Run one baseline trial with camera recording
    print("🔄 Running baseline trial with camera recording...")
    
    from omni.isaac.sensor import Camera
    
    # Initialize cameras and capture screenshots
    camera_sensors = {}
    for camera_info in all_cameras:
        try:
            camera_sensor = Camera(
                prim_path=camera_info['path'],
                name=f"{camera_info['name']}_sensor",
                frequency=20,
                resolution=(1920, 1080),
                dt=0.05
            )
            camera_sensor.initialize()
            camera_sensors[camera_info['name']] = camera_sensor
            print(f"   ✅ Initialized camera: {camera_info['name']}")
        except Exception as e:
            print(f"   ❌ Failed to initialize {camera_info['name']}: {e}")
    
    # Run simulation steps with periodic screenshots
    screenshot_interval = 20  # Every 20 steps (1/3 second at 60 FPS)
    total_steps = 300  # 5 seconds total
    screenshot_count = 0
    
    for step in range(total_steps):
        world.step(render=True)
        
        # Capture screenshots at intervals
        if step % screenshot_interval == 0:
            timestamp = time.time()
            
            for camera_name, camera_sensor in camera_sensors.items():
                try:
                    current_frame = camera_sensor.get_current_frame()
                    
                    if current_frame is not None and "rgba" in current_frame:
                        rgba_data = current_frame["rgba"]
                        
                        # Check if image is problematic
                        if rgba_data.max() > 0.99 and rgba_data.min() > 0.95:
                            print(f"   ⚠️ {camera_name}: Potential white image detected!")
                        
                        # Convert to uint8 if needed
                        if rgba_data.dtype != np.uint8:
                            rgba_data = (rgba_data * 255).astype(np.uint8)
                        
                        # Remove alpha channel
                        rgb_data = rgba_data[:, :, :3]
                        
                        # Save using PIL
                        from PIL import Image
                        image = Image.fromarray(rgb_data, 'RGB')
                        
                        screenshot_path = output_dir / f"{camera_name}_step_{step:04d}.png"
                        image.save(str(screenshot_path))
                        screenshot_count += 1
                        
                        if step % 60 == 0:  # Log every second
                            print(f"   📸 {camera_name}: {screenshot_path}")
                            
                except Exception as e:
                    if step % 60 == 0:  # Only log errors occasionally
                        print(f"   ❌ Error capturing {camera_name}: {e}")
    
    # Create summary
    summary = {
        'total_cameras': len(all_cameras),
        'successful_cameras': len(camera_sensors),
        'total_screenshots': screenshot_count,
        'simulation_steps': total_steps,
        'cameras': [{'name': c['name'], 'path': c['path']} for c in all_cameras],
        'output_directory': str(output_dir)
    }
    
    # Save summary
    summary_path = output_dir / "capture_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 CAPTURE SUMMARY:")
    print(f"   - Total cameras: {len(all_cameras)}")
    print(f"   - Successful cameras: {len(camera_sensors)}")
    print(f"   - Total screenshots: {screenshot_count}")
    print(f"   - Output directory: {output_dir}")
    print(f"   - Summary file: {summary_path}")
    
    return summary

def main():
    try:
        results = run_baseline_with_multi_camera()
        print("✅ Multi-camera baseline completed successfully")
        return results
    except Exception as e:
        print(f"❌ Error in multi-camera baseline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
