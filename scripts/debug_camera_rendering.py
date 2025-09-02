#!/usr/bin/env python3

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from isaacsim import SimulationApp

# Launch Isaac Sim with RTX disabled to avoid rendering issues
simulation_app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080,
    "physics_gpu": 0,
    "physics_dt": 1.0/240.0,
    "rendering_dt": 1.0/60.0,
    "renderer": "RayTracedLighting",  # Try RTX
    "anti_aliasing": 3,
    "multi_gpu": False,
})

from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import open_stage
from omni.isaac.sensor import Camera
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
import omni.kit.commands

def create_additional_cameras(stage):
    """Create top-down and perspective cameras manually"""
    cameras_created = []
    
    # Create top-down camera
    top_camera_path = "/World/TopCamera"
    top_camera_prim = UsdGeom.Camera.Define(stage, top_camera_path)
    top_camera_prim.CreateFocalLengthAttr(35.0)
    top_camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000.0))
    
    # Position top camera looking down
    top_xform = UsdGeom.Xformable(top_camera_prim)
    top_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 3.0))  # 3m above origin
    top_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))  # Look down
    
    cameras_created.append({
        'name': 'TopCamera',
        'path': top_camera_path,
        'description': 'Top-down view'
    })
    
    # Create perspective camera
    persp_camera_path = "/World/PerspectiveCamera"
    persp_camera_prim = UsdGeom.Camera.Define(stage, persp_camera_path)
    persp_camera_prim.CreateFocalLengthAttr(50.0)
    persp_camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000.0))
    
    # Position perspective camera at an angle
    persp_xform = UsdGeom.Xformable(persp_camera_prim)
    persp_xform.AddTranslateOp().Set(Gf.Vec3d(2.0, 2.0, 1.5))  # Angled position
    persp_xform.AddRotateXYZOp().Set(Gf.Vec3f(-30, 45, 0))  # Angled view
    
    cameras_created.append({
        'name': 'PerspectiveCamera',
        'path': persp_camera_path,
        'description': 'Angled perspective view'
    })
    
    # Create side camera
    side_camera_path = "/World/SideCamera"
    side_camera_prim = UsdGeom.Camera.Define(stage, side_camera_path)
    side_camera_prim.CreateFocalLengthAttr(35.0)
    side_camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000.0))
    
    # Position side camera
    side_xform = UsdGeom.Xformable(side_camera_prim)
    side_xform.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, 1.0))  # Side position
    side_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))  # Look towards center
    
    cameras_created.append({
        'name': 'SideCamera',
        'path': side_camera_path,
        'description': 'Side view'
    })
    
    return cameras_created

def find_all_cameras(stage):
    """Find all cameras in the scene"""
    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cameras.append({
                'path': str(prim.GetPath()),
                'name': prim.GetName(),
                'prim': prim
            })
    return cameras

def setup_enhanced_lighting(stage):
    """Set up multiple light sources to avoid white images"""
    print("🔆 Setting up enhanced lighting")
    
    # Remove existing dome light if it exists and is causing issues
    dome_light_path = "/World/DomeLight"
    dome_prim = stage.GetPrimAtPath(dome_light_path)
    if dome_prim.IsValid():
        dome_light = UsdLux.DomeLight(dome_prim)
        dome_light.CreateIntensityAttr(300.0)  # Reduce intensity
        dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))  # Slightly blue
        print(f"   - Adjusted DomeLight intensity to 300")
    
    # Add multiple directional lights
    lights_created = []
    
    # Main key light
    key_light_path = "/World/KeyLight"
    key_light = UsdLux.DirectionalLight.Define(stage, key_light_path)
    key_light.CreateIntensityAttr(500.0)
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.9))  # Warm white
    key_xform = UsdGeom.Xformable(key_light)
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
    lights_created.append("KeyLight")
    
    # Fill light
    fill_light_path = "/World/FillLight"
    fill_light = UsdLux.DirectionalLight.Define(stage, fill_light_path)
    fill_light.CreateIntensityAttr(200.0)
    fill_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))  # Cool white
    fill_xform = UsdGeom.Xformable(fill_light)
    fill_xform.AddRotateXYZOp().Set(Gf.Vec3f(-30, -45, 0))
    lights_created.append("FillLight")
    
    # Rim light from behind
    rim_light_path = "/World/RimLight"
    rim_light = UsdLux.DirectionalLight.Define(stage, rim_light_path)
    rim_light.CreateIntensityAttr(150.0)
    rim_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.95))  # Slightly warm
    rim_xform = UsdGeom.Xformable(rim_light)
    rim_xform.AddRotateXYZOp().Set(Gf.Vec3f(-60, 180, 0))
    lights_created.append("RimLight")
    
    print(f"   - Created lights: {', '.join(lights_created)}")
    return lights_created

def capture_camera_with_debug_info(camera_path, camera_name, world, output_dir):
    """Capture from camera with extensive debug information"""
    print(f"\n🎯 Capturing from {camera_name} at {camera_path}")
    
    try:
        # Create camera sensor with specific settings
        camera_sensor = Camera(
            prim_path=camera_path,
            name=f"{camera_name}_sensor",
            frequency=20,
            resolution=(1920, 1080),
            dt=0.05
        )
        
        # Initialize camera
        camera_sensor.initialize()
        print(f"   ✅ Camera sensor initialized")
        
        # Let simulation run for multiple steps to ensure proper initialization
        print(f"   🔄 Running simulation steps...")
        for i in range(60):  # 60 steps = 1 second at 60 FPS
            world.step(render=True)
            if i % 20 == 0:
                print(f"     Step {i}/60")
        
        # Get camera data
        current_frame = camera_sensor.get_current_frame()
        print(f"   📊 Frame data keys: {list(current_frame.keys()) if current_frame else 'None'}")
        
        if current_frame is not None and "rgba" in current_frame:
            rgba_data = current_frame["rgba"]
            print(f"   📏 Image shape: {rgba_data.shape}")
            print(f"   🎨 Data type: {rgba_data.dtype}")
            print(f"   📈 Value range: [{rgba_data.min():.3f}, {rgba_data.max():.3f}]")
            print(f"   🔢 Unique values: {len(np.unique(rgba_data))}")
            
            # Check if image is all white
            if rgba_data.max() > 0.99 and rgba_data.min() > 0.99:
                print(f"   ⚠️ WARNING: Image appears to be all white!")
            elif rgba_data.max() < 0.01:
                print(f"   ⚠️ WARNING: Image appears to be all black!")
            else:
                print(f"   ✅ Image has proper value range")
            
            # Convert to uint8 if needed
            if rgba_data.dtype != np.uint8:
                rgba_data = (rgba_data * 255).astype(np.uint8)
            
            # Remove alpha channel
            rgb_data = rgba_data[:, :, :3]
            
            # Save using PIL
            from PIL import Image
            image = Image.fromarray(rgb_data, 'RGB')
            
            timestamp = int(time.time())
            screenshot_path = output_dir / f"{camera_name}_debug_{timestamp}.png"
            image.save(str(screenshot_path))
            
            print(f"   💾 Saved: {screenshot_path}")
            
            return {
                'camera': camera_name,
                'path': str(screenshot_path),
                'capture_successful': True,
                'shape': rgb_data.shape,
                'value_range': [float(rgba_data.min()), float(rgba_data.max())],
                'unique_values': int(len(np.unique(rgba_data)))
            }
        else:
            print(f"   ❌ No RGBA data available")
            return {
                'camera': camera_name,
                'path': None,
                'capture_successful': False,
                'error': "No RGBA data in frame"
            }
            
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'camera': camera_name,
            'path': None,
            'capture_successful': False,
            'error': str(e)
        }

def debug_camera_rendering():
    """Comprehensive camera rendering debug"""
    print("🔍 COMPREHENSIVE CAMERA RENDERING DEBUG")
    
    # Create output directory
    output_dir = Path("/ros2_ws/output/camera_debug_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize world and load scene
    world = World(stage_units_in_meters=1.0)
    open_stage("/ros2_ws/assets/evaluation_scene.usd")
    world.scene.add_default_ground_plane()
    world.reset()
    
    print(f"📥 Scene loaded: {world.stage}")
    
    # Set up enhanced lighting
    lights_created = setup_enhanced_lighting(world.stage)
    
    # Create additional cameras
    additional_cameras = create_additional_cameras(world.stage)
    print(f"📷 Created additional cameras: {[c['name'] for c in additional_cameras]}")
    
    # Find all cameras (existing + new)
    all_cameras = find_all_cameras(world.stage)
    print(f"📷 Total cameras found: {len(all_cameras)}")
    for cam in all_cameras:
        print(f"   - {cam['name']} at {cam['path']}")
    
    # Let simulation settle
    print("⏳ Letting simulation settle...")
    for _ in range(120):  # 2 seconds
        world.step(render=True)
    
    # Capture from each camera
    results = []
    for camera_info in all_cameras:
        result = capture_camera_with_debug_info(
            camera_info['path'], 
            camera_info['name'], 
            world, 
            output_dir
        )
        results.append(result)
    
    # Save debug report
    debug_report = {
        'timestamp': time.time(),
        'cameras_found': len(all_cameras),
        'cameras_created': additional_cameras,
        'lights_created': lights_created,
        'capture_results': results,
        'simulation_config': {
            'renderer': "RayTracedLighting",
            'physics_gpu': 0,
            'resolution': [1920, 1080]
        }
    }
    
    report_path = output_dir / "debug_report.json"
    with open(report_path, 'w') as f:
        json.dump(debug_report, f, indent=2)
    
    print(f"\n📊 DEBUG SUMMARY:")
    print(f"   - Cameras processed: {len(all_cameras)}")
    successful_captures = [r for r in results if r['capture_successful']]
    print(f"   - Successful captures: {len(successful_captures)}")
    print(f"   - Failed captures: {len(results) - len(successful_captures)}")
    print(f"   - Debug report: {report_path}")
    
    if successful_captures:
        print(f"\n✅ SUCCESSFUL CAPTURES:")
        for result in successful_captures:
            print(f"   - {result['camera']}: {result['path']}")
    
    failed_captures = [r for r in results if not r['capture_successful']]
    if failed_captures:
        print(f"\n❌ FAILED CAPTURES:")
        for result in failed_captures:
            print(f"   - {result['camera']}: {result.get('error', 'Unknown error')}")
    
    return debug_report

def main():
    try:
        results = debug_camera_rendering()
        print("✅ Camera debug completed successfully")
        return results
    except Exception as e:
        print(f"❌ Error in camera debug: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
