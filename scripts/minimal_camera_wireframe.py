#!/usr/bin/env python3

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from isaacsim import SimulationApp

# Launch Isaac Sim with wireframe/debug rendering
simulation_app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080,
    "physics_gpu": 0,
    "physics_dt": 1.0/240.0,
    "rendering_dt": 1.0/60.0,
    "renderer": "RayTracedLighting",
    "anti_aliasing": 1,
    "multi_gpu": False,
})

from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import open_stage
from omni.isaac.sensor import Camera
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf

def create_wireframe_materials(stage):
    """Create wireframe materials for better visibility"""
    print("🎨 Creating wireframe materials")
    
    # Enable wireframe view via render settings
    # This is a fallback approach when materials don't work
    try:
        import omni.kit.commands
        
        # Set viewport to wireframe mode
        omni.kit.commands.execute("ChangeProperty",
            prop_path="/Render/RenderProduct_Replicator/displayMode",
            value="wireframe",
            prev=None)
        print("   ✅ Set wireframe display mode")
    except Exception as e:
        print(f"   ⚠️ Could not set wireframe mode: {e}")

def create_simple_lighting(stage):
    """Create very simple, strong lighting"""
    print("💡 Creating simple strong lighting")
    
    # Disable dome light if it exists
    dome_light_path = "/World/DomeLight"
    dome_prim = stage.GetPrimAtPath(dome_light_path)
    if dome_prim.IsValid():
        stage.RemovePrim(dome_light_path)
        print("   - Removed dome light")
    
    # Create single strong directional light
    light_path = "/World/StrongDirectionalLight"
    light = UsdLux.DirectionalLight.Define(stage, light_path)
    light.CreateIntensityAttr(5000.0)  # Very strong
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))  # Pure white
    
    # Point light towards center
    light_xform = UsdGeom.Xformable(light)
    light_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
    
    print("   ✅ Created strong directional light (5000 intensity)")

def create_debug_cameras(stage):
    """Create cameras with very simple, debug-focused positions"""
    cameras = []
    
    # Simple top camera
    top_path = "/World/DebugTopCamera"
    top_cam = UsdGeom.Camera.Define(stage, top_path)
    top_cam.CreateFocalLengthAttr(24.0)  # Wide angle
    top_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    
    top_xform = UsdGeom.Xformable(top_cam)
    top_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))  # High up
    top_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))  # Look down
    
    cameras.append({"name": "DebugTopCamera", "path": top_path})
    
    # Simple front camera
    front_path = "/World/DebugFrontCamera"
    front_cam = UsdGeom.Camera.Define(stage, front_path)
    front_cam.CreateFocalLengthAttr(35.0)
    front_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    
    front_xform = UsdGeom.Xformable(front_cam)
    front_xform.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, 1.5))  # In front
    front_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))  # Look at origin
    
    cameras.append({"name": "DebugFrontCamera", "path": front_path})
    
    print(f"📷 Created {len(cameras)} debug cameras")
    return cameras

def add_simple_objects(stage):
    """Add very simple geometric objects for visibility testing"""
    print("📦 Adding simple test objects")
    
    # Create a simple colored cube
    cube_path = "/World/TestCube"
    cube_prim = UsdGeom.Cube.Define(stage, cube_path)
    cube_prim.CreateSizeAttr(0.2)  # 20cm cube
    
    cube_xform = UsdGeom.Xformable(cube_prim)
    cube_xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.1))
    
    # Create a simple sphere
    sphere_path = "/World/TestSphere"
    sphere_prim = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere_prim.CreateRadiusAttr(0.1)  # 10cm radius
    
    sphere_xform = UsdGeom.Xformable(sphere_prim)
    sphere_xform.AddTranslateOp().Set(Gf.Vec3d(-0.5, 0.0, 0.1))
    
    # Create a plane for reference
    plane_path = "/World/ReferencePlane"
    plane_prim = UsdGeom.Plane.Define(stage, plane_path)
    plane_prim.CreateWidthAttr(2.0)
    plane_prim.CreateLengthAttr(2.0)
    
    print("   ✅ Added cube, sphere, and reference plane")

def capture_from_all_cameras(stage, output_dir):
    """Capture screenshots from all cameras in the scene"""
    print("📸 Capturing from all cameras")
    
    # Find all cameras
    all_cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            all_cameras.append({
                'path': str(prim.GetPath()),
                'name': prim.GetName()
            })
    
    print(f"   Found {len(all_cameras)} cameras:")
    for cam in all_cameras:
        print(f"     - {cam['name']} at {cam['path']}")
    
    # Initialize camera sensors
    camera_sensors = {}
    for cam_info in all_cameras:
        try:
            sensor = Camera(
                prim_path=cam_info['path'],
                name=f"{cam_info['name']}_sensor",
                frequency=10,  # Lower frequency
                resolution=(1920, 1080),
                dt=0.1
            )
            sensor.initialize()
            camera_sensors[cam_info['name']] = {
                'sensor': sensor,
                'path': cam_info['path']
            }
            print(f"   ✅ Initialized {cam_info['name']}")
        except Exception as e:
            print(f"   ❌ Failed to initialize {cam_info['name']}: {e}")
    
    if not camera_sensors:
        print("   ❌ No cameras successfully initialized!")
        return
    
    # Run simulation and capture
    world = World(stage_units_in_meters=1.0)
    world.reset()
    
    print("   🔄 Running simulation...")
    
    # Let simulation settle
    for _ in range(120):  # 2 seconds
        world.step(render=True)
    
    # Capture screenshots
    successful_captures = 0
    
    for camera_name, cam_data in camera_sensors.items():
        try:
            print(f"   📷 Capturing from {camera_name}...")
            
            sensor = cam_data['sensor']
            
            # Get multiple frames to ensure good data
            best_frame = None
            for attempt in range(5):
                world.step(render=True)
                current_frame = sensor.get_current_frame()
                
                if current_frame is not None and "rgba" in current_frame:
                    rgba_data = current_frame["rgba"]
                    
                    # Check for good data
                    value_range = rgba_data.max() - rgba_data.min()
                    unique_values = len(np.unique(rgba_data))
                    
                    print(f"     Attempt {attempt+1}: range={value_range:.3f}, unique={unique_values}")
                    
                    if value_range > 0.1 and unique_values > 10:  # Good data
                        best_frame = current_frame
                        break
                    elif best_frame is None:  # Keep the first frame as backup
                        best_frame = current_frame
            
            if best_frame is not None and "rgba" in best_frame:
                rgba_data = best_frame["rgba"]
                
                # Convert to uint8
                if rgba_data.dtype != np.uint8:
                    rgba_data = (rgba_data * 255).astype(np.uint8)
                
                # Remove alpha channel
                rgb_data = rgba_data[:, :, :3]
                
                # Save using PIL
                from PIL import Image
                image = Image.fromarray(rgb_data, 'RGB')
                
                timestamp = int(time.time())
                screenshot_path = output_dir / f"{camera_name}_wireframe_{timestamp}.png"
                image.save(str(screenshot_path))
                
                # Check if the image is problematic
                gray_image = image.convert('L')
                pixel_values = list(gray_image.getdata())
                unique_pixels = len(set(pixel_values))
                avg_brightness = sum(pixel_values) / len(pixel_values)
                
                status = "✅"
                if unique_pixels < 10:
                    status = "⚠️ Low detail"
                elif avg_brightness > 250:
                    status = "⚠️ Too bright"
                elif avg_brightness < 5:
                    status = "⚠️ Too dark"
                
                print(f"     {status} Saved: {screenshot_path}")
                print(f"     Unique pixels: {unique_pixels}, Avg brightness: {avg_brightness:.1f}")
                
                successful_captures += 1
                
            else:
                print(f"     ❌ No valid frame data")
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    print(f"\n📊 Capture summary: {successful_captures}/{len(camera_sensors)} successful")
    return successful_captures

def main():
    """Main debug rendering function"""
    print("🎬 MINIMAL CAMERA WIREFRAME DEBUG")
    
    # Create output directory
    output_dir = Path("/ros2_ws/output/wireframe_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize world
    world = World(stage_units_in_meters=1.0)
    
    # Load the scene
    open_stage("/ros2_ws/assets/evaluation_scene.usd")
    
    # Reset world
    world.reset()
    
    print(f"📥 Scene loaded: {world.stage}")
    
    # Set up wireframe rendering
    create_wireframe_materials(world.stage)
    
    # Set up simple lighting
    create_simple_lighting(world.stage)
    
    # Add simple test objects
    add_simple_objects(world.stage)
    
    # Create debug cameras
    debug_cameras = create_debug_cameras(world.stage)
    
    # Capture from all cameras
    successful_captures = capture_from_all_cameras(world.stage, output_dir)
    
    # Create summary
    summary = {
        'output_directory': str(output_dir),
        'successful_captures': successful_captures,
        'rendering_mode': 'wireframe_debug',
        'timestamp': time.time()
    }
    
    summary_path = output_dir / "wireframe_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Wireframe debug completed!")
    print(f"   Output: {output_dir}")
    print(f"   Summary: {summary_path}")
    
    return summary

if __name__ == "__main__":
    try:
        result = main()
        print("✅ Script completed successfully")
    except Exception as e:
        print(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
