#!/usr/bin/env python3

import time
import json
from pathlib import Path
from typing import List, Dict, Any
from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080,
    "physics_gpu": 0,
    "physics_dt": 1.0/240.0,
    "rendering_dt": 1.0/60.0,
})

from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.render_product import create_hydra_texture, save_render_product
from pxr import Usd, UsdGeom, UsdLux, Gf
import sys
sys.path.append('/ros2_ws/scripts')
from scene_complexity_manager import SceneComplexityManager

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

def get_camera_info(camera_prim):
    """Get detailed camera information"""
    camera = UsdGeom.Camera(camera_prim)
    xform = UsdGeom.Xformable(camera_prim)
    
    # Get transform
    transform_matrix = xform.ComputeLocalToWorldTransform(0)
    position = transform_matrix.ExtractTranslation()
    
    # Get camera properties
    focal_length = camera.GetFocalLengthAttr().Get() if camera.GetFocalLengthAttr() else "N/A"
    
    return {
        'position': [float(position[0]), float(position[1]), float(position[2])],
        'focal_length': focal_length
    }

def setup_basic_scene_level_1():
    """Set up Level 1 complexity scene for camera testing"""
    complexity_manager = SceneComplexityManager()
    complexity_manager.setup_complexity_level(1)
    print("✅ Scene Level 1 setup complete")

def capture_multi_camera_screenshots():
    """Capture screenshots from all available cameras"""
    print("🎬 MULTI-CAMERA SCREENSHOT CAPTURE")
    
    # Create output directory
    output_dir = Path("/ros2_ws/output/multi_camera_views")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize world and load scene
    world = World(stage_units_in_meters=1.0)
    open_stage("/ros2_ws/assets/evaluation_scene.usd")
    world.scene.add_default_ground_plane()
    world.reset()
    
    # Set up a basic scene for testing
    setup_basic_scene_level_1()
    
    # Let physics settle
    for _ in range(120):  # 2 seconds at 60 FPS
        world.step(render=True)
    
    # Find all cameras
    cameras = find_all_cameras(world.stage)
    print(f"📷 Found {len(cameras)} cameras:")
    
    camera_info = {}
    screenshots_captured = []
    
    for camera in cameras:
        camera_path = camera['path']
        camera_name = camera['name']
        
        print(f"   - {camera_name} at {camera_path}")
        
        # Get camera details
        info = get_camera_info(camera['prim'])
        camera_info[camera_name] = {
            'path': camera_path,
            **info
        }
        
        try:
            # Create render product for this camera
            print(f"🎯 Capturing from {camera_name}...")
            rp = create_hydra_texture(camera_path, [1920, 1080])
            
            # Save screenshot
            screenshot_path = output_dir / f"{camera_name}_view.png"
            save_render_product(rp, str(screenshot_path))
            
            screenshots_captured.append({
                'camera': camera_name,
                'path': str(screenshot_path),
                'capture_successful': True
            })
            
            print(f"✅ {camera_name} screenshot saved to: {screenshot_path}")
            
        except Exception as e:
            print(f"❌ Failed to capture {camera_name}: {str(e)}")
            screenshots_captured.append({
                'camera': camera_name,
                'path': None,
                'capture_successful': False,
                'error': str(e)
            })
    
    # Save camera information and capture results
    results = {
        'timestamp': time.time(),
        'cameras_found': len(cameras),
        'camera_info': camera_info,
        'screenshots_captured': screenshots_captured,
        'scene_complexity_level': 1,
        'scene_file': '/ros2_ws/assets/evaluation_scene.usd'
    }
    
    results_path = output_dir / "camera_capture_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 CAPTURE SUMMARY:")
    print(f"   - Cameras found: {len(cameras)}")
    print(f"   - Screenshots captured: {len([s for s in screenshots_captured if s['capture_successful']])}")
    print(f"   - Results saved to: {results_path}")
    
    # Copy to host for inspection
    print("\n📁 Copying screenshots to host...")
    
    return results

def main():
    try:
        results = capture_multi_camera_screenshots()
        print("✅ Multi-camera capture completed successfully")
        return results
    except Exception as e:
        print(f"❌ Error in multi-camera capture: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
