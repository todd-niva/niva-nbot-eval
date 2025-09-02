#!/usr/bin/env python3

"""
Capture Eval Camera Screenshot
==============================

Captures a screenshot using the eval_camera for visual inspection.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Visual verification of eval_camera positioning and scene setup
"""

import time
from pathlib import Path

# Isaac Sim imports using the same structure as our working evaluations
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080
})

from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.render_product import create_hydra_texture
from pxr import UsdGeom, Gf, UsdLux, Usd

def capture_screenshot():
    """Capture a screenshot using the eval_camera."""
    print("📸 CAPTURING EVAL CAMERA SCREENSHOT")
    print("")
    
    # Load our edited scene with the eval_camera
    scene_path = "/ros2_ws/assets/evaluation_scene.usd"
    print(f"📁 Loading scene: {scene_path}")
    open_stage(scene_path)
    
    # Create world
    world = World(stage_units_in_meters=1.0)
    world.reset()
    
    # Wait for scene to load
    print("⏳ Waiting for scene to load...")
    time.sleep(3.0)
    
    # Find the eval_camera
    stage = world.stage
    eval_camera_path = None
    eval_camera_paths = [
        "/World/eval_camera",
        "/World/EvaluationCamera", 
        "/World/EvalCamera"
    ]
    
    for path in eval_camera_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            eval_camera_path = path
            print(f"✅ Found evaluation camera: {path}")
            break
    
    if not eval_camera_path:
        print("❌ No evaluation camera found in scene")
        return False
    
    # Get camera properties
    camera_prim = stage.GetPrimAtPath(eval_camera_path)
    camera = UsdGeom.Camera(camera_prim)
    xform = UsdGeom.Xformable(camera_prim)
    transform_matrix = xform.ComputeLocalToWorldTransform(0)
    position = transform_matrix.ExtractTranslation()
    
    print(f"📷 Camera Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    
    # Step the world to ensure everything is rendered
    print("🔄 Stepping simulation for rendering...")
    for i in range(10):
        world.step()
        time.sleep(0.1)
    
    # Create output directory
    output_dir = Path("/ros2_ws/output/camera_screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create render product for the camera
        print("🎬 Creating render product for eval_camera...")
        render_product = create_hydra_texture(
            camera_path=eval_camera_path,
            resolution=(1920, 1080)
        )
        
        # Step a few more times to ensure rendering
        for i in range(5):
            world.step()
            time.sleep(0.2)
        
        # Get the rendered data
        from omni.syntheticdata import SyntheticData
        import numpy as np
        from PIL import Image
        
        # Get RGB data from render product
        rgb_data = SyntheticData.get_groundtruth(
            ["rgb"], render_product, viewport=None
        )
        
        if rgb_data and "rgb" in rgb_data:
            # Convert to PIL Image and save
            rgb_array = rgb_data["rgb"][:, :, :3]  # Remove alpha channel if present
            
            # Convert from float [0,1] to uint8 [0,255] if needed
            if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float64:
                rgb_array = (rgb_array * 255).astype(np.uint8)
            
            # Create PIL Image
            image = Image.fromarray(rgb_array)
            
            # Save screenshot
            screenshot_path = output_dir / "eval_camera_view.png"
            image.save(screenshot_path)
            print(f"✅ Screenshot saved: {screenshot_path}")
            
            # Copy to host output directory for easy access
            host_output_dir = Path("/ros2_ws/output_host/camera_screenshots")
            host_output_dir.mkdir(parents=True, exist_ok=True)
            host_screenshot_path = host_output_dir / "eval_camera_view.png"
            image.save(host_screenshot_path)
            print(f"✅ Screenshot also saved to host: {host_screenshot_path}")
            
            return True
        else:
            print("❌ No RGB data captured")
            return False
            
    except Exception as e:
        print(f"❌ Error capturing screenshot: {e}")
        # Try alternative approach using kit screenshot
        try:
            print("🔄 Trying alternative screenshot method...")
            import omni.kit.app
            
            # Save using kit's screenshot function
            screenshot_path = output_dir / "eval_camera_fallback.png"
            success = omni.kit.app.get_app().print_and_log(f"Screenshot saved to {screenshot_path}")
            
            if success:
                print(f"✅ Fallback screenshot saved: {screenshot_path}")
                return True
            else:
                print("❌ Fallback screenshot failed")
                return False
                
        except Exception as e2:
            print(f"❌ Fallback method also failed: {e2}")
            return False

def main():
    """Main function."""
    try:
        success = capture_screenshot()
        
        if success:
            print("")
            print("🎯 SCREENSHOT CAPTURE COMPLETE!")
            print("✅ Check the output directories for the eval_camera view")
            print("📁 Container: /ros2_ws/output/camera_screenshots/")
            print("📁 Host: /ros2_ws/output_host/camera_screenshots/")
        else:
            print("")
            print("❌ SCREENSHOT CAPTURE FAILED!")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
