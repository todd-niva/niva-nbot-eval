#!/usr/bin/env python3

"""
Simple Camera Capture
=====================

Simple script to capture screenshots from all available cameras without complex lighting setup.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Debug camera setup and capture screenshots
"""

import time
import os
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
from pxr import UsdGeom, Gf, UsdLux, Usd

def find_all_cameras(stage):
    """Find all cameras in the scene."""
    cameras = []
    
    print("📷 FINDING CAMERAS")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_path = str(prim.GetPath())
            cameras.append(camera_path)
            
            # Get camera properties
            camera = UsdGeom.Camera(prim)
            xform = UsdGeom.Xformable(prim)
            transform_matrix = xform.ComputeLocalToWorldTransform(0)
            position = transform_matrix.ExtractTranslation()
            
            print(f"   📷 Camera: {camera_path}")
            print(f"      Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    
    return cameras

def find_all_lights(stage):
    """Find all lights in the scene."""
    lights = []
    
    print("💡 FINDING LIGHTS")
    for prim in stage.Traverse():
        # Check for different light types
        if (prim.IsA(UsdLux.SphereLight) or 
            prim.IsA(UsdLux.RectLight) or 
            prim.IsA(UsdLux.DomeLight) or
            prim.IsA(UsdLux.DistantLight)):
            
            light_path = str(prim.GetPath())
            lights.append(light_path)
            
            # Get light type
            light_type = "Unknown"
            if prim.IsA(UsdLux.SphereLight):
                light_type = "SphereLight"
            elif prim.IsA(UsdLux.RectLight):
                light_type = "RectLight"
            elif prim.IsA(UsdLux.DomeLight):
                light_type = "DomeLight"
            elif prim.IsA(UsdLux.DistantLight):
                light_type = "DistantLight"
            
            print(f"   💡 Light: {light_path} ({light_type})")
    
    return lights

def take_basic_screenshot(world, output_dir, filename="scene_overview.png"):
    """Take a basic screenshot of the current scene."""
    try:
        # Step the world to ensure everything is rendered
        world.step(render=True)
        time.sleep(1.0)  # Wait for rendering
        
        output_path = f"{output_dir}/{filename}"
        
        # Try to capture using replicator
        import omni.replicator.core as rep
        
        # Create a render product for the current viewport
        render_product = rep.create.render_product("/OmniverseKit_Persp", (1920, 1080))
        
        # Capture the image
        rep.settings.set_render_pathtraced(samples_per_pixel=32)
        
        # Render a frame
        rep.orchestrator.step()
        
        # Get the data
        data = rep.BackendDispatch.get_instance().dispatch_sync()
        if render_product in data:
            image_data = data[render_product]["rgb"]
            if image_data is not None:
                # Save the image
                import numpy as np
                from PIL import Image
                
                # Convert to PIL image and save
                img = Image.fromarray(image_data.astype(np.uint8))
                img.save(output_path)
                
                print(f"   ✅ Screenshot saved: {output_path}")
                return True
        
        print(f"   ❌ Failed to capture screenshot")
        return False
        
    except Exception as e:
        print(f"   ❌ Error capturing screenshot: {e}")
        return False

def main():
    """Main function to capture camera screenshots."""
    print("📸 SIMPLE CAMERA CAPTURE DEBUG")
    print("")
    
    # Create output directory
    output_dir = "/ros2_ws/output/camera_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load our edited scene
    scene_path = "/ros2_ws/assets/evaluation_scene.usd"
    print(f"🎬 Loading scene: {scene_path}")
    
    # Open the stage
    open_stage(scene_path)
    world = World()
    world.reset()
    
    # Get the stage
    stage = world.stage
    
    print("")
    cameras = find_all_cameras(stage)
    
    print("")
    lights = find_all_lights(stage)
    
    print("")
    print("📸 TAKING BASIC SCREENSHOT")
    success = take_basic_screenshot(world, output_dir, "scene_with_eval_camera.png")
    
    if success:
        print("✅ Screenshot captured successfully!")
    else:
        print("❌ Screenshot capture failed")
    
    print("")
    print(f"📁 Output directory: {output_dir}")
    print(f"   Found {len(cameras)} cameras")
    print(f"   Found {len(lights)} lights")
    
    # Wait a moment for all operations to complete
    time.sleep(2.0)
    
    # Clean up
    simulation_app.close()

if __name__ == "__main__":
    main()
