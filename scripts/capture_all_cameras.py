#!/usr/bin/env python3

"""
Capture All Cameras
===================

Captures screenshots from all available cameras with proper lighting setup.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Debug camera setup and lighting issues
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
from isaacsim.core.utils.render_product import create_hydra_texture
from pxr import UsdGeom, Gf, UsdLux, Usd
import omni.kit.viewport.utility as vp_utils
import carb

def setup_lighting(stage):
    """Setup lighting - disable global illumination, enable only dome light."""
    print("🔧 Setting up lighting...")
    
    # Disable global illumination
    try:
        # Get render settings
        render_settings = carb.settings.get_settings()
        render_settings.set("/rtx/pathtracing/enabled", False)
        render_settings.set("/rtx/pathtracing/totalSpp", 1)
        render_settings.set("/rtx/indirectDiffuse/enabled", False)
        render_settings.set("/rtx/reflections/enabled", False)
        render_settings.set("/rtx/gi/enabled", False)
        print("   ✅ Global illumination disabled")
    except Exception as e:
        print(f"   ⚠️ Could not disable global illumination: {e}")
    
    # Find and configure lights
    lights_found = []
    dome_light_found = False
    
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.Light):
            light_path = str(prim.GetPath())
            lights_found.append(light_path)
            
            # Check if this is our dome light
            if "dome" in light_path.lower() or "Dome" in light_path:
                dome_light_found = True
                # Enable the dome light
                light = UsdLux.DomeLight(prim)
                light.CreateIntensityAttr(1.0)
                print(f"   ✅ Enabled dome light: {light_path}")
            else:
                # Disable other lights
                light = UsdLux.Light(prim)
                light.CreateIntensityAttr(0.0)
                print(f"   🔇 Disabled light: {light_path}")
    
    print(f"   Found {len(lights_found)} lights total")
    if not dome_light_found:
        print("   ⚠️ No dome light found!")
    
    return lights_found

def find_all_cameras(stage):
    """Find all cameras in the scene."""
    cameras = []
    
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

def capture_camera_screenshot(camera_path, output_dir):
    """Capture a screenshot from a specific camera."""
    try:
        # Get camera name for filename
        camera_name = camera_path.split("/")[-1]
        if not camera_name:
            camera_name = "unknown_camera"
        
        output_path = f"{output_dir}/{camera_name}_screenshot.png"
        
        print(f"   📸 Capturing from {camera_path} -> {output_path}")
        
        # Set the viewport camera
        viewport_api = vp_utils.get_active_viewport()
        if viewport_api:
            viewport_api.set_camera_path(camera_path)
            time.sleep(1.0)  # Wait for camera to update
            
            # Capture screenshot
            viewport_api.capture_viewport_to_file(output_path)
            time.sleep(0.5)  # Wait for capture to complete
            
            if os.path.exists(output_path):
                print(f"   ✅ Screenshot saved: {output_path}")
                return True
            else:
                print(f"   ❌ Screenshot not found: {output_path}")
                return False
        else:
            print(f"   ❌ Could not get viewport API")
            return False
            
    except Exception as e:
        print(f"   ❌ Error capturing from {camera_path}: {e}")
        return False

def main():
    """Main function to capture all camera screenshots."""
    print("📸 CAPTURING ALL CAMERA SCREENSHOTS")
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
    print("🔧 LIGHTING SETUP")
    lights = setup_lighting(stage)
    
    print("")
    print("📷 FINDING CAMERAS")
    cameras = find_all_cameras(stage)
    
    if not cameras:
        print("❌ No cameras found in scene!")
        return
    
    print(f"Found {len(cameras)} cameras")
    print("")
    
    print("📸 CAPTURING SCREENSHOTS")
    successful_captures = 0
    
    for camera_path in cameras:
        success = capture_camera_screenshot(camera_path, output_dir)
        if success:
            successful_captures += 1
    
    print("")
    print(f"✅ Successfully captured {successful_captures}/{len(cameras)} screenshots")
    print(f"📁 Screenshots saved to: {output_dir}")
    
    # Wait a moment for all operations to complete
    time.sleep(2.0)
    
    # Clean up
    simulation_app.close()

if __name__ == "__main__":
    main()
