#!/usr/bin/env python3

"""
Test Evaluation Camera
======================

Simple script to test the eval_camera from the edited scene and take a screenshot.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Test the edited camera configuration
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim in headless mode
simulation_app = SimulationApp({"headless": True})

# USD imports
from pxr import UsdGeom, Gf, UsdLux, Usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.render_product import create_hydra_texture
from omni.isaac.sensor import Camera

def test_eval_camera():
    """Test the eval_camera from the edited scene."""
    print("🔄 TESTING EVALUATION CAMERA")
    print("")
    
    # Load the edited scene
    scene_path = "/ros2_ws/assets/evaluation_scene.usd"
    if not Path(scene_path).exists():
        print(f"❌ Scene file not found: {scene_path}")
        return False
    
    print(f"📂 Loading scene: {scene_path}")
    
    # Open the stage
    open_stage(scene_path)
    
    # Create world
    world = World(stage_units_in_meters=1.0)
    
    # Reset the world
    world.reset()
    
    # Wait for scene to load
    time.sleep(2.0)
    
    # Get the stage
    stage = world.stage
    
    # Find the eval_camera
    eval_camera_path = "/World/eval_camera"
    eval_camera_prim = stage.GetPrimAtPath(eval_camera_path)
    
    if not eval_camera_prim.IsValid():
        print(f"❌ eval_camera not found at: {eval_camera_path}")
        # Try alternative paths
        alternative_paths = [
            "/World/EvaluationCamera",
            "/World/eval_camera",
            "/World/EvalCamera"
        ]
        
        for alt_path in alternative_paths:
            alt_prim = stage.GetPrimAtPath(alt_path)
            if alt_prim.IsValid():
                print(f"✅ Found camera at: {alt_path}")
                eval_camera_path = alt_path
                eval_camera_prim = alt_prim
                break
        else:
            print("❌ No evaluation camera found in scene")
            return False
    
    print(f"✅ Found evaluation camera: {eval_camera_path}")
    
    # Get camera properties
    camera = UsdGeom.Camera(eval_camera_prim)
    
    # Get transform
    xform = UsdGeom.Xformable(eval_camera_prim)
    transform_matrix = xform.ComputeLocalToWorldTransform(0)
    
    position = transform_matrix.ExtractTranslation()
    rotation_matrix = transform_matrix.ExtractRotationMatrix()
    
    print(f"📍 Camera position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    print(f"🔄 Camera rotation matrix:")
    for i, row in enumerate(rotation_matrix):
        print(f"   [{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}]")
    
    # Get camera intrinsic properties
    focal_length = camera.GetFocalLengthAttr().Get()
    horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
    vertical_aperture = camera.GetVerticalApertureAttr().Get()
    
    print(f"📷 Camera properties:")
    print(f"   Focal length: {focal_length}")
    print(f"   Horizontal aperture: {horizontal_aperture}")
    print(f"   Vertical aperture: {vertical_aperture}")
    
    # Create a render product for the camera
    print("🎬 Creating render product...")
    
    try:
        # Create render product
        render_product_path = "/Render/Product"
        render_product = create_hydra_texture(
            prim_path=render_product_path,
            camera_path=eval_camera_path,
            resolution=(1920, 1080)
        )
        
        print("✅ Render product created successfully")
        
        # Step the world a few times to ensure everything is ready
        for i in range(10):
            world.step()
            time.sleep(0.1)
        
        # Take a screenshot
        print("📸 Taking screenshot...")
        
        # Get the render product
        render_product_prim = stage.GetPrimAtPath(render_product_path)
        if render_product_prim.IsValid():
            # Save the image
            output_path = "/ros2_ws/output/eval_camera_test.png"
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use Isaac Sim's render product to save image
            from omni.isaac.core.utils.render_product import get_render_product_path
            from omni.isaac.core.utils.render_product import save_render_product
            
            # Save the render product
            success = save_render_product(render_product_path, output_path)
            
            if success:
                print(f"✅ Screenshot saved: {output_path}")
                
                # Copy to local machine
                local_output = "/home/todd/ur10e_2f140_topic_based_ros2_control/eval_camera_test.png"
                import shutil
                shutil.copy2(output_path, local_output)
                print(f"✅ Screenshot copied to: {local_output}")
                
                return True
            else:
                print("❌ Failed to save screenshot")
                return False
        else:
            print("❌ Render product not valid")
            return False
            
    except Exception as e:
        print(f"❌ Error creating render product: {e}")
        return False

def main():
    """Main function."""
    try:
        success = test_eval_camera()
        
        if success:
            print("")
            print("🎯 EVALUATION CAMERA TEST COMPLETE!")
            print("The eval_camera is working and a screenshot has been taken.")
        else:
            print("")
            print("❌ EVALUATION CAMERA TEST FAILED!")
            print("Check the error messages above for details.")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
