#!/usr/bin/env python3

"""
Debug Scene Contents
====================

Simple script to show what's in the scene and print information about cameras and lights.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Debug scene contents
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

def analyze_scene():
    """Analyze the scene contents."""
    print("📸 ANALYZING SCENE CONTENTS")
    print("")
    
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
    print("🔍 SCENE ANALYSIS")
    print("")
    
    # Count different types of prims
    cameras = []
    lights = []
    robots = []
    objects = []
    
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_type = prim.GetPrimTypeInfo().GetTypeName()
        
        # Check for cameras
        if prim.IsA(UsdGeom.Camera):
            cameras.append(prim_path)
            
        # Check for lights
        elif (prim.IsA(UsdLux.SphereLight) or 
              prim.IsA(UsdLux.RectLight) or 
              prim.IsA(UsdLux.DomeLight) or
              prim.IsA(UsdLux.DistantLight)):
            lights.append(prim_path)
            
        # Check for robots (look for UR in the path)
        elif "Robot" in prim_path or "UR" in prim_path:
            robots.append(prim_path)
            
        # Check for cylinders or other objects
        elif "Cylinder" in prim_path or "cylinder" in prim_path:
            objects.append(prim_path)
    
    # Print results
    print("📷 CAMERAS FOUND:")
    if cameras:
        for camera in cameras:
            print(f"   ✅ {camera}")
            
            # Get camera details
            camera_prim = stage.GetPrimAtPath(camera)
            if camera_prim.IsValid():
                camera_obj = UsdGeom.Camera(camera_prim)
                xform = UsdGeom.Xformable(camera_prim)
                transform_matrix = xform.ComputeLocalToWorldTransform(0)
                position = transform_matrix.ExtractTranslation()
                print(f"      Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
                
                # Get focal length if available
                focal_length_attr = camera_obj.GetFocalLengthAttr()
                if focal_length_attr.HasValue():
                    focal_length = focal_length_attr.Get()
                    print(f"      Focal Length: {focal_length}")
    else:
        print("   ❌ No cameras found")
    
    print("")
    print("💡 LIGHTS FOUND:")
    if lights:
        for light in lights:
            print(f"   ✅ {light}")
            
            # Get light details
            light_prim = stage.GetPrimAtPath(light)
            if light_prim.IsValid():
                light_type = "Unknown"
                if light_prim.IsA(UsdLux.SphereLight):
                    light_type = "SphereLight"
                elif light_prim.IsA(UsdLux.RectLight):
                    light_type = "RectLight"
                elif light_prim.IsA(UsdLux.DomeLight):
                    light_type = "DomeLight"
                elif light_prim.IsA(UsdLux.DistantLight):
                    light_type = "DistantLight"
                
                print(f"      Type: {light_type}")
    else:
        print("   ❌ No lights found")
    
    print("")
    print("🤖 ROBOT PARTS FOUND:")
    if robots:
        robot_count = min(5, len(robots))  # Show first 5
        for robot in robots[:robot_count]:
            print(f"   ✅ {robot}")
        if len(robots) > 5:
            print(f"   ... and {len(robots) - 5} more robot parts")
    else:
        print("   ❌ No robot parts found")
    
    print("")
    print("🔵 OBJECTS FOUND:")
    if objects:
        for obj in objects:
            print(f"   ✅ {obj}")
    else:
        print("   ❌ No cylinders found")
    
    print("")
    print(f"📊 SUMMARY:")
    print(f"   📷 Cameras: {len(cameras)}")
    print(f"   💡 Lights: {len(lights)}")
    print(f"   🤖 Robot parts: {len(robots)}")
    print(f"   🔵 Objects: {len(objects)}")
    
    # Step the world to ensure everything is loaded
    print("")
    print("🔄 Stepping world to ensure everything is rendered...")
    world.step(render=True)
    time.sleep(2.0)
    
    print("✅ Scene analysis complete!")
    
    # Clean up
    simulation_app.close()

if __name__ == "__main__":
    analyze_scene()
