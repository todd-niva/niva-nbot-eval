#!/usr/bin/env python3

"""
Analyze Scene Cameras
=====================

Simple script to analyze the cameras in the edited scene.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Analyze camera configuration in the edited scene
"""

import sys
sys.path.append('/isaac-sim/kit/python/lib')

from pxr import Usd, UsdGeom

def analyze_scene():
    """Analyze the scene cameras."""
    print("🔍 SCENE ANALYSIS")
    print("")
    
    # Load the scene
    stage = Usd.Stage.Open('/ros2_ws/assets/evaluation_scene.usd')
    
    print("📷 Available cameras:")
    cameras_found = []
    
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cameras_found.append(prim)
            print(f"   - {prim.GetPath()}")
            
            # Get camera properties
            camera = UsdGeom.Camera(prim)
            xform = UsdGeom.Xformable(prim)
            transform_matrix = xform.ComputeLocalToWorldTransform(0)
            position = transform_matrix.ExtractTranslation()
            
            print(f"     Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
            
            # Get camera intrinsic properties
            focal_length = camera.GetFocalLengthAttr().Get()
            horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
            vertical_aperture = camera.GetVerticalApertureAttr().Get()
            
            print(f"     Focal length: {focal_length}")
            print(f"     Horizontal aperture: {horizontal_aperture}")
            print(f"     Vertical aperture: {vertical_aperture}")
            print()
    
    if cameras_found:
        print("✅ Camera integration successful!")
        print(f"Found {len(cameras_found)} camera(s) in the scene")
        
        # Check for eval_camera specifically
        eval_camera_paths = [
            "/World/eval_camera",
            "/World/EvaluationCamera", 
            "/World/EvalCamera"
        ]
        
        for path in eval_camera_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                print(f"🎯 Found evaluation camera at: {path}")
                break
        else:
            print("⚠️  No evaluation camera found with expected names")
    else:
        print("❌ No cameras found in the scene")

if __name__ == "__main__":
    analyze_scene()
