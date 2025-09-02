#!/usr/bin/env python3

"""
Setup Evaluation Camera for Training Validation
==============================================

This script helps you set up and save a camera configuration for our training validation
evaluations. You can:

1. Open the scene in Isaac Sim
2. Add and position a camera manually
3. Save the camera configuration
4. Use it in our evaluation framework

The saved camera configuration will be used across all our evaluation scripts
for consistent visual recording and inspection.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Camera setup for training validation recordings
"""

import math
import sys
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False, "width": 1920, "height": 1080})

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.stage import set_stage_up_axis

# USD imports
import omni.usd
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema

def log(msg: str) -> None:
    """Enhanced logging with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

class EvaluationCameraSetup:
    """Setup and save camera configuration for training validation."""
    
    def __init__(self):
        self.stage = None
        self.world = None
        self.camera_config = {}
        
    def setup_scene(self):
        """Setup the basic scene with robot and objects."""
        log("Setting up evaluation scene...")
        
        # Get stage
        self.stage = omni.usd.get_context().get_stage()
        
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        
        # Load the robot scene
        robot_usd_path = "/ros2_ws/assets/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
        if not Path(robot_usd_path).exists():
            log(f"❌ Robot USD not found at: {robot_usd_path}")
            return False
            
        # Add robot to stage
        add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/Robot")
        log("✅ Robot loaded successfully")
        
        # Add a simple cylinder for reference
        cylinder_path = "/World/TestCylinder"
        cylinder_prim = UsdGeom.Cylinder.Define(self.stage, cylinder_path)
        
        # Set cylinder properties
        cylinder_prim.CreateRadiusAttr(0.03)  # 3cm radius
        cylinder_prim.CreateHeightAttr(0.12)  # 12cm height
        
        # Position cylinder in front of robot
        xform = UsdGeom.Xformable(cylinder_prim)
        xform.SetTranslate(Gf.Vec3d(0.55, 0.0, 0.06))  # In front of robot
        
        # Add physics
        UsdPhysics.RigidBodyAPI.Apply(cylinder_prim.GetPrim())
        UsdPhysics.MassAPI.Apply(cylinder_prim.GetPrim()).CreateMassAttr(0.5)  # 500g
        
        log("✅ Test cylinder added")
        
        # Add ground plane
        ground_path = "/World/GroundPlane"
        ground_prim = UsdGeom.Plane.Define(self.stage, ground_path)
        ground_prim.CreateSizeAttr(2.0)  # 2m x 2m
        
        # Position ground
        ground_xform = UsdGeom.Xformable(ground_prim)
        ground_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
        
        # Add physics to ground
        UsdPhysics.CollisionAPI.Apply(ground_prim.GetPrim())
        
        log("✅ Ground plane added")
        
        return True
    
    def add_camera(self, camera_name: str = "EvaluationCamera") -> str:
        """Add a camera to the scene."""
        log(f"Adding camera: {camera_name}")
        
        camera_path = f"/World/{camera_name}"
        camera_prim = UsdGeom.Camera.Define(self.stage, camera_path)
        
        # Set camera properties
        camera_prim.CreateFocalLengthAttr(24.0)  # 24mm focal length
        camera_prim.CreateHorizontalApertureAttr(20.955)  # Standard aperture
        camera_prim.CreateVerticalApertureAttr(15.956)  # Standard aperture
        
        # Set initial position (you can adjust this manually in Isaac Sim)
        xform = UsdGeom.Xformable(camera_prim)
        xform.SetTranslate(Gf.Vec3d(1.5, 1.5, 1.2))  # Default position
        
        log(f"✅ Camera added at: {camera_path}")
        return camera_path
    
    def save_camera_config(self, camera_path: str, config_name: str = "evaluation_camera"):
        """Save camera configuration to file."""
        log(f"Saving camera configuration: {config_name}")
        
        # Get camera prim
        camera_prim = get_prim_at_path(camera_path)
        if not camera_prim:
            log(f"❌ Camera not found at: {camera_path}")
            return False
        
        # Get camera transform
        xform = UsdGeom.Xformable(camera_prim)
        transform_matrix = xform.ComputeLocalToWorldTransform(0.0)
        
        # Extract position and rotation
        position = transform_matrix.ExtractTranslation()
        rotation_matrix = transform_matrix.ExtractRotationMatrix()
        rotation_quat = Gf.Quatd(rotation_matrix)
        
        # Get camera properties
        camera_schema = UsdGeom.Camera(camera_prim)
        focal_length = camera_schema.GetFocalLengthAttr().Get()
        horizontal_aperture = camera_schema.GetHorizontalApertureAttr().Get()
        vertical_aperture = camera_schema.GetVerticalApertureAttr().Get()
        
        # Create configuration
        config = {
            "camera_path": camera_path,
            "position": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2])
            },
            "rotation": {
                "x": float(rotation_quat.GetImaginary()[0]),
                "y": float(rotation_quat.GetImaginary()[1]),
                "z": float(rotation_quat.GetImaginary()[2]),
                "w": float(rotation_quat.GetReal())
            },
            "camera_properties": {
                "focal_length": float(focal_length),
                "horizontal_aperture": float(horizontal_aperture),
                "vertical_aperture": float(vertical_aperture)
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "description": "Camera configuration for training validation recordings"
        }
        
        # Save to file
        config_file = Path(f"/ros2_ws/config/{config_name}.json")
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        log(f"✅ Camera configuration saved to: {config_file}")
        
        # Also save a human-readable summary
        summary_file = Path(f"/ros2_ws/config/{config_name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Camera Configuration: {config_name}\n")
            f.write(f"Timestamp: {config['timestamp']}\n")
            f.write(f"Camera Path: {camera_path}\n\n")
            f.write(f"Position:\n")
            f.write(f"  X: {config['position']['x']:.3f}m\n")
            f.write(f"  Y: {config['position']['y']:.3f}m\n")
            f.write(f"  Z: {config['position']['z']:.3f}m\n\n")
            f.write(f"Rotation (Quaternion):\n")
            f.write(f"  X: {config['rotation']['x']:.3f}\n")
            f.write(f"  Y: {config['rotation']['y']:.3f}\n")
            f.write(f"  Z: {config['rotation']['z']:.3f}\n")
            f.write(f"  W: {config['rotation']['w']:.3f}\n\n")
            f.write(f"Camera Properties:\n")
            f.write(f"  Focal Length: {config['camera_properties']['focal_length']:.1f}mm\n")
            f.write(f"  Horizontal Aperture: {config['camera_properties']['horizontal_aperture']:.3f}\n")
            f.write(f"  Vertical Aperture: {config['camera_properties']['vertical_aperture']:.3f}\n")
        
        log(f"✅ Camera summary saved to: {summary_file}")
        
        return True
    
    def load_camera_config(self, config_name: str = "evaluation_camera") -> Optional[Dict]:
        """Load camera configuration from file."""
        config_file = Path(f"/ros2_ws/config/{config_name}.json")
        
        if not config_file.exists():
            log(f"❌ Camera configuration not found: {config_file}")
            return None
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        log(f"✅ Camera configuration loaded: {config_name}")
        return config
    
    def apply_camera_config(self, config: Dict, camera_path: str):
        """Apply camera configuration to scene."""
        log(f"Applying camera configuration to: {camera_path}")
        
        # Get camera prim
        camera_prim = get_prim_at_path(camera_path)
        if not camera_prim:
            log(f"❌ Camera not found at: {camera_path}")
            return False
        
        # Apply position
        xform = UsdGeom.Xformable(camera_prim)
        position = Gf.Vec3d(
            config['position']['x'],
            config['position']['y'],
            config['position']['z']
        )
        xform.SetTranslate(position)
        
        # Apply rotation
        rotation_quat = Gf.Quatd(
            config['rotation']['w'],
            config['rotation']['x'],
            config['rotation']['y'],
            config['rotation']['z']
        )
        xform.SetOrient(rotation_quat)
        
        # Apply camera properties
        camera_schema = UsdGeom.Camera(camera_prim)
        camera_schema.GetFocalLengthAttr().Set(config['camera_properties']['focal_length'])
        camera_schema.GetHorizontalApertureAttr().Set(config['camera_properties']['horizontal_aperture'])
        camera_schema.GetVerticalApertureAttr().Set(config['camera_properties']['vertical_aperture'])
        
        log("✅ Camera configuration applied successfully")
        return True

def main():
    """Main function for camera setup."""
    log("🎥 EVALUATION CAMERA SETUP")
    log("This script helps you set up a camera for training validation recordings")
    log("")
    
    setup = EvaluationCameraSetup()
    
    # Setup scene
    if not setup.setup_scene():
        log("❌ Failed to setup scene")
        return
    
    # Add camera
    camera_path = setup.add_camera("EvaluationCamera")
    
    log("")
    log("🎯 CAMERA SETUP INSTRUCTIONS:")
    log("1. In Isaac Sim, you can now manually position the camera")
    log("2. Use the camera controls to get the perfect view of the robot and workspace")
    log("3. When you're happy with the camera position, run this script again with --save")
    log("4. The camera configuration will be saved for use in our evaluation framework")
    log("")
    log("💡 TIP: Position the camera to capture:")
    log("   - Full robot workspace")
    log("   - Clear view of pick-and-place operations")
    log("   - Good lighting and contrast")
    log("   - Minimal occlusion")
    log("")
    log("🔧 To save the camera configuration, run:")
    log("   python setup_evaluation_camera.py --save")
    log("")
    log("📁 Camera will be saved to: /ros2_ws/config/evaluation_camera.json")
    
    # Check if we should save the configuration
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        log("")
        log("💾 SAVING CAMERA CONFIGURATION...")
        if setup.save_camera_config(camera_path, "evaluation_camera"):
            log("✅ Camera configuration saved successfully!")
            log("🎬 You can now use this camera in our evaluation framework")
        else:
            log("❌ Failed to save camera configuration")
    
    # Keep the simulation running so you can adjust the camera
    log("")
    log("🎮 Isaac Sim is running - you can now adjust the camera position")
    log("Press Ctrl+C to exit when done")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log("")
        log("👋 Exiting camera setup")
        simulation_app.close()

if __name__ == "__main__":
    main()
