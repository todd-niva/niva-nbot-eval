#!/usr/bin/env python3

"""
Create Portable Scene for Cross-Platform Editing
===============================================

This script creates a portable USD scene file that can be:
1. Created on Linux (headless)
2. Copied to Windows
3. Edited in Isaac Sim GUI (cameras, lighting, etc.)
4. Copied back to Linux
5. Used in our evaluation framework

Author: Training Validation Team
Date: 2025-09-02
Purpose: Cross-platform scene editing workflow
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

# Launch Isaac Sim in headless mode
simulation_app = SimulationApp({"headless": True})

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
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema, UsdLux

def log(msg: str) -> None:
    """Enhanced logging with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

class PortableSceneCreator:
    """Creates portable USD scenes for cross-platform editing."""
    
    def __init__(self):
        self.stage = None
        self.world = None
        self.scene_info = {}
        
    def create_evaluation_scene(self, output_path: str = "/ros2_ws/assets/evaluation_scene.usd"):
        """Create a complete evaluation scene with robot, objects, and basic setup."""
        log("Creating portable evaluation scene...")
        
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
        
        # Add test objects for different complexity levels
        self._add_test_objects()
        
        # Add basic lighting setup
        self._add_basic_lighting()
        
        # Add basic camera (will be replaced/edited on Windows)
        self._add_basic_camera()
        
        # Add ground plane
        self._add_ground_plane()
        
        # Save the scene
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.stage.GetRootLayer().Export(str(output_file))
        log(f"✅ Scene saved to: {output_file}")
        
        # Create scene information file
        self._create_scene_info(output_file)
        
        return True
    
    def _add_test_objects(self):
        """Add test objects for different complexity levels."""
        log("Adding test objects...")
        
        # Level 1: Single target cylinder
        self._add_cylinder("/World/Level1/TargetCylinder", 
                          position=(0.55, 0.0, 0.06), 
                          radius=0.03, height=0.12, 
                          color=(0.9, 0.2, 0.1), mass=0.5)
        
        # Level 2: Multiple cylinders with different poses
        self._add_cylinder("/World/Level2/TargetCylinder", 
                          position=(0.55, 0.0, 0.06), 
                          radius=0.03, height=0.12, 
                          color=(0.9, 0.2, 0.1), mass=0.5)
        
        self._add_cylinder("/World/Level2/DistractorCylinder", 
                          position=(0.50, 0.15, 0.06), 
                          radius=0.025, height=0.10, 
                          color=(0.2, 0.2, 0.8), mass=0.4)
        
        # Level 3: Environmental objects
        self._add_cylinder("/World/Level3/TargetCylinder", 
                          position=(0.55, 0.0, 0.06), 
                          radius=0.03, height=0.12, 
                          color=(0.9, 0.2, 0.1), mass=0.5)
        
        self._add_cylinder("/World/Level3/Distractor1", 
                          position=(0.50, 0.15, 0.06), 
                          radius=0.025, height=0.10, 
                          color=(0.2, 0.2, 0.8), mass=0.4)
        
        self._add_cylinder("/World/Level3/Distractor2", 
                          position=(0.60, -0.10, 0.06), 
                          radius=0.035, height=0.14, 
                          color=(0.2, 0.8, 0.2), mass=0.6)
        
        # Level 4: Multi-object scene
        self._add_cylinder("/World/Level4/TargetCylinder", 
                          position=(0.55, 0.0, 0.06), 
                          radius=0.03, height=0.12, 
                          color=(0.9, 0.2, 0.1), mass=0.5)
        
        self._add_cylinder("/World/Level4/Distractor1", 
                          position=(0.50, 0.15, 0.06), 
                          radius=0.025, height=0.10, 
                          color=(0.2, 0.2, 0.8), mass=0.4)
        
        self._add_cylinder("/World/Level4/Distractor2", 
                          position=(0.60, -0.10, 0.06), 
                          radius=0.035, height=0.14, 
                          color=(0.2, 0.8, 0.2), mass=0.6)
        
        self._add_cylinder("/World/Level4/Distractor3", 
                          position=(0.45, -0.05, 0.06), 
                          radius=0.028, height=0.11, 
                          color=(0.8, 0.8, 0.2), mass=0.45)
        
        # Level 5: Maximum complexity
        self._add_cylinder("/World/Level5/TargetCylinder", 
                          position=(0.55, 0.0, 0.06), 
                          radius=0.03, height=0.12, 
                          color=(0.9, 0.2, 0.1), mass=0.5)
        
        self._add_cylinder("/World/Level5/SimilarDistractor1", 
                          position=(0.50, 0.15, 0.06), 
                          radius=0.032, height=0.13, 
                          color=(0.8, 0.1, 0.1), mass=0.52)  # Similar to target
        
        self._add_cylinder("/World/Level5/SimilarDistractor2", 
                          position=(0.60, -0.10, 0.06), 
                          radius=0.028, height=0.11, 
                          color=(0.7, 0.15, 0.15), mass=0.48)  # Similar to target
        
        self._add_cylinder("/World/Level5/DifferentDistractor1", 
                          position=(0.45, -0.05, 0.06), 
                          radius=0.025, height=0.10, 
                          color=(0.2, 0.8, 0.2), mass=0.4)
        
        self._add_cylinder("/World/Level5/DifferentDistractor2", 
                          position=(0.65, 0.08, 0.06), 
                          radius=0.040, height=0.16, 
                          color=(0.2, 0.2, 0.8), mass=0.7)
        
        log("✅ Test objects added for all complexity levels")
    
    def _add_cylinder(self, path: str, position: Tuple[float, float, float], 
                     radius: float, height: float, color: Tuple[float, float, float], 
                     mass: float):
        """Add a cylinder with physics properties."""
        cylinder_prim = UsdGeom.Cylinder.Define(self.stage, path)
        
        # Set cylinder properties
        cylinder_prim.CreateRadiusAttr(radius)
        cylinder_prim.CreateHeightAttr(height)
        
        # Set position
        xform = UsdGeom.Xformable(cylinder_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))
        
        # Set color
        UsdGeom.Gprim.GetDisplayColorAttr(cylinder_prim).Set([color])
        
        # Add physics
        UsdPhysics.RigidBodyAPI.Apply(cylinder_prim.GetPrim())
        UsdPhysics.MassAPI.Apply(cylinder_prim.GetPrim()).CreateMassAttr(mass)
        UsdPhysics.CollisionAPI.Apply(cylinder_prim.GetPrim())
    
    def _add_basic_lighting(self):
        """Add basic lighting setup."""
        log("Adding basic lighting...")
        
        # Create a simple sphere light for basic illumination
        light_path = "/World/MainLight"
        light_prim = UsdLux.SphereLight.Define(self.stage, light_path)
        light_prim.CreateIntensityAttr(1000.0)
        light_prim.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        light_prim.CreateRadiusAttr(0.1)
        
        # Position the light
        light_xform = UsdGeom.Xformable(light_prim)
        light_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.0))
        
        # Create a second sphere light for fill
        fill_light_path = "/World/FillLight"
        fill_light_prim = UsdLux.SphereLight.Define(self.stage, fill_light_path)
        fill_light_prim.CreateIntensityAttr(300.0)
        fill_light_prim.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))
        fill_light_prim.CreateRadiusAttr(0.1)
        
        # Position the fill light
        fill_light_xform = UsdGeom.Xformable(fill_light_prim)
        fill_light_xform.AddTranslateOp().Set(Gf.Vec3d(-1.0, 1.0, 1.5))
        
        log("✅ Basic lighting added")
    
    def _add_basic_camera(self):
        """Add a basic camera (will be edited on Windows)."""
        log("Adding basic camera...")
        
        camera_path = "/World/EvaluationCamera"
        camera_prim = UsdGeom.Camera.Define(self.stage, camera_path)
        
        # Set camera properties
        camera_prim.CreateFocalLengthAttr(24.0)
        camera_prim.CreateHorizontalApertureAttr(20.955)
        camera_prim.CreateVerticalApertureAttr(15.956)
        
        # Set initial position (will be adjusted on Windows)
        xform = UsdGeom.Xformable(camera_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(1.5, 1.5, 1.2))
        
        log("✅ Basic camera added (will be positioned on Windows)")
    
    def _add_ground_plane(self):
        """Add ground plane."""
        log("Adding ground plane...")
        
        ground_path = "/World/GroundPlane"
        ground_prim = UsdGeom.Plane.Define(self.stage, ground_path)
        ground_prim.CreateWidthAttr(2.0)
        ground_prim.CreateLengthAttr(2.0)
        
        # Position ground
        ground_xform = UsdGeom.Xformable(ground_prim)
        ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        
        # Add physics to ground
        UsdPhysics.CollisionAPI.Apply(ground_prim.GetPrim())
        
        log("✅ Ground plane added")
    
    def _create_scene_info(self, scene_file: Path):
        """Create scene information file."""
        log("Creating scene information file...")
        
        scene_info = {
            "scene_file": str(scene_file),
            "created_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "description": "Portable evaluation scene for cross-platform editing",
            "complexity_levels": {
                "level_1": {
                    "objects": ["TargetCylinder"],
                    "description": "Basic validation - single cylinder"
                },
                "level_2": {
                    "objects": ["TargetCylinder", "DistractorCylinder"],
                    "description": "Pose variation - multiple cylinders"
                },
                "level_3": {
                    "objects": ["TargetCylinder", "Distractor1", "Distractor2"],
                    "description": "Environmental challenges - varied objects"
                },
                "level_4": {
                    "objects": ["TargetCylinder", "Distractor1", "Distractor2", "Distractor3"],
                    "description": "Multi-object scenes - 4 objects"
                },
                "level_5": {
                    "objects": ["TargetCylinder", "SimilarDistractor1", "SimilarDistractor2", "DifferentDistractor1", "DifferentDistractor2"],
                    "description": "Maximum complexity - 5 objects with similar distractors"
                }
            },
            "editing_instructions": {
                "windows_editing": [
                    "1. Copy scene file to Windows machine",
                    "2. Open in Isaac Sim GUI",
                    "3. Position camera for optimal view",
                    "4. Adjust lighting as needed",
                    "5. Save the scene",
                    "6. Copy back to Linux machine"
                ],
                "camera_positioning": [
                    "Position camera to capture full robot workspace",
                    "Ensure clear view of pick-and-place operations",
                    "Good lighting and contrast",
                    "Minimal occlusion"
                ],
                "lighting_adjustments": [
                    "Adjust main light intensity and direction",
                    "Add fill lights if needed",
                    "Ensure good contrast for object detection",
                    "Test lighting across all complexity levels"
                ]
            }
        }
        
        info_file = scene_file.parent / f"{scene_file.stem}_info.json"
        with open(info_file, 'w') as f:
            json.dump(scene_info, f, indent=2)
        
        log(f"✅ Scene information saved to: {info_file}")

def main():
    """Main function for creating portable scene."""
    log("🎬 CREATING PORTABLE EVALUATION SCENE")
    log("This scene can be edited on Windows and copied back to Linux")
    log("")
    
    creator = PortableSceneCreator()
    
    # Create the scene
    scene_path = "/ros2_ws/assets/evaluation_scene.usd"
    if creator.create_evaluation_scene(scene_path):
        log("")
        log("✅ PORTABLE SCENE CREATED SUCCESSFULLY!")
        log("")
        log("📋 NEXT STEPS:")
        log("1. Copy the scene file to your Windows machine:")
        log(f"   scp {scene_path} your_windows_machine:/path/to/destination/")
        log("")
        log("2. On Windows, open Isaac Sim and load the scene:")
        log(f"   File → Open → {Path(scene_path).name}")
        log("")
        log("3. Edit the scene:")
        log("   - Position the camera for optimal view")
        log("   - Adjust lighting as needed")
        log("   - Save the scene")
        log("")
        log("4. Copy the edited scene back to Linux:")
        log(f"   scp your_windows_machine:/path/to/edited_scene.usd {scene_path}")
        log("")
        log("5. Use the edited scene in our evaluation framework")
        log("")
        log("📁 Files created:")
        log(f"   - Scene: {scene_path}")
        log(f"   - Info: {Path(scene_path).parent / 'evaluation_scene_info.json'}")
    else:
        log("❌ Failed to create portable scene")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
