#!/usr/bin/env python3

"""
Camera Configuration Integration for Evaluation Framework
========================================================

This script provides utilities to integrate saved camera configurations
into our training validation evaluation framework.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Camera integration for evaluation recordings
"""

import json
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

# USD imports
from pxr import UsdGeom, Gf

class CameraConfigManager:
    """Manages camera configurations for evaluation framework."""
    
    def __init__(self, config_dir: str = "/ros2_ws/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_camera_config(self, config_name: str = "evaluation_camera") -> Optional[Dict]:
        """Load camera configuration from file."""
        config_file = self.config_dir / f"{config_name}.json"
        
        if not config_file.exists():
            print(f"❌ Camera configuration not found: {config_file}")
            return None
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Camera configuration loaded: {config_name}")
        return config
    
    def save_camera_config(self, config: Dict, config_name: str = "evaluation_camera"):
        """Save camera configuration to file."""
        config_file = self.config_dir / f"{config_name}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Camera configuration saved: {config_file}")
    
    def create_camera_from_config(self, stage, config: Dict, camera_path: str = "/World/EvaluationCamera"):
        """Create a camera in the stage using the saved configuration."""
        # Create camera prim
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        
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
        
        print(f"✅ Camera created at: {camera_path}")
        return camera_prim
    
    def get_camera_transform_matrix(self, config: Dict) -> Gf.Matrix4d:
        """Get the camera transform matrix from configuration."""
        # Create position vector
        position = Gf.Vec3d(
            config['position']['x'],
            config['position']['y'],
            config['position']['z']
        )
        
        # Create rotation quaternion
        rotation_quat = Gf.Quatd(
            config['rotation']['w'],
            config['rotation']['x'],
            config['rotation']['y'],
            config['rotation']['z']
        )
        
        # Create transform matrix
        transform_matrix = Gf.Matrix4d()
        transform_matrix.SetTranslateOnly(position)
        transform_matrix.SetRotateOnly(rotation_quat)
        
        return transform_matrix
    
    def get_camera_view_matrix(self, config: Dict) -> Gf.Matrix4d:
        """Get the camera view matrix (inverse of transform matrix)."""
        transform_matrix = self.get_camera_transform_matrix(config)
        return transform_matrix.GetInverse()
    
    def get_camera_projection_matrix(self, config: Dict, width: int = 1920, height: int = 1080) -> Gf.Matrix4d:
        """Get the camera projection matrix."""
        focal_length = config['camera_properties']['focal_length']
        horizontal_aperture = config['camera_properties']['horizontal_aperture']
        vertical_aperture = config['camera_properties']['vertical_aperture']
        
        # Calculate field of view
        fov_x = 2 * math.atan(horizontal_aperture / (2 * focal_length))
        fov_y = 2 * math.atan(vertical_aperture / (2 * focal_length))
        
        # Create projection matrix
        projection_matrix = Gf.Matrix4d()
        projection_matrix.SetIdentity()
        
        # Set perspective projection
        aspect_ratio = width / height
        near_plane = 0.1
        far_plane = 100.0
        
        # Calculate projection parameters
        f = 1.0 / math.tan(fov_y / 2.0)
        
        projection_matrix[0, 0] = f / aspect_ratio
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
        projection_matrix[2, 3] = (2 * far_plane * near_plane) / (near_plane - far_plane)
        projection_matrix[3, 2] = -1.0
        projection_matrix[3, 3] = 0.0
        
        return projection_matrix

def integrate_camera_into_evaluation_script(script_path: str, camera_config_name: str = "evaluation_camera"):
    """Add camera integration code to an evaluation script."""
    
    camera_integration_code = f'''
# Camera Configuration Integration
from scripts.camera_config_integration import CameraConfigManager

# Initialize camera manager
camera_manager = CameraConfigManager()

# Load camera configuration
camera_config = camera_manager.load_camera_config("{camera_config_name}")
if camera_config:
    # Create camera in stage
    camera_prim = camera_manager.create_camera_from_config(stage, camera_config, "/World/EvaluationCamera")
    print("✅ Evaluation camera configured from saved settings")
else:
    print("⚠️  Using default camera configuration")
    # Fallback to default camera setup
    cam_path = "/World/EvaluationCamera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    # ... default camera setup code ...
'''
    
    print("📝 Camera integration code:")
    print(camera_integration_code)
    print("")
    print("💡 Add this code to your evaluation scripts to use the saved camera configuration")

def main():
    """Main function for camera configuration management."""
    print("🎥 CAMERA CONFIGURATION INTEGRATION")
    print("")
    
    manager = CameraConfigManager()
    
    # Check if camera configuration exists
    config = manager.load_camera_config("evaluation_camera")
    
    if config:
        print("✅ Camera configuration found!")
        print(f"   Position: ({config['position']['x']:.3f}, {config['position']['y']:.3f}, {config['position']['z']:.3f})")
        print(f"   Focal Length: {config['camera_properties']['focal_length']:.1f}mm")
        print(f"   Timestamp: {config['timestamp']}")
        print("")
        print("🔧 To integrate this camera into evaluation scripts:")
        integrate_camera_into_evaluation_script("your_evaluation_script.py")
    else:
        print("❌ No camera configuration found")
        print("")
        print("📋 To set up a camera configuration:")
        print("1. Run: python setup_evaluation_camera.py")
        print("2. Position the camera in Isaac Sim")
        print("3. Run: python setup_evaluation_camera.py --save")
        print("4. Then run this script again to integrate the camera")

if __name__ == "__main__":
    main()
