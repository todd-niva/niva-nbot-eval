#!/usr/bin/env python3

"""
Integrate Edited Scene from Windows
==================================

This script helps integrate the scene edited on Windows back into our
evaluation framework, extracting camera and lighting configurations.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Integration of cross-platform edited scenes
"""

import json
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim in headless mode
simulation_app = SimulationApp({"headless": True})

# USD imports
from pxr import UsdGeom, Gf, UsdLux, Usd

class EditedSceneIntegrator:
    """Integrates edited scenes from Windows back into Linux evaluation framework."""
    
    def __init__(self, scene_path: str = "/ros2_ws/assets/evaluation_scene.usd"):
        self.scene_path = Path(scene_path)
        self.camera_configs = {}
        self.lighting_configs = {}
        
    def analyze_edited_scene(self) -> Dict[str, Any]:
        """Analyze the edited scene and extract configurations."""
        print(f"🔍 Analyzing edited scene: {self.scene_path}")
        
        if not self.scene_path.exists():
            print(f"❌ Scene file not found: {self.scene_path}")
            return {}
        
        # Import USD stage
        try:
            stage = Usd.Stage.Open(str(self.scene_path))
        except Exception as e:
            print(f"❌ Failed to open USD stage: {e}")
            return {}
        
        # Extract camera configurations
        self._extract_camera_configs(stage)
        
        # Extract lighting configurations
        self._extract_lighting_configs(stage)
        
        # Create integration report
        integration_report = {
            "scene_path": str(self.scene_path),
            "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cameras": self.camera_configs,
            "lighting": self.lighting_configs,
            "integration_ready": len(self.camera_configs) > 0
        }
        
        return integration_report
    
    def _extract_camera_configs(self, stage):
        """Extract camera configurations from the edited scene."""
        print("📷 Extracting camera configurations...")
        
        # Find all cameras in the scene
        cameras = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                cameras.append(prim)
        
        for camera_prim in cameras:
            camera_path = str(camera_prim.GetPath())
            print(f"   Found camera: {camera_path}")
            
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
            
            # Store configuration
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
                }
            }
            
            self.camera_configs[camera_path] = config
            print(f"   ✅ Camera configuration extracted")
    
    def _extract_lighting_configs(self, stage):
        """Extract lighting configurations from the edited scene."""
        print("💡 Extracting lighting configurations...")
        
        # Find all lights in the scene
        lights = []
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.Light):
                lights.append(prim)
        
        for light_prim in lights:
            light_path = str(light_prim.GetPath())
            print(f"   Found light: {light_path}")
            
            # Get light transform
            xform = UsdGeom.Xformable(light_prim)
            transform_matrix = xform.ComputeLocalToWorldTransform(0.0)
            position = transform_matrix.ExtractTranslation()
            
            # Get light properties
            light_schema = UsdLux.Light(light_prim)
            intensity = light_schema.GetIntensityAttr().Get()
            color = light_schema.GetColorAttr().Get()
            
            # Store configuration
            config = {
                "light_path": light_path,
                "position": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2])
                },
                "properties": {
                    "intensity": float(intensity),
                    "color": {
                        "r": float(color[0]),
                        "g": float(color[1]),
                        "b": float(color[2])
                    }
                }
            }
            
            self.lighting_configs[light_path] = config
            print(f"   ✅ Light configuration extracted")
    
    def save_integration_configs(self, output_dir: str = "/ros2_ws/config"):
        """Save extracted configurations for use in evaluation framework."""
        print("💾 Saving integration configurations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save camera configurations
        if self.camera_configs:
            camera_config_file = output_path / "edited_scene_cameras.json"
            with open(camera_config_file, 'w') as f:
                json.dump(self.camera_configs, f, indent=2)
            print(f"   ✅ Camera configs saved: {camera_config_file}")
        
        # Save lighting configurations
        if self.lighting_configs:
            lighting_config_file = output_path / "edited_scene_lighting.json"
            with open(lighting_config_file, 'w') as f:
                json.dump(self.lighting_configs, f, indent=2)
            print(f"   ✅ Lighting configs saved: {lighting_config_file}")
        
        # Save integration report
        integration_report = self.analyze_edited_scene()
        report_file = output_path / "scene_integration_report.json"
        with open(report_file, 'w') as f:
            json.dump(integration_report, f, indent=2)
        print(f"   ✅ Integration report saved: {report_file}")
    
    def generate_evaluation_integration_code(self) -> str:
        """Generate code to integrate the edited scene into evaluation scripts."""
        print("📝 Generating evaluation integration code...")
        
        integration_code = '''
# Integration code for edited scene
from scripts.integrate_edited_scene import EditedSceneIntegrator
import json
from pathlib import Path

# Load edited scene configurations
def load_edited_scene_configs():
    """Load camera and lighting configurations from edited scene."""
    config_dir = Path("/ros2_ws/config")
    
    # Load camera configs
    camera_config_file = config_dir / "edited_scene_cameras.json"
    if camera_config_file.exists():
        with open(camera_config_file, 'r') as f:
            camera_configs = json.load(f)
    else:
        camera_configs = {}
    
    # Load lighting configs
    lighting_config_file = config_dir / "edited_scene_lighting.json"
    if lighting_config_file.exists():
        with open(lighting_config_file, 'r') as f:
            lighting_configs = json.load(f)
    else:
        lighting_configs = {}
    
    return camera_configs, lighting_configs

# Apply edited scene configurations to stage
def apply_edited_scene_configs(stage, camera_configs, lighting_configs):
    """Apply camera and lighting configurations to the stage."""
    from pxr import UsdGeom, Gf, UsdLux
    
    # Apply camera configurations
    for camera_path, config in camera_configs.items():
        camera_prim = stage.GetPrimAtPath(camera_path)
        if camera_prim and camera_prim.IsA(UsdGeom.Camera):
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
    
    # Apply lighting configurations
    for light_path, config in lighting_configs.items():
        light_prim = stage.GetPrimAtPath(light_path)
        if light_prim and light_prim.IsA(UsdLux.Light):
            # Apply position
            xform = UsdGeom.Xformable(light_prim)
            position = Gf.Vec3d(
                config['position']['x'],
                config['position']['y'],
                config['position']['z']
            )
            xform.SetTranslate(position)
            
            # Apply light properties
            light_schema = UsdLux.Light(light_prim)
            light_schema.GetIntensityAttr().Set(config['properties']['intensity'])
            color = Gf.Vec3f(
                config['properties']['color']['r'],
                config['properties']['color']['g'],
                config['properties']['color']['b']
            )
            light_schema.GetColorAttr().Set(color)

# Usage in evaluation scripts:
# camera_configs, lighting_configs = load_edited_scene_configs()
# apply_edited_scene_configs(stage, camera_configs, lighting_configs)
'''
        
        return integration_code

def main():
    """Main function for integrating edited scene."""
    print("🔄 INTEGRATING EDITED SCENE FROM WINDOWS")
    print("")
    
    # Check if edited scene exists
    scene_path = "/ros2_ws/assets/evaluation_scene.usd"
    if not Path(scene_path).exists():
        print(f"❌ Edited scene not found: {scene_path}")
        print("")
        print("📋 Make sure you have:")
        print("1. Created the portable scene on Linux")
        print("2. Copied it to Windows")
        print("3. Edited it in Isaac Sim on Windows")
        print("4. Copied the edited scene back to Linux")
        return
    
    # Integrate the edited scene
    integrator = EditedSceneIntegrator(scene_path)
    
    # Analyze the scene
    integration_report = integrator.analyze_edited_scene()
    
    if integration_report.get("integration_ready", False):
        print("✅ Scene analysis complete!")
        print(f"   Found {len(integrator.camera_configs)} cameras")
        print(f"   Found {len(integrator.lighting_configs)} lights")
        
        # Save configurations
        integrator.save_integration_configs()
        
        # Generate integration code
        integration_code = integrator.generate_evaluation_integration_code()
        
        # Save integration code
        code_file = Path("/ros2_ws/config/evaluation_integration_code.py")
        with open(code_file, 'w') as f:
            f.write(integration_code)
        print(f"   ✅ Integration code saved: {code_file}")
        
        print("")
        print("🎯 INTEGRATION COMPLETE!")
        print("The edited scene configurations are now ready for use in evaluation scripts.")
        print("")
        print("📝 To use in evaluation scripts:")
        print("1. Import the integration functions")
        print("2. Load the configurations")
        print("3. Apply them to your stage")
        print("")
        print("💡 See the generated integration code for details.")
        
    else:
        print("❌ Scene integration not ready")
        print("Make sure the scene was properly edited on Windows")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
