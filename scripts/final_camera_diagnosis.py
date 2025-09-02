#!/usr/bin/env python3

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from isaacsim import SimulationApp

# Launch Isaac Sim with disabled RTX and DLSS (based on search results)
simulation_app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080,
    "physics_gpu": 0,
    "physics_dt": 1.0/240.0,
    "rendering_dt": 1.0/60.0,
    "renderer": "RayTracedLighting",  # Try RTX first
    "anti_aliasing": 1,
    "multi_gpu": False,
    "rtx_reflections": False,  # Disable RTX features that may cause issues
    "rtx_gi": False,
    "rtx_shadows": False,
})

from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import open_stage
from omni.isaac.sensor import Camera
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf

def disable_physics_cache(stage):
    """Disable physics cache as suggested in search results"""
    print("🔧 Disabling physics cache (search result fix)")
    try:
        # Find physics scene and disable fast cache
        for prim in stage.Traverse():
            if prim.GetTypeName() == "PhysicsScene":
                # Add attributes to disable fast cache
                prim.CreateAttribute("physics:updateToUsd", Sdf.ValueTypeNames.Bool).Set(True)
                prim.CreateAttribute("physics:useFastCache", Sdf.ValueTypeNames.Bool).Set(False)
                print("   ✅ Disabled physics fast cache")
                break
    except Exception as e:
        print(f"   ⚠️ Could not modify physics cache: {e}")

def fix_camera_exposure_settings(camera_prim):
    """Fix camera exposure and gain settings to prevent white images"""
    try:
        camera = UsdGeom.Camera(camera_prim)
        
        # Set reasonable exposure settings
        camera.CreateAttribute("camera:exposureCompensation", Sdf.ValueTypeNames.Float).Set(0.0)
        camera.CreateAttribute("camera:exposure", Sdf.ValueTypeNames.Float).Set(0.0)
        
        # Set manual exposure mode if available
        camera.CreateAttribute("camera:exposureMode", Sdf.ValueTypeNames.Token).Set("manual")
        
        # Set reasonable ISO/gain
        camera.CreateAttribute("camera:iso", Sdf.ValueTypeNames.Float).Set(100.0)
        
        print(f"   ✅ Fixed exposure settings for {camera_prim.GetPath()}")
        return True
    except Exception as e:
        print(f"   ⚠️ Could not fix exposure for {camera_prim.GetPath()}: {e}")
        return False

def create_debug_lighting_multiple_approaches(stage):
    """Try multiple lighting approaches based on search results"""
    print("💡 Creating debug lighting with multiple approaches")
    
    # Remove all existing lights first
    lights_to_remove = []
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.Light):
            lights_to_remove.append(str(prim.GetPath()))
    
    for light_path in lights_to_remove:
        stage.RemovePrim(light_path)
        print(f"   - Removed existing light: {light_path}")
    
    # Approach 1: Single very strong directional light
    main_light_path = "/World/DebugMainLight"
    main_light = UsdLux.DirectionalLight.Define(stage, main_light_path)
    main_light.CreateIntensityAttr(10000.0)  # Very strong
    main_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    main_light.CreateEnableColorTemperatureAttr(False)
    
    main_xform = UsdGeom.Xformable(main_light)
    main_xform.AddRotateXYZOp().Set(Gf.Vec3f(-30, 30, 0))
    
    # Approach 2: Add ambient/environment light  
    env_light_path = "/World/DebugEnvLight"
    env_light = UsdLux.SphereLight.Define(stage, env_light_path)
    env_light.CreateIntensityAttr(1000.0)
    env_light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))
    env_light.CreateRadiusAttr(5.0)  # Large radius for ambient effect
    
    env_xform = UsdGeom.Xformable(env_light)
    env_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))
    
    # Approach 3: Add point light at camera position for each camera
    return [main_light_path, env_light_path]

def run_comprehensive_camera_diagnosis():
    """Run comprehensive camera diagnosis with multiple fixes"""
    print("🔍 COMPREHENSIVE CAMERA DIAGNOSIS")
    
    # Create output directory
    output_dir = Path("/ros2_ws/output/comprehensive_camera_diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize world
        world = World(stage_units_in_meters=1.0)
        
        # Load scene
        print("📥 Loading scene...")
        open_stage("/ros2_ws/assets/evaluation_scene.usd")
        
        print("🔧 Applying fixes from search results...")
        
        # Apply physics cache fix
        disable_physics_cache(world.stage)
        
        # Create proper lighting
        light_paths = create_debug_lighting_multiple_approaches(world.stage)
        
        # Find all cameras and fix their settings
        cameras_found = []
        for prim in world.stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                cameras_found.append(prim)
                fix_camera_exposure_settings(prim)
        
        print(f"📷 Found {len(cameras_found)} cameras:")
        for cam in cameras_found:
            print(f"   - {cam.GetName()} at {cam.GetPath()}")
        
        # Reset world after modifications
        print("🔄 Resetting world...")
        world.reset()
        
        # Run simulation for several steps (search result suggestion)
        print("⏳ Running simulation steps before capture...")
        for i in range(180):  # 3 seconds at 60 FPS
            world.step(render=True)
            if i % 60 == 0:
                print(f"   Step {i}/180")
        
        # Initialize camera sensors
        print("📷 Initializing camera sensors...")
        camera_sensors = {}
        for cam_prim in cameras_found:
            try:
                sensor = Camera(
                    prim_path=str(cam_prim.GetPath()),
                    name=f"{cam_prim.GetName()}_diagnosis",
                    frequency=5,  # Low frequency for stability
                    resolution=(1920, 1080),
                    dt=0.2
                )
                sensor.initialize()
                camera_sensors[cam_prim.GetName()] = {
                    'sensor': sensor,
                    'prim': cam_prim
                }
                print(f"   ✅ {cam_prim.GetName()}")
            except Exception as e:
                print(f"   ❌ {cam_prim.GetName()}: {e}")
        
        # Capture with multiple timing approaches
        print("📸 Capturing with multiple approaches...")
        
        successful_captures = 0
        results = []
        
        for camera_name, cam_data in camera_sensors.items():
            print(f"\n🎯 Diagnosing {camera_name}")
            
            sensor = cam_data['sensor']
            
            # Approach 1: Immediate capture
            print("   📸 Approach 1: Immediate capture")
            frame = sensor.get_current_frame()
            result = analyze_and_save_frame(frame, camera_name, "immediate", output_dir)
            results.append(result)
            
            # Approach 2: After additional steps
            print("   📸 Approach 2: After 10 simulation steps")
            for _ in range(10):
                world.step(render=True)
            
            frame = sensor.get_current_frame()
            result = analyze_and_save_frame(frame, camera_name, "after_steps", output_dir)
            results.append(result)
            
            # Approach 3: Multiple attempts
            print("   📸 Approach 3: Best of 5 attempts")
            best_frame = None
            best_score = -1
            
            for attempt in range(5):
                world.step(render=True)
                time.sleep(0.1)  # Small delay
                frame = sensor.get_current_frame()
                
                if frame and "rgba" in frame:
                    rgba = frame["rgba"]
                    # Score based on value range and unique pixels
                    value_range = rgba.max() - rgba.min()
                    unique_count = len(np.unique(rgba))
                    score = value_range * unique_count
                    
                    print(f"     Attempt {attempt+1}: score={score:.1f} (range={value_range:.3f}, unique={unique_count})")
                    
                    if score > best_score:
                        best_score = score
                        best_frame = frame
            
            result = analyze_and_save_frame(best_frame, camera_name, "best_attempt", output_dir)
            results.append(result)
            
            if any(r['success'] for r in results[-3:]):
                successful_captures += 1
        
        # Save comprehensive results
        diagnosis_report = {
            'timestamp': time.time(),
            'total_cameras': len(cameras_found),
            'successful_cameras': successful_captures,
            'rendering_config': {
                'renderer': 'RayTracedLighting',
                'rtx_disabled': True,
                'physics_cache_disabled': True,
                'strong_lighting': True
            },
            'fixes_applied': [
                'Physics cache disabled (Update to USD enabled)',
                'Camera exposure settings fixed',
                'Multiple lighting approaches',
                'Extended simulation steps before capture',
                'Multiple capture timing strategies'
            ],
            'capture_results': results
        }
        
        report_path = output_dir / "comprehensive_diagnosis.json"
        with open(report_path, 'w') as f:
            json.dump(diagnosis_report, f, indent=2)
        
        print(f"\n📊 DIAGNOSIS COMPLETE")
        print(f"   Successful cameras: {successful_captures}/{len(cameras_found)}")
        print(f"   Total attempts: {len(results)}")
        print(f"   Report: {report_path}")
        
        return diagnosis_report
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_and_save_frame(frame, camera_name, approach, output_dir):
    """Analyze frame quality and save if valid"""
    result = {
        'camera': camera_name,
        'approach': approach,
        'success': False,
        'timestamp': time.time()
    }
    
    try:
        if frame is None or "rgba" not in frame:
            result['error'] = "No frame data"
            print(f"     ❌ No frame data")
            return result
        
        rgba_data = frame["rgba"]
        
        # Analysis
        value_range = rgba_data.max() - rgba_data.min()
        unique_pixels = len(np.unique(rgba_data))
        mean_brightness = rgba_data.mean()
        
        result.update({
            'value_range': float(value_range),
            'unique_pixels': int(unique_pixels),
            'mean_brightness': float(mean_brightness),
            'shape': list(rgba_data.shape)
        })
        
        # Quality assessment
        is_all_white = mean_brightness > 0.95 and value_range < 0.1
        is_all_black = mean_brightness < 0.05 and value_range < 0.1
        has_detail = unique_pixels > 100 and value_range > 0.2
        
        result.update({
            'is_all_white': is_all_white,
            'is_all_black': is_all_black,
            'has_detail': has_detail
        })
        
        # Save if it looks reasonable
        if has_detail or (not is_all_white and not is_all_black):
            # Convert and save
            if rgba_data.dtype != np.uint8:
                rgba_data = (rgba_data * 255).astype(np.uint8)
            
            rgb_data = rgba_data[:, :, :3]
            
            from PIL import Image
            image = Image.fromarray(rgb_data, 'RGB')
            
            timestamp = int(time.time())
            image_path = output_dir / f"{camera_name}_{approach}_{timestamp}.png"
            image.save(str(image_path))
            
            result['image_path'] = str(image_path)
            result['success'] = True
            
            status = "✅ Good" if has_detail else "⚠️ Poor quality"
            print(f"     {status}: {image_path}")
        else:
            status = "⚠️ All white" if is_all_white else "⚠️ All black"
            print(f"     {status}: range={value_range:.3f}, brightness={mean_brightness:.3f}")
            
    except Exception as e:
        result['error'] = str(e)
        print(f"     ❌ Error: {e}")
    
    return result

def main():
    try:
        results = run_comprehensive_camera_diagnosis()
        if results:
            print("✅ Comprehensive diagnosis completed successfully")
        return results
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
