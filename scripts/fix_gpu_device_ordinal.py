#!/usr/bin/env python3

"""
Fix GPU Device Ordinal
======================

Fixes the CUDA device ordinal mismatch that's preventing GPU acceleration.
The system was trying to use device 1, but we only have device 0.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Fix GPU device ordinal for proper acceleration
"""

import os
import re

def fix_device_ordinal_in_scripts():
    """Fix CUDA device ordinal in all relevant scripts."""
    
    scripts_to_fix = [
        "phase2_realistic_baseline_framework.py",
        "phase2_scene_complexity.py", 
        "phase3_domain_randomization_trainer.py",
        "test_gpu_acceleration.py"
    ]
    
    print("🔧 FIXING CUDA DEVICE ORDINAL")
    print("=" * 40)
    print("")
    
    for script_name in scripts_to_fix:
        script_path = f"/ros2_ws/scripts/{script_name}"
        
        print(f"📝 Processing {script_name}...")
        
        try:
            # Read the script
            with open(script_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Fix physics_gpu device ordinal
            # Change from physics_gpu: 1 to physics_gpu: 0
            content = re.sub(
                r'"physics_gpu":\s*1',
                '"physics_gpu": 0',
                content
            )
            
            # Fix CUDA device setting in simulation config
            content = re.sub(
                r'--/physics/cudaDevice=1',
                '--/physics/cudaDevice=0',
                content
            )
            
            # Check if changes were made
            if content != original_content:
                # Write the fixed version
                with open(script_path, 'w') as f:
                    f.write(content)
                print(f"   ✅ Fixed device ordinal in {script_name}")
            else:
                print(f"   ℹ️ No device ordinal changes needed in {script_name}")
                
        except Exception as e:
            print(f"   ❌ Error processing {script_name}: {e}")
    
    print("")
    print("✅ Device ordinal fix complete!")
    print("")
    print("🔧 Changes applied:")
    print("   • physics_gpu: 1 → physics_gpu: 0")
    print("   • cudaDevice=1 → cudaDevice=0")
    print("")

def verify_cuda_devices():
    """Verify available CUDA devices."""
    print("🔍 VERIFYING CUDA DEVICES")
    print("=" * 30)
    print("")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("Available CUDA devices:")
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    device_id, name, memory = parts[0], parts[1], parts[2]
                    print(f"   Device {device_id}: {name} ({memory})")
            
            # Determine the correct device to use (should be 0 for single GPU)
            if len(lines) == 1:
                print("")
                print("✅ Single GPU system detected")
                print("   Correct device ordinal: 0")
                return 0
            else:
                print("")
                print(f"⚠️ Multiple GPU system detected ({len(lines)} GPUs)")
                print("   Using device 0 (first GPU)")
                return 0
        else:
            print("❌ Could not query GPU devices")
            return None
            
    except Exception as e:
        print(f"❌ Error checking CUDA devices: {e}")
        return None

def create_fixed_test_script():
    """Create a fixed version of the GPU test with correct device ordinal."""
    
    test_script_content = '''#!/usr/bin/env python3

"""
Fixed GPU Acceleration Test
===========================

GPU acceleration test with correct CUDA device ordinal (0 instead of 1).

Author: Training Validation Team
Date: 2025-09-02
Purpose: Test GPU physics acceleration with correct device
"""

import time
import subprocess
import os
from pathlib import Path

# Isaac Sim imports with corrected GPU configuration
from isaacsim import SimulationApp

# GPU-optimized configuration with correct device ordinal
sim_config = {
    "headless": True,
    "width": 1280,
    "height": 720,
    "physics_gpu": 0,  # Use device 0 (correct ordinal)
    "physics_dt": 1.0/240.0,  # Higher frequency for GPU stress test
    "rendering_dt": 1.0/60.0,
}

simulation_app = SimulationApp(sim_config)

from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import open_stage
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd

def monitor_gpu_usage():
    """Monitor GPU usage during the test."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
            return int(gpu_util), int(mem_used), int(mem_total)
        return 0, 0, 0
    except:
        return 0, 0, 0

def setup_gpu_physics_scene():
    """Setup a physics scene with corrected GPU acceleration."""
    print("🔧 Setting up GPU-accelerated physics scene (device 0)...")
    
    # Load our scene with eval_camera
    scene_path = "/ros2_ws/assets/evaluation_scene.usd"
    open_stage(scene_path)
    
    world = World()
    world.reset()
    stage = world.stage
    
    # Configure PhysX scene for GPU acceleration on device 0
    physics_scene_path = "/physicsScene"
    if not stage.GetPrimAtPath(physics_scene_path):
        UsdPhysics.Scene.Define(stage, physics_scene_path)
    
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(physics_scene_path))
    
    # Enable GPU dynamics with correct device
    physx_scene.CreateEnableGPUDynamicsAttr(True)  # GPU dynamics enabled on device 0
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableStabilizationAttr(True)
    physx_scene.CreateBroadphaseTypeAttr("MBP")
    physx_scene.CreateSolverTypeAttr("TGS")
    
    # GPU-optimized PhysX settings for device 0
    physx_scene.CreateGpuMaxNumPartitionsAttr(8)
    physx_scene.CreateGpuCollisionStackSizeAttr(67108864)  # 64MB
    physx_scene.CreateGpuTempBufferCapacityAttr(16777216)  # 16MB
    physx_scene.CreateGpuMaxRigidContactCountAttr(524288)  # 512K contacts
    physx_scene.CreateGpuMaxRigidPatchCountAttr(163840)    # 160K patches
    physx_scene.CreateGpuFoundLostPairsCapacityAttr(262144)  # 256K pairs
    
    print("   ✅ GPU physics scene configured for device 0")
    return world

def run_corrected_gpu_test():
    """Run GPU acceleration test with corrected device ordinal."""
    print("🚀 CORRECTED GPU ACCELERATION TEST")
    print("=" * 45)
    print("")
    
    # Check initial GPU state
    initial_gpu_util, initial_mem, total_mem = monitor_gpu_usage()
    print(f"📊 Initial GPU state:")
    print(f"   Utilization: {initial_gpu_util}%")
    print(f"   Memory: {initial_mem}MB / {total_mem}MB")
    print("")
    
    # Setup physics scene
    world = setup_gpu_physics_scene()
    
    # Check GPU state after setup
    setup_gpu_util, setup_mem, _ = monitor_gpu_usage()
    print(f"📊 GPU state after scene setup:")
    print(f"   Utilization: {setup_gpu_util}%")
    print(f"   Memory: {setup_mem}MB / {total_mem}MB (+{setup_mem - initial_mem}MB)")
    print("")
    
    # Run shorter physics simulation test (10 seconds)
    print("🔄 Running corrected GPU physics test...")
    print("   (10 seconds of GPU physics simulation)")
    
    max_gpu_util = 0
    max_mem_usage = setup_mem
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < 10.0:  # 10 second test
        # Step the physics simulation
        world.step(render=False)  # No rendering to focus on physics
        step_count += 1
        
        # Monitor GPU every 50 steps
        if step_count % 50 == 0:
            gpu_util, mem_used, _ = monitor_gpu_usage()
            max_gpu_util = max(max_gpu_util, gpu_util)
            max_mem_usage = max(max_mem_usage, mem_used)
            
            elapsed = time.time() - start_time
            print(f"   {elapsed:.1f}s - GPU: {gpu_util}% util, {mem_used}MB mem (steps: {step_count})")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    steps_per_second = step_count / elapsed_time
    
    print("")
    print("✅ Corrected GPU test complete!")
    print("")
    print("📊 PERFORMANCE RESULTS:")
    print(f"   Duration: {elapsed_time:.1f} seconds")
    print(f"   Physics steps: {step_count}")
    print(f"   Steps per second: {steps_per_second:.1f}")
    print(f"   Max GPU utilization: {max_gpu_util}%")
    print(f"   Max memory usage: {max_mem_usage}MB")
    print(f"   Memory increase: +{max_mem_usage - initial_mem}MB")
    print("")
    
    # Evaluate results
    if max_gpu_util > 50:
        print("🚀 EXCELLENT: GPU acceleration working perfectly!")
        print(f"   High compute utilization achieved ({max_gpu_util}% peak)")
    elif max_gpu_util > 20:
        print("✅ GOOD: GPU acceleration working effectively")
        print(f"   Moderate compute utilization ({max_gpu_util}% peak)")
    elif max_gpu_util > 5:
        print("⚠️ LIMITED: Some GPU utilization detected")
        print(f"   Low but present compute utilization ({max_gpu_util}% peak)")
    else:
        print("❌ ISSUE: Still very low GPU utilization")
        print(f"   GPU compute not being used effectively ({max_gpu_util}% peak)")
    
    return {
        "max_gpu_utilization": max_gpu_util,
        "max_memory_usage": max_mem_usage,
        "steps_per_second": steps_per_second,
        "total_steps": step_count,
        "duration": elapsed_time,
        "device_ordinal_fixed": True
    }

def main():
    """Main function."""
    try:
        results = run_corrected_gpu_test()
        
        # Save results for comparison
        import json
        results_path = "/ros2_ws/output/gpu_acceleration_fixed_test.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error during corrected GPU test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        simulation_app.close()

if __name__ == "__main__":
    main()
'''
    
    # Write the fixed test script
    test_script_path = "/ros2_ws/scripts/test_gpu_acceleration_fixed.py"
    with open(test_script_path, 'w') as f:
        f.write(test_script_content)
    
    print(f"📝 Created fixed GPU test script: test_gpu_acceleration_fixed.py")

def main():
    """Main function to fix GPU device ordinal issues."""
    
    # Verify CUDA devices
    correct_device = verify_cuda_devices()
    
    print("")
    
    # Fix device ordinal in scripts
    fix_device_ordinal_in_scripts()
    
    # Create a fixed test script
    create_fixed_test_script()
    
    print("📋 NEXT STEPS:")
    print("1. Run the fixed GPU acceleration test")
    print("2. Verify GPU compute utilization is now working")
    print("3. Re-run baseline evaluation with proper GPU acceleration")
    print("")
    print("Expected outcome:")
    print("• No more CUDA device ordinal errors")
    print("• Sustained GPU compute utilization > 20%")
    print("• 3-10x performance improvement")

if __name__ == "__main__":
    main()
