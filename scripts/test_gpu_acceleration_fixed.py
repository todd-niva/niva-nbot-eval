#!/usr/bin/env python3

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
