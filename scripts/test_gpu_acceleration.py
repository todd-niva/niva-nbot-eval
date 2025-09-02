#!/usr/bin/env python3

"""
Test GPU Acceleration
=====================

Quick test to verify GPU physics acceleration is working.
This will run a short physics simulation and monitor GPU utilization.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Verify GPU physics acceleration
"""

import time
import subprocess
import os
from pathlib import Path

# Isaac Sim imports with GPU acceleration
from isaacsim import SimulationApp

# GPU-optimized configuration
sim_config = {
    "headless": True,
    "width": 1280,
    "height": 720,
    "physics_gpu": 0,  # GPU acceleration enabled
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
    """Setup a physics scene with GPU acceleration."""
    print("🔧 Setting up GPU-accelerated physics scene...")
    
    # Load our scene with eval_camera
    scene_path = "/ros2_ws/assets/evaluation_scene.usd"
    open_stage(scene_path)
    
    world = World()
    world.reset()
    stage = world.stage
    
    # Configure PhysX scene for GPU acceleration
    physics_scene_path = "/physicsScene"
    if not stage.GetPrimAtPath(physics_scene_path):
        UsdPhysics.Scene.Define(stage, physics_scene_path)
    
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(physics_scene_path))
    
    # Enable GPU dynamics and optimization
    physx_scene.CreateEnableGPUDynamicsAttr(True)  # GPU dynamics enabled
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableStabilizationAttr(True)
    physx_scene.CreateBroadphaseTypeAttr("MBP")
    physx_scene.CreateSolverTypeAttr("TGS")
    
    # GPU-optimized PhysX settings
    physx_scene.CreateGpuMaxNumPartitionsAttr(8)
    physx_scene.CreateGpuCollisionStackSizeAttr(67108864)  # 64MB
    physx_scene.CreateGpuTempBufferCapacityAttr(16777216)  # 16MB
    physx_scene.CreateGpuMaxRigidContactCountAttr(524288)  # 512K contacts
    physx_scene.CreateGpuMaxRigidPatchCountAttr(163840)    # 160K patches
    physx_scene.CreateGpuFoundLostPairsCapacityAttr(262144)  # 256K pairs
    
    print("   ✅ GPU physics scene configured")
    return world

def run_gpu_acceleration_test():
    """Run GPU acceleration test."""
    print("🚀 GPU ACCELERATION TEST")
    print("=" * 40)
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
    
    # Run physics simulation stress test
    print("🔄 Running physics simulation stress test...")
    print("   (30 seconds of intensive physics simulation)")
    
    max_gpu_util = 0
    max_mem_usage = setup_mem
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < 30.0:  # 30 second test
        # Step the physics simulation
        world.step(render=False)  # No rendering to focus on physics
        step_count += 1
        
        # Monitor GPU every 100 steps
        if step_count % 100 == 0:
            gpu_util, mem_used, _ = monitor_gpu_usage()
            max_gpu_util = max(max_gpu_util, gpu_util)
            max_mem_usage = max(max_mem_usage, mem_used)
            
            elapsed = time.time() - start_time
            print(f"   {elapsed:.1f}s - GPU: {gpu_util}% util, {mem_used}MB mem (steps: {step_count})")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    steps_per_second = step_count / elapsed_time
    
    print("")
    print("✅ GPU acceleration test complete!")
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
        print("🚀 EXCELLENT: High GPU compute utilization detected!")
        print(f"   GPU acceleration is working effectively ({max_gpu_util}% peak)")
    elif max_gpu_util > 20:
        print("✅ GOOD: Moderate GPU utilization detected")
        print(f"   GPU acceleration is working ({max_gpu_util}% peak)")
    elif max_gpu_util > 5:
        print("⚠️ LIMITED: Low GPU utilization")
        print(f"   GPU acceleration may be working but underutilized ({max_gpu_util}% peak)")
    else:
        print("❌ PROBLEM: Very low GPU utilization")
        print(f"   GPU acceleration may not be working properly ({max_gpu_util}% peak)")
        print("   Check GPU physics configuration")
    
    print("")
    print("🔍 For comparison:")
    print("   CPU-only physics typically shows 0-2% GPU utilization")
    print("   GPU-accelerated physics should show 20-80% GPU utilization")
    
    return {
        "max_gpu_utilization": max_gpu_util,
        "max_memory_usage": max_mem_usage,
        "steps_per_second": steps_per_second,
        "total_steps": step_count,
        "duration": elapsed_time
    }

def main():
    """Main function."""
    try:
        results = run_gpu_acceleration_test()
        
        # Save results for comparison
        import json
        results_path = "/ros2_ws/output/gpu_acceleration_test.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error during GPU test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        simulation_app.close()

if __name__ == "__main__":
    main()
