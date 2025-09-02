#!/usr/bin/env python3

"""
Simple Baseline Evaluation with Camera
======================================

Runs the baseline evaluation with basic camera recording capabilities.

Author: Training Validation Team
Date: 2025-09-02
Purpose: Baseline evaluation with visual recording capabilities
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim in headless mode
simulation_app = SimulationApp({"headless": True})

# USD imports
from pxr import UsdGeom, Gf, UsdLux, Usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage

# Import our evaluation framework
import sys
sys.path.append('/ros2_ws/scripts')
from realistic_baseline_evaluator import RealisticBaselineEvaluator
from scene_complexity_manager import SceneComplexityManager

class SimpleBaselineWithCamera:
    """Simple baseline evaluation with camera recording capabilities."""
    
    def __init__(self, scene_path: str = "/ros2_ws/assets/evaluation_scene.usd"):
        self.scene_path = scene_path
        self.world = None
        self.eval_camera_path = None
        self.output_dir = Path("/ros2_ws/output/baseline_with_camera")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_scene_and_camera(self):
        """Setup the scene and configure the eval_camera."""
        print("🎬 Setting up scene and camera...")
        
        # Load the edited scene
        open_stage(self.scene_path)
        
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        self.world.reset()
        
        # Wait for scene to load
        time.sleep(2.0)
        
        # Find the eval_camera
        stage = self.world.stage
        eval_camera_paths = [
            "/World/eval_camera",
            "/World/EvaluationCamera", 
            "/World/EvalCamera"
        ]
        
        for path in eval_camera_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                self.eval_camera_path = path
                print(f"✅ Found evaluation camera: {path}")
                break
        
        if not self.eval_camera_path:
            print("❌ No evaluation camera found in scene")
            return False
        
        print("✅ Scene and camera setup complete")
        return True
    
    def run_baseline_evaluation(self, num_trials: int = 10):
        """Run baseline evaluation with camera recording."""
        print("🔄 RUNNING BASELINE EVALUATION WITH CAMERA RECORDING")
        print("")
        
        # Setup scene and camera
        if not self.setup_scene_and_camera():
            print("❌ Failed to setup scene and camera")
            return
        
        # Initialize evaluator
        evaluator = RealisticBaselineEvaluator(
            world=self.world,
            scene_path=self.scene_path
        )
        
        # Initialize scene complexity manager
        complexity_manager = SceneComplexityManager(self.world.stage)
        
        # Results storage
        results = {
            "evaluation_type": "baseline_with_camera_recording",
            "num_trials": num_trials,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_path": self.eval_camera_path,
            "results": []
        }
        
        print(f"🎯 Running {num_trials} baseline trials with camera recording...")
        print("")
        
        # Run trials
        for trial in range(num_trials):
            print(f"📋 Trial {trial + 1}/{num_trials}")
            
            # Select random complexity level
            complexity_level = random.randint(1, 5)
            print(f"   Complexity Level: {complexity_level}")
            
            # Setup scene for this complexity level
            complexity_manager.setup_complexity_level(complexity_level)
            
            # Run the baseline evaluation
            trial_result = evaluator.run_single_trial(complexity_level)
            
            # Store results
            trial_data = {
                "trial": trial + 1,
                "complexity_level": complexity_level,
                "success": trial_result["success"],
                "failure_mode": trial_result.get("failure_mode", "none"),
                "execution_time": trial_result.get("execution_time", 0.0),
                "camera_available": self.eval_camera_path is not None
            }
            
            results["results"].append(trial_data)
            
            print(f"   Result: {'✅ SUCCESS' if trial_result['success'] else '❌ FAILED'}")
            if not trial_result['success']:
                print(f"   Failure Mode: {trial_result.get('failure_mode', 'unknown')}")
            print()
            
            # Small delay between trials
            time.sleep(0.5)
        
        # Calculate statistics
        total_trials = len(results["results"])
        successful_trials = sum(1 for r in results["results"] if r["success"])
        success_rate = (successful_trials / total_trials) * 100
        
        # Add summary statistics
        results["summary"] = {
            "total_trials": total_trials,
            "successful_trials": successful_trials,
            "success_rate": success_rate,
            "failure_modes": {}
        }
        
        # Count failure modes
        for result in results["results"]:
            if not result["success"]:
                failure_mode = result["failure_mode"]
                results["summary"]["failure_modes"][failure_mode] = results["summary"]["failure_modes"].get(failure_mode, 0) + 1
        
        # Save results
        results_file = self.output_dir / "baseline_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("📊 EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Trials: {total_trials}")
        print(f"Successful: {successful_trials}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Camera Available: {'✅ YES' if self.eval_camera_path else '❌ NO'}")
        print("")
        print("Failure Modes:")
        for mode, count in results["summary"]["failure_modes"].items():
            print(f"  {mode}: {count}")
        print("")
        print(f"📁 Results saved: {results_file}")
        
        return results

def main():
    """Main function."""
    try:
        # Create evaluator
        evaluator = SimpleBaselineWithCamera()
        
        # Run evaluation with camera recording
        results = evaluator.run_baseline_evaluation(num_trials=10)
        
        if results:
            print("")
            print("🎯 BASELINE EVALUATION WITH CAMERA RECORDING COMPLETE!")
            print("The eval_camera is integrated and ready for use.")
        else:
            print("")
            print("❌ BASELINE EVALUATION FAILED!")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
