#!/usr/bin/env python3
"""
Phase 2: Real Performance Testing Across All Complexity Levels
==============================================================

This script executes REAL performance testing (no mocked data) across all 5 complexity levels
to establish genuine baseline performance metrics for training validation.

Execution Plan:
- 10 trials per complexity level (50 total trials)
- Real robot control with physics simulation
- Actual success/failure detection
- Comprehensive performance statistics
- Statistical analysis and confidence intervals

Author: Training Validation Team
Date: 2025-09-01
Phase: 2 - Real Performance Data Collection
"""

import math
import sys
import time
import json
import statistics
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# Import Phase 2 components
from phase2_scene_complexity import ComplexityLevel, SceneComplexityManager
from phase2_baseline_controller import BaselineController


def log(msg: str) -> None:
    """Enhanced logging with timestamp for phase tracking."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


class RealPerformanceTester:
    """
    Real Performance Testing Framework
    
    Executes actual pick-place trials across all complexity levels
    with genuine physics simulation and robot control.
    """
    
    def __init__(self, trials_per_level: int = 10):
        self.trials_per_level = trials_per_level
        self.results = {}
        self.detailed_results = []
        
    def execute_comprehensive_testing(self, stage, world, robot_articulation) -> Dict[str, Any]:
        """
        Execute comprehensive performance testing across all complexity levels.
        
        Returns:
            Dictionary containing complete performance analysis
        """
        log("🎯 STARTING COMPREHENSIVE REAL PERFORMANCE TESTING")
        log(f"📊 Configuration: {self.trials_per_level} trials per level × 5 levels = {self.trials_per_level * 5} total trials")
        
        # Initialize components
        complexity_manager = SceneComplexityManager(stage, world, random_seed=42)
        baseline_controller = BaselineController(stage, world, robot_articulation)
        
        # Initialize baseline controller
        if not baseline_controller.initialize_robot_control():
            raise Exception("Failed to initialize baseline controller")
        
        # Execute testing for each complexity level
        overall_start_time = time.time()
        
        for level in ComplexityLevel:
            log(f"\n{'='*80}")
            log(f"🎭 TESTING COMPLEXITY LEVEL: {level.name}")
            log(f"{'='*80}")
            
            level_results = self._test_complexity_level(
                level, complexity_manager, baseline_controller
            )
            
            self.results[level.name] = level_results
            
            # Display level summary
            success_rate = level_results["success_rate"] * 100
            avg_time = level_results.get("average_execution_time", 0)
            
            log(f"📊 {level.name} SUMMARY:")
            log(f"   Success Rate: {level_results['successes']}/{level_results['total_trials']} ({success_rate:.1f}%)")
            log(f"   Average Time: {avg_time:.2f}s")
            log(f"   Failures: {level_results['failures']}")
            
        # Generate comprehensive analysis
        overall_time = time.time() - overall_start_time
        analysis = self._generate_comprehensive_analysis(overall_time)
        
        return analysis
    
    def _test_complexity_level(self, level: ComplexityLevel, 
                              complexity_manager: SceneComplexityManager,
                              baseline_controller: BaselineController) -> Dict[str, Any]:
        """Execute trials for a specific complexity level."""
        
        level_start_time = time.time()
        trial_results = []
        successes = 0
        failures = 0
        execution_times = []
        
        for trial_idx in range(self.trials_per_level):
            log(f"\n🔄 Trial {trial_idx + 1}/{self.trials_per_level} - {level.name}")
            
            trial_start_time = time.time()
            
            try:
                # Create scene for this trial
                scene_config = complexity_manager.create_scene(level, trial_index=trial_idx)
                
                # Execute pick-place cycle
                result = baseline_controller.execute_pick_place_cycle(level, scene_config)
                
                # Record detailed result
                trial_result = {
                    "trial": trial_idx + 1,
                    "level": level.name,
                    "success": result["success"],
                    "execution_time": result.get("execution_time"),
                    "strategy": result.get("strategy"),
                    "error_message": result.get("error_message"),
                    "scene_objects": len(scene_config["objects"]),
                    "lighting_intensity": scene_config["lighting"]["intensity"]
                }
                
                trial_results.append(trial_result)
                self.detailed_results.append(trial_result)
                
                if result["success"]:
                    successes += 1
                    execution_times.append(result["execution_time"])
                    log(f"   ✅ SUCCESS in {result['execution_time']:.2f}s")
                else:
                    failures += 1
                    log(f"   ❌ FAILURE: {result['error_message']}")
                    
            except Exception as e:
                failures += 1
                trial_result = {
                    "trial": trial_idx + 1,
                    "level": level.name,
                    "success": False,
                    "execution_time": None,
                    "strategy": None,
                    "error_message": f"Trial exception: {e}",
                    "scene_objects": 0,
                    "lighting_intensity": 0
                }
                trial_results.append(trial_result)
                self.detailed_results.append(trial_result)
                log(f"   ❌ TRIAL EXCEPTION: {e}")
            
            # Brief pause between trials
            time.sleep(0.5)
        
        # Calculate level statistics
        level_time = time.time() - level_start_time
        success_rate = successes / self.trials_per_level
        
        level_stats = {
            "total_trials": self.trials_per_level,
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate,
            "level_execution_time": level_time,
            "trial_results": trial_results
        }
        
        if execution_times:
            level_stats["average_execution_time"] = statistics.mean(execution_times)
            level_stats["execution_time_std"] = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            level_stats["min_execution_time"] = min(execution_times)
            level_stats["max_execution_time"] = max(execution_times)
        
        return level_stats
    
    def _generate_comprehensive_analysis(self, overall_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""
        
        log(f"\n{'='*80}")
        log("📊 GENERATING COMPREHENSIVE PERFORMANCE ANALYSIS")
        log(f"{'='*80}")
        
        # Overall statistics
        total_trials = sum(result["total_trials"] for result in self.results.values())
        total_successes = sum(result["successes"] for result in self.results.values())
        overall_success_rate = total_successes / total_trials
        
        # Success rates by level
        success_rates = {}
        for level_name, result in self.results.items():
            success_rates[level_name] = result["success_rate"]
        
        # Statistical analysis
        success_rate_values = list(success_rates.values())
        
        analysis = {
            "test_configuration": {
                "trials_per_level": self.trials_per_level,
                "total_trials": total_trials,
                "total_execution_time": overall_time,
                "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "overall_performance": {
                "total_successes": total_successes,
                "total_failures": total_trials - total_successes,
                "overall_success_rate": overall_success_rate,
                "overall_success_percentage": overall_success_rate * 100
            },
            "complexity_level_performance": {},
            "statistical_analysis": {
                "success_rate_mean": statistics.mean(success_rate_values),
                "success_rate_std": statistics.stdev(success_rate_values) if len(success_rate_values) > 1 else 0,
                "success_rate_range": {
                    "min": min(success_rate_values),
                    "max": max(success_rate_values)
                }
            },
            "detailed_results": self.detailed_results,
            "level_breakdown": self.results
        }
        
        # Add per-level analysis
        for level_name, result in self.results.items():
            analysis["complexity_level_performance"][level_name] = {
                "success_percentage": result["success_rate"] * 100,
                "success_ratio": f"{result['successes']}/{result['total_trials']}",
                "average_execution_time": result.get("average_execution_time", 0),
                "performance_tier": self._classify_performance(result["success_rate"])
            }
        
        return analysis
    
    def _classify_performance(self, success_rate: float) -> str:
        """Classify performance tier based on success rate."""
        if success_rate >= 0.8:
            return "EXCELLENT"
        elif success_rate >= 0.6:
            return "GOOD"
        elif success_rate >= 0.4:
            return "MODERATE"
        elif success_rate >= 0.2:
            return "POOR"
        else:
            return "CRITICAL"
    
    def save_results(self, analysis: Dict[str, Any], filename: str = "real_performance_results.json"):
        """Save performance analysis to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2)
            log(f"💾 Results saved to {filename}")
        except Exception as e:
            log(f"❌ Failed to save results: {e}")
    
    def print_executive_summary(self, analysis: Dict[str, Any]):
        """Print executive summary of performance testing."""
        
        log(f"\n{'='*80}")
        log("🎯 EXECUTIVE SUMMARY - REAL PERFORMANCE TESTING")
        log(f"{'='*80}")
        
        config = analysis["test_configuration"]
        overall = analysis["overall_performance"]
        
        log(f"📊 TEST CONFIGURATION:")
        log(f"   Total Trials: {config['total_trials']}")
        log(f"   Trials per Level: {config['trials_per_level']}")
        log(f"   Total Execution Time: {config['total_execution_time']:.1f} seconds")
        
        log(f"\n🎯 OVERALL PERFORMANCE:")
        log(f"   Success Rate: {overall['total_successes']}/{config['total_trials']} ({overall['overall_success_percentage']:.1f}%)")
        log(f"   Total Failures: {overall['total_failures']}")
        
        log(f"\n📈 COMPLEXITY LEVEL BREAKDOWN:")
        for level_name, perf in analysis["complexity_level_performance"].items():
            tier = perf["performance_tier"]
            pct = perf["success_percentage"]
            ratio = perf["success_ratio"]
            avg_time = perf["average_execution_time"]
            
            log(f"   {level_name:20s}: {ratio:5s} ({pct:4.1f}%) - {tier:8s} - Avg: {avg_time:.2f}s")
        
        # Statistical insights
        stats = analysis["statistical_analysis"]
        log(f"\n📊 STATISTICAL ANALYSIS:")
        log(f"   Mean Success Rate: {stats['success_rate_mean']*100:.1f}%")
        log(f"   Success Rate Std Dev: {stats['success_rate_std']*100:.1f}%")
        log(f"   Range: {stats['success_rate_range']['min']*100:.1f}% - {stats['success_rate_range']['max']*100:.1f}%")
        
        # Performance insights
        log(f"\n🔍 KEY INSIGHTS:")
        
        # Find best and worst performing levels
        level_perf = analysis["complexity_level_performance"]
        best_level = max(level_perf.items(), key=lambda x: x[1]["success_percentage"])
        worst_level = min(level_perf.items(), key=lambda x: x[1]["success_percentage"])
        
        log(f"   Best Performance: {best_level[0]} ({best_level[1]['success_percentage']:.1f}%)")
        log(f"   Worst Performance: {worst_level[0]} ({worst_level[1]['success_percentage']:.1f}%)")
        
        # Performance degradation
        complexity_order = ["LEVEL_1_BASIC", "LEVEL_2_POSE_VARIATION", "LEVEL_3_ENVIRONMENTAL", 
                          "LEVEL_4_MULTI_OBJECT", "LEVEL_5_MAXIMUM_CHALLENGE"]
        
        if len(complexity_order) == len(level_perf):
            degradation_rates = []
            for i in range(1, len(complexity_order)):
                prev_rate = level_perf[complexity_order[i-1]]["success_percentage"]
                curr_rate = level_perf[complexity_order[i]]["success_percentage"]
                degradation = prev_rate - curr_rate
                degradation_rates.append(degradation)
            
            if degradation_rates:
                avg_degradation = statistics.mean(degradation_rates)
                log(f"   Average Performance Degradation: {avg_degradation:.1f}% per level")


def main() -> None:
    """
    Phase 2 Main: Real Performance Testing
    
    Execute genuine performance testing across all 5 complexity levels
    with actual robot control and physics simulation.
    """
    log("🚀 PHASE 2: REAL PERFORMANCE TESTING - NO MOCKED DATA")
    
    # Initialize Isaac Sim
    from isaacsim.simulation_app import SimulationApp

    sim_config = {
        "headless": True,
        "renderer": "RayTracedLighting",
        "physics_dt": 1.0/60.0,
        "rendering_dt": 1.0/30.0,
        "physics_gpu": 0,
    }
    
    sim_app = SimulationApp(sim_config)

    import omni
    from pxr import UsdGeom, UsdLux, Gf, UsdPhysics
    from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation

    try:
        # Setup base environment
        log("🔧 Setting up robot environment for real performance testing")
        create_new_stage()
        add_reference_to_stage(
            "/ros2_ws/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd",
            "/World/UR10e_Robotiq_2F_140"
        )

        # Wait for stage ready
        usd_ctx = omni.usd.get_context()
        stage = None
        for _ in range(400):
            stage = usd_ctx.get_stage()
            if stage is not None:
                break
            sim_app.update()
            time.sleep(0.02)

        if stage is None:
            raise Exception("USD stage failed to load")

        # Initialize world
        world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
        
        # Add ground plane
        ground_geom = UsdGeom.Plane.Define(stage, "/World/GroundPlane/Geometry")
        ground_geom.CreateExtentAttr([(-2.0, -2.0, 0), (2.0, 2.0, 0)])
        ground_geom.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(ground_geom.GetPrim())
        ground_rigid = UsdPhysics.RigidBodyAPI.Apply(ground_geom.GetPrim())
        ground_rigid.CreateKinematicEnabledAttr().Set(True)

        # Initialize robot
        robot_path = "/World/UR10e_Robotiq_2F_140"
        robot_articulation = SingleArticulation(robot_path)
        
        world.reset()
        for _ in range(60):
            world.step(render=False)
            
        # Initialize articulation after world reset
        robot_articulation.initialize()

        log("✅ Robot environment ready for performance testing")

        # Create and execute performance tester
        performance_tester = RealPerformanceTester(trials_per_level=10)
        
        log("\n🎯 BEGINNING REAL PERFORMANCE DATA COLLECTION")
        log("⚠️  This will execute ACTUAL robot control with physics simulation")
        log("⚠️  No mocked or hardcoded data - only real results")
        
        # Execute comprehensive testing
        analysis = performance_tester.execute_comprehensive_testing(
            stage, world, robot_articulation
        )
        
        # Print executive summary
        performance_tester.print_executive_summary(analysis)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"real_performance_results_{timestamp}.json"
        performance_tester.save_results(analysis, filename)
        
        log(f"\n🎉 REAL PERFORMANCE TESTING COMPLETE")
        log(f"📊 {analysis['test_configuration']['total_trials']} trials executed")
        log(f"💾 Results saved to {filename}")
        log(f"🎯 Overall Success Rate: {analysis['overall_performance']['overall_success_percentage']:.1f}%")

    except Exception as e:
        log(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'world' in locals():
                world.stop()
            sim_app.close()
        except Exception as e:
            log(f"Cleanup warning: {e}")

    log("🏁 REAL PERFORMANCE TESTING COMPLETED")


if __name__ == "__main__":
    main()
