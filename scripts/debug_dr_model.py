#!/usr/bin/env python3
"""
DEBUG DR MODEL PREDICTIONS
==========================

Debug script to examine what the DR model is actually predicting
and identify why it's achieving 0% success rate.

This will help us understand:
1. Are the model predictions reasonable?
2. Is there a scaling/normalization issue?
3. Is the synthetic physics too strict?
4. Is there a fundamental training issue?

Author: NIVA Training Team
Date: 2025-01-02
Status: Debug Analysis
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Any, Tuple

# Import our trained model architecture and training config
sys.path.append('/home/todd/niva-nbot-eval/scripts')
from fair_comparison_training import FairComparisonConfig, StandardizedVisuoMotorPolicy
from full_dr_training import DRTrainingConfig

def debug_model_predictions(model_path: str, num_samples: int = 10):
    """Debug model predictions to understand the 0% success rate"""
    
    print("🔍 DEBUGGING DR MODEL PREDICTIONS")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"📥 Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract configuration
    if 'dr_config' in checkpoint:
        dr_config = checkpoint['dr_config']
        config = dr_config.base_config
    else:
        config = FairComparisonConfig()
    
    # Initialize model
    model = StandardizedVisuoMotorPolicy(config, "debug").to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Architecture hash: {checkpoint.get('architecture_hash', 'unknown')}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with various input scenarios
    scenarios = [
        "random_inputs",
        "zero_inputs", 
        "typical_robot_state",
        "extreme_inputs"
    ]
    
    for scenario in scenarios:
        print(f"\n🧪 Testing scenario: {scenario}")
        print("-" * 30)
        
        # Generate test inputs based on scenario
        if scenario == "random_inputs":
            images = np.random.rand(1, 192, 192, 3).astype(np.float32)
            robot_states = np.random.rand(1, 15).astype(np.float32)
        elif scenario == "zero_inputs":
            images = np.zeros((1, 192, 192, 3), dtype=np.float32)
            robot_states = np.zeros((1, 15), dtype=np.float32)
        elif scenario == "typical_robot_state":
            images = np.random.rand(1, 192, 192, 3).astype(np.float32) * 0.5 + 0.25
            # Typical joint angles for UR10e
            robot_states = np.array([[0.1, -1.5, 1.8, -1.8, -1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype(np.float32)
        elif scenario == "extreme_inputs":
            images = np.ones((1, 192, 192, 3), dtype=np.float32)
            robot_states = np.ones((1, 15), dtype=np.float32) * 10.0
        
        # Get model predictions
        with torch.no_grad():
            batch = {
                'images': torch.FloatTensor(images).to(device),
                'robot_states': torch.FloatTensor(robot_states).to(device)
            }
            
            action_pred = model(batch)
            action_np = action_pred.cpu().numpy().flatten()
        
        print(f"   Input robot state: {robot_states.flatten()[:7]}")  # First 7 joints
        print(f"   Predicted action: {action_np}")
        print(f"   Action range: [{action_np.min():.4f}, {action_np.max():.4f}]")
        print(f"   Action mean: {action_np.mean():.4f}")
        print(f"   Action std: {action_np.std():.4f}")
        
        # Check for problematic predictions
        issues = []
        if np.any(np.isnan(action_np)):
            issues.append("NaN values detected")
        if np.any(np.isinf(action_np)):
            issues.append("Infinite values detected")
        if np.abs(action_np).max() > 10.0:
            issues.append(f"Extreme values (max: {np.abs(action_np).max():.2f})")
        if np.abs(action_np).max() < 1e-6:
            issues.append("Near-zero predictions")
        
        if issues:
            print(f"   ⚠️ Issues: {', '.join(issues)}")
        else:
            print(f"   ✅ Predictions look reasonable")

def analyze_training_convergence(model_path: str):
    """Analyze if the training actually converged properly"""
    
    print(f"\n📊 ANALYZING TRAINING CONVERGENCE")
    print("=" * 40)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'training_stats' in checkpoint:
        training_log = checkpoint['training_stats'].get('training_log', [])
        if training_log:
            print(f"   Training epochs: {len(training_log)}")
            
            # Extract loss progression
            losses = [epoch.get('avg_loss', 0) for epoch in training_log]
            epochs = [epoch.get('epoch', i+1) for i, epoch in enumerate(training_log)]
            
            print(f"   Loss progression:")
            for epoch, loss in zip(epochs, losses):
                print(f"     Epoch {epoch}: {loss:.6f}")
            
            # Check convergence
            if len(losses) >= 3:
                recent_losses = losses[-3:]
                if max(recent_losses) - min(recent_losses) < 0.001:
                    print(f"   ✅ Training converged (loss stabilized)")
                else:
                    print(f"   ⚠️ Training may not have converged (loss still changing)")
            
            # Check final loss value
            final_loss = losses[-1]
            if final_loss < 0.01:
                print(f"   ✅ Final loss is low ({final_loss:.6f})")
            elif final_loss > 0.1:
                print(f"   ⚠️ Final loss is high ({final_loss:.6f})")
            else:
                print(f"   🤔 Final loss is moderate ({final_loss:.6f})")
        else:
            print(f"   ❌ No training log found")
    else:
        print(f"   ❌ No training stats found")

def check_synthetic_physics_logic():
    """Check if the synthetic physics success conditions are too strict"""
    
    print(f"\n🔧 CHECKING SYNTHETIC PHYSICS LOGIC")
    print("=" * 40)
    
    # Simulate the success condition logic
    print("   Success condition requirements:")
    print("     1. Robot must be within 0.2 units of target object")
    print("     2. Gripper must be closed (action[6] > 0.5)")
    print("     3. Random success probability based on DR quality")
    
    # Test different scenarios
    scenarios = [
        {"distance": 0.1, "gripper": 0.8, "desc": "Good positioning + closed gripper"},
        {"distance": 0.15, "gripper": 0.3, "desc": "Good positioning + open gripper"},  
        {"distance": 0.5, "gripper": 0.8, "desc": "Poor positioning + closed gripper"},
        {"distance": 0.1, "gripper": 0.8, "desc": "Perfect positioning + closed gripper"}
    ]
    
    for scenario in scenarios:
        distance = scenario["distance"]
        gripper = scenario["gripper"]
        desc = scenario["desc"]
        
        # Simulate success logic
        if distance < 0.2 and gripper > 0.5:
            base_success_prob = 0.15 - (1 - 1) * 0.02  # Level 1
            dr_improvement = 3.0  # Expected DR improvement
            success_prob = min(0.85, base_success_prob * dr_improvement)
            status = f"ELIGIBLE (prob: {success_prob:.1%})"
        else:
            status = "FAILS CONDITIONS"
        
        print(f"     {desc}: {status}")
        print(f"       Distance: {distance:.2f} ({'✅' if distance < 0.2 else '❌'})")
        print(f"       Gripper: {gripper:.2f} ({'✅' if gripper > 0.5 else '❌'})")

def main():
    """Main debug execution"""
    model_path = "/home/todd/niva-nbot-eval/models/dr_trained_model_final.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Debug model predictions
    debug_model_predictions(model_path, num_samples=5)
    
    # Analyze training convergence
    analyze_training_convergence(model_path)
    
    # Check synthetic physics
    check_synthetic_physics_logic()
    
    print(f"\n🎯 LIKELY CAUSES OF 0% SUCCESS RATE:")
    print("=" * 45)
    print("1. Model predictions are too small/conservative")
    print("2. Action scaling mismatch between training and evaluation")
    print("3. Synthetic physics positioning logic is too strict")
    print("4. Training data was procedural, evaluation expects real actions")
    print("5. Model never learned meaningful grasping behaviors")
    print("")
    print("🔧 RECOMMENDED FIXES:")
    print("1. Scale up model predictions in evaluation")
    print("2. Relax synthetic physics success conditions")
    print("3. Add action range analysis to training")
    print("4. Implement more realistic success detection")

if __name__ == "__main__":
    main()


