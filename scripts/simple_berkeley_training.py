#!/usr/bin/env python3
"""
SIMPLE BERKELEY TRAINING TEST
============================

Simple test to verify Berkeley dataset can be loaded and used for training
without complex framework dependencies.

Author: NIVA Training Team
Date: 2025-01-02
Status: Basic Berkeley Training Test
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
import time

# Add our framework
sys.path.append('/home/todd/niva-nbot-eval/scripts')
from fair_comparison_training import FairComparisonConfig, StandardizedVisuoMotorPolicy

def simple_berkeley_training_test():
    """Simple test of Berkeley training setup"""
    
    print("🎯 SIMPLE BERKELEY TRAINING TEST")
    print("=" * 40)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Create model
    print(f"\n🏗️ CREATING MODEL")
    print("=" * 20)
    
    config = FairComparisonConfig()
    model = StandardizedVisuoMotorPolicy(
        config=config,
        approach_name="berkeley_test"
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model: {total_params:,} parameters")
    
    # Create dummy data to test forward pass
    print(f"\n🔬 TESTING FORWARD PASS")
    print("=" * 25)
    
    batch_size = 2
    seq_len = config.max_sequence_length
    image_size = config.image_size
    
    # Create dummy batch
    dummy_batch = {
        'images': torch.randn(seq_len, image_size[0], image_size[1], 3).to(device),
        'robot_states': torch.randn(seq_len, 15).to(device),  # 15D robot state
        'actions': torch.randn(seq_len, 7).to(device)  # 7D actions (Berkeley action space)
    }
    
    print(f"   Images shape: {dummy_batch['images'].shape}")
    print(f"   Robot states shape: {dummy_batch['robot_states'].shape}")
    print(f"   Actions shape: {dummy_batch['actions'].shape}")
    
    try:
        # Forward pass
        model.train()
        predicted_actions = model(dummy_batch)
        
        print(f"   Predicted actions shape: {predicted_actions.shape}")
        
        # Compute loss
        loss = criterion(predicted_actions, dummy_batch['actions'])
        print(f"   Loss: {loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ Forward/backward pass successful!")
        
        # Try a few training steps
        print(f"\n🚀 TESTING TRAINING LOOP")
        print("=" * 25)
        
        for step in range(5):
            # Create new dummy data
            dummy_batch['actions'] = torch.randn(seq_len, 7).to(device)
            
            optimizer.zero_grad()
            predicted_actions = model(dummy_batch)
            loss = criterion(predicted_actions, dummy_batch['actions'])
            loss.backward()
            optimizer.step()
            
            print(f"   Step {step + 1}: Loss {loss.item():.6f}")
        
        print(f"\n🎉 BERKELEY TRAINING TEST SUCCESSFUL!")
        print("=" * 40)
        print(f"✅ Model creation: OK")
        print(f"✅ Forward pass: OK")
        print(f"✅ Backward pass: OK")
        print(f"✅ Training loop: OK")
        print(f"\n🎯 Ready for real Berkeley dataset training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    success = simple_berkeley_training_test()
    
    if success:
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Load real Berkeley dataset")
        print(f"   2. Implement data parsing for TensorFlow records")
        print(f"   3. Execute full training with 989 episodes")
        print(f"   4. Compare results to DR training failure")
    else:
        print(f"\n❌ Fix issues before proceeding to real dataset")

if __name__ == "__main__":
    main()
