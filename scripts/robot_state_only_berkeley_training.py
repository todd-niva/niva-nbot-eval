#!/usr/bin/env python3
"""
Robot State Only Berkeley Training: Focus on Real Robot Data Advantage
====================================================================

This script implements Berkeley training focused on robot states only,
bypassing image processing issues while leveraging the core advantage:
real robot state and action patterns from 989 human demonstrations.

Key Strategy:
- Use only robot states from Berkeley (skip images for now)
- Focus on the fundamental advantage: real vs synthetic control patterns
- Maintain identical architecture for fair comparison with DR
"""

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add fair comparison framework
sys.path.append('/home/todd/niva-nbot-eval/scripts')
from berkeley_dataset_parser import BerkeleyDatasetParser, BerkeleyConfig

@dataclass
class RobotStateOnlyConfig:
    """Configuration for robot-state-only Berkeley training"""
    
    # Core Training
    num_epochs: int = 20
    batch_size: int = 64  # Larger batch for state-only
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Berkeley Data
    dataset_path: str = "/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0"
    sample_timesteps_per_episode: int = 10  # More samples per episode
    
    # Device & Optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_mixed_precision: bool = True
    
    # Output
    save_dir: str = "/home/todd/niva-nbot-eval/robot_state_berkeley_training"

class SimpleRobotStatePolicy(nn.Module):
    """Simple policy that only uses robot states (no images)"""
    
    def __init__(self, config: RobotStateOnlyConfig):
        super().__init__()
        self.config = config
        
        # Simple MLP for robot state -> action mapping
        self.policy_network = nn.Sequential(
            nn.Linear(15, 128),  # Berkeley robot state is 15D
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 7)  # Berkeley actions are 7D
        )
        
        print(f"🤖 Simple Robot State Policy:")
        print(f"   Input: Robot State (15D)")
        print(f"   Output: Actions (7D)")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Parameters: {total_params:,}")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass using only robot states"""
        robot_states = batch['robot_states']  # [batch, 15]
        actions = self.policy_network(robot_states)  # [batch, 7]
        return actions

class RobotStateOnlyTrainer:
    """Berkeley trainer using only robot states"""
    
    def __init__(self, config: RobotStateOnlyConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        print("🎯 ROBOT STATE ONLY BERKELEY TRAINING")
        print("====================================")
        print(f"📂 Dataset: {config.dataset_path}")
        print(f"🤖 Focus: Real robot state patterns only")
        print(f"🚀 Device: {config.device}")
        print(f"💾 Save directory: {config.save_dir}")
        
    def setup_training(self):
        """Setup training infrastructure"""
        print("\n🚀 STARTING ROBOT STATE TRAINING")
        print("=================================")
        
        # Setup simple model
        self.model = SimpleRobotStatePolicy(self.config)
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup mixed precision
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
            print("✅ Mixed precision enabled")
        
    def setup_dataset(self):
        """Setup Berkeley dataset for robot states only"""
        print("\n📊 SETTING UP ROBOT STATE DATASET")
        print("==================================")
        
        # Configure Berkeley parser
        berkeley_config = BerkeleyConfig(
            dataset_path=self.config.dataset_path,
            batch_size=1,  # We'll handle batching ourselves
            max_sequence_length=50,
            use_hand_camera=False,  # Skip images
            use_language=False  # Skip language
        )
        
        self.dataset_parser = BerkeleyDatasetParser(berkeley_config)
        self.raw_dataset = self.dataset_parser.create_dataset('train')
        
        # Convert to robot state samples
        self.robot_state_samples = self._create_robot_state_samples()
        
        print(f"📁 Generated {len(self.robot_state_samples)} robot state samples")
        print(f"📊 Dataset Statistics:")
        print(f"   • Samples: {len(self.robot_state_samples)}")
        print(f"   • Batch size: {self.config.batch_size}")
        
    def _create_robot_state_samples(self) -> List[Dict]:
        """Convert Berkeley episodes to robot state samples"""
        samples = []
        
        print("🔄 Extracting robot state patterns from Berkeley dataset...")
        
        for i, batch in enumerate(self.raw_dataset.take(800)):  # More episodes for robot states
            try:
                # Extract robot states and actions
                robot_states = batch['robot_states'].numpy()[0]  # [seq, 15]
                actions = batch['actions'].numpy()[0]  # [seq, 7]
                
                # Sample timesteps from episode
                seq_len = robot_states.shape[0]
                if seq_len < 2:
                    continue
                    
                # Sample random timesteps (excluding last to have next action)
                sample_indices = np.random.choice(
                    seq_len - 1, 
                    min(self.config.sample_timesteps_per_episode, seq_len - 1),
                    replace=False
                )
                
                for idx in sample_indices:
                    sample = {
                        'robot_state': robot_states[idx],  # [15] - current state
                        'action': actions[idx + 1],       # [7] - next action
                    }
                    samples.append(sample)
                    
                if i % 100 == 0:
                    print(f"   Processed {i} episodes -> {len(samples)} samples")
                    
            except Exception as e:
                print(f"   ⚠️ Episode {i} failed: {e}")
                continue
        
        print(f"✅ Generated {len(samples)} robot state training samples")
        return samples
    
    def create_batch(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Create a batch from robot state samples"""
        batch_size = len(samples)
        
        # Stack samples
        robot_states = np.stack([s['robot_state'] for s in samples])  # [batch, 15]
        actions = np.stack([s['action'] for s in samples])  # [batch, 7]
        
        # Convert to torch tensors
        batch = {
            'robot_states': torch.from_numpy(robot_states).float().to(self.device),  # [batch, 15]
            'actions': torch.from_numpy(actions).float().to(self.device)  # [batch, 7]
        }
        
        return batch
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\n🚀 EPOCH {self.epoch + 1}/{self.config.num_epochs}")
        print("=" * 30)
        
        # Shuffle samples
        shuffled_samples = self.robot_state_samples.copy()
        np.random.shuffle(shuffled_samples)
        
        # Create batches
        batch_size = self.config.batch_size
        num_samples = len(shuffled_samples)
        
        start_time = time.time()
        successful_batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch_samples = shuffled_samples[i:i + batch_size]
            if len(batch_samples) < batch_size // 2:  # Skip small final batch
                continue
                
            try:
                # Create batch
                batch = self.create_batch(batch_samples)
                
                # Forward pass
                if self.config.use_mixed_precision:
                    with autocast():
                        predicted_actions = self.model(batch)
                        loss = nn.MSELoss()(predicted_actions, batch['actions'])
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    predicted_actions = self.model(batch)
                    loss = nn.MSELoss()(predicted_actions, batch['actions'])
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                successful_batches += 1
                self.global_step += 1
                
                if successful_batches % 20 == 0:
                    print(f"   Batch {successful_batches}: Loss {loss.item():.6f}")
                    
            except Exception as e:
                print(f"   ⚠️ Batch {num_batches} failed: {e}")
                continue
            
            num_batches += 1
        
        epoch_time = time.time() - start_time
        avg_epoch_loss = epoch_loss / max(successful_batches, 1)
        
        print(f"✅ Epoch completed: Loss {avg_epoch_loss:.6f} | Successful batches: {successful_batches} | Time: {epoch_time:.1f}s")
        
        return {
            'epoch_loss': avg_epoch_loss,
            'num_batches': successful_batches,
            'epoch_time': epoch_time
        }
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.config.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = os.path.join(self.config.save_dir, f"{name}.pth")
        torch.save(checkpoint, save_path)
        print(f"💾 Checkpoint saved: {save_path}")
    
    def train(self) -> Dict:
        """Execute full training"""
        self.setup_training()
        self.setup_dataset()
        
        if len(self.robot_state_samples) == 0:
            print("❌ No robot state samples generated!")
            return {'error': 'No training data'}
        
        epochs = []
        losses = []
        times = []
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_results = self.train_epoch()
            current_loss = epoch_results['epoch_loss']
            
            epochs.append(epoch + 1)
            losses.append(current_loss)
            times.append(epoch_results['epoch_time'])
            
            # Save checkpoint
            self.save_checkpoint(f"robot_state_epoch_{epoch + 1}")
            
            # Check for improvement
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint("robot_state_best_model")
        
        # Save final results
        results = {
            'epochs': epochs,
            'losses': losses,
            'times': times,
            'best_loss': self.best_loss,
            'total_samples': len(self.robot_state_samples),
            'architecture': 'robot_state_only'
        }
        
        results_path = os.path.join(self.config.save_dir, "robot_state_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n🎉 ROBOT STATE BERKELEY TRAINING COMPLETE")
        print("==========================================")
        print(f"📊 Final Results:")
        print(f"   • Epochs completed: {len(epochs)}")
        print(f"   • Best loss: {self.best_loss:.6f}")
        print(f"   • Total samples: {len(self.robot_state_samples)}")
        print(f"   • Results saved: {results_path}")
        
        return results

def main():
    """Main training execution"""
    config = RobotStateOnlyConfig()
    trainer = RobotStateOnlyTrainer(config)
    results = trainer.train()
    return results

if __name__ == "__main__":
    results = main()
    print(f"Final results: {results}")
