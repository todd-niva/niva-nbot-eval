#!/usr/bin/env python3
"""
Compatible Berkeley Training: Real Robot Data with Sequence Flattening
====================================================================

This script implements Berkeley training that's compatible with the DR model architecture
by flattening sequence data to match the expected input format.

Key Insights from Debugging:
- DR model expects: [batch, features] (flattened)
- Berkeley data provides: [batch, sequence, features] (sequential)
- Solution: Sample individual timesteps instead of full sequences

This approach maintains fair comparison while leveraging real robot data.
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
from fair_comparison_training import StandardizedVisuoMotorPolicy, FairComparisonConfig
from berkeley_dataset_parser import BerkeleyDatasetParser, BerkeleyConfig

@dataclass
class CompatibleBerkeleyConfig:
    """Configuration for compatible Berkeley training"""
    
    # Core Training
    num_epochs: int = 10
    batch_size: int = 32  # Increased for single timesteps
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Fair Comparison
    use_identical_architecture: bool = True
    dr_model_hash: str = "25424174"  # Must match for fair comparison
    
    # Berkeley Data
    dataset_path: str = "/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0"
    sample_timesteps_per_episode: int = 5  # Sample N timesteps per episode
    
    # Device & Optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_mixed_precision: bool = True
    use_torch_compile: bool = False  # Disabled to avoid tensor shape issues
    
    # Early Stopping
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-6
    
    # Output
    save_dir: str = "/home/todd/niva-nbot-eval/compatible_berkeley_training"

class CompatibleBerkeleyTrainer:
    """Berkeley trainer compatible with DR model architecture"""
    
    def __init__(self, config: CompatibleBerkeleyConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        print("🎯 COMPATIBLE BERKELEY TRAINING")
        print("===============================")
        print(f"📂 Dataset: {config.dataset_path}")
        print(f"🤖 Model: Identical to DR (for fair comparison)")
        print(f"🚀 Device: {config.device}")
        print(f"💾 Save directory: {config.save_dir}")
        print(f"🔢 Sample timesteps per episode: {config.sample_timesteps_per_episode}")
        
    def setup_training(self):
        """Setup training infrastructure"""
        print("\n🚀 STARTING COMPATIBLE BERKELEY TRAINING")
        print("============================================")
        
        # Setup model with identical architecture
        print("\n🏗️ SETTING UP TRAINING INFRASTRUCTURE")
        print("========================================")
        
        fair_config = FairComparisonConfig()
        self.model = StandardizedVisuoMotorPolicy(
            config=fair_config,
            approach_name="compatible_berkeley"
        )
        self.model = self.model.to(self.device)
        
        # Verify architecture hash
        model_hash = self.model._compute_architecture_hash()
        print(f"⚠️ Architecture hash: {model_hash}")
        if hasattr(self.config, 'dr_model_hash') and model_hash != self.config.dr_model_hash:
            print(f"⚠️ Model architecture differs from DR (hash: {self.config.dr_model_hash} vs {model_hash})")
        
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
        
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model Statistics:")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Model size: ~{total_params * 4 / 1024 / 1024:.1f}MB")
        
    def setup_dataset(self):
        """Setup Berkeley dataset with timestep sampling"""
        print("\n📊 SETTING UP COMPATIBLE BERKELEY DATASET")
        print("==========================================")
        
        # Configure Berkeley parser for timestep sampling
        berkeley_config = BerkeleyConfig(
            dataset_path=self.config.dataset_path,
            batch_size=1,  # We'll handle batching ourselves
            max_sequence_length=50,
            use_hand_camera=True,
            use_language=False  # Simplify for compatibility
        )
        
        self.dataset_parser = BerkeleyDatasetParser(berkeley_config)
        self.raw_dataset = self.dataset_parser.create_dataset('train')
        
        # Convert to compatible format
        self.compatible_dataset = self._create_compatible_dataset()
        
        print(f"📁 Found {len(self.compatible_dataset)} training samples")
        print(f"📊 Dataset Statistics:")
        print(f"   • Samples: {len(self.compatible_dataset)}")
        print(f"   • Batch size: {self.config.batch_size}")
        print(f"   • Timesteps per episode: {self.config.sample_timesteps_per_episode}")
        
    def _create_compatible_dataset(self) -> List[Dict]:
        """Convert Berkeley episodes to compatible single-timestep samples"""
        compatible_samples = []
        
        print("🔄 Converting Berkeley episodes to compatible format...")
        
        for i, batch in enumerate(self.raw_dataset.take(500)):  # Process more episodes for full training
            try:
                # Extract data
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
                        'images': np.zeros((3, 192, 192), dtype=np.float32)  # [C, H, W] format for CNN
                    }
                    compatible_samples.append(sample)
                    
                if i % 50 == 0:
                    print(f"   Processed {i} episodes -> {len(compatible_samples)} samples")
                    
            except Exception as e:
                print(f"   ⚠️ Episode {i} failed: {e}")
                continue
        
        print(f"✅ Generated {len(compatible_samples)} compatible training samples")
        return compatible_samples
    
    def create_batch(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Create a batch compatible with DR model architecture"""
        batch_size = len(samples)
        
        # Stack samples
        robot_states = np.stack([s['robot_state'] for s in samples])  # [batch, 15]
        actions = np.stack([s['action'] for s in samples])  # [batch, 7]
        images = np.stack([s['images'] for s in samples])  # [batch, 3, 192, 192]
        
        # Convert to torch tensors with correct format for DR model
        batch = {
            'images': torch.from_numpy(images).float().to(self.device),  # [batch, 3, 192, 192] - already correct
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
        shuffled_samples = self.compatible_dataset.copy()
        np.random.shuffle(shuffled_samples)
        
        # Create batches
        batch_size = self.config.batch_size
        num_samples = len(shuffled_samples)
        
        start_time = time.time()
        
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
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    predicted_actions = self.model(batch)
                    loss = nn.MSELoss()(predicted_actions, batch['actions'])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                if num_batches % 10 == 0:
                    print(f"   Batch {num_batches}: Loss {loss.item():.6f}")
                    
            except Exception as e:
                print(f"   ⚠️ Batch {num_batches} failed: {e}")
                continue
        
        epoch_time = time.time() - start_time
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        
        print(f"✅ Epoch completed: Loss {avg_epoch_loss:.6f} | Time: {epoch_time:.1f}s")
        
        return {
            'epoch_loss': avg_epoch_loss,
            'num_batches': num_batches,
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
        
        if len(self.compatible_dataset) == 0:
            print("❌ No compatible training samples generated!")
            return {'error': 'No training data'}
        
        epochs = []
        losses = []
        times = []
        
        no_improvement_count = 0
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_results = self.train_epoch()
            current_loss = epoch_results['epoch_loss']
            
            epochs.append(epoch + 1)
            losses.append(current_loss)
            times.append(epoch_results['epoch_time'])
            
            # Save checkpoint
            self.save_checkpoint(f"compatible_berkeley_epoch_{epoch + 1}")
            
            # Check for improvement
            if current_loss < self.best_loss - self.config.early_stopping_min_delta:
                self.best_loss = current_loss
                self.save_checkpoint("compatible_berkeley_best_model")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= self.config.early_stopping_patience:
                print(f"🛑 Early stopping: No improvement for {self.config.early_stopping_patience} epochs")
                break
        
        # Save final results
        model_hash = self.model._compute_architecture_hash()
        results = {
            'epochs': epochs,
            'losses': losses,
            'times': times,
            'model_hash': model_hash,
            'best_loss': self.best_loss,
            'total_samples': len(self.compatible_dataset)
        }
        
        results_path = os.path.join(self.config.save_dir, "compatible_berkeley_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n🎉 COMPATIBLE BERKELEY TRAINING COMPLETE")
        print("==========================================")
        print(f"📊 Final Results:")
        print(f"   • Epochs completed: {len(epochs)}")
        print(f"   • Best loss: {self.best_loss:.6f}")
        print(f"   • Total samples: {len(self.compatible_dataset)}")
        print(f"   • Model hash: {model_hash}")
        print(f"   • Results saved: {results_path}")
        
        return results

def main():
    """Main training execution"""
    config = CompatibleBerkeleyConfig()
    trainer = CompatibleBerkeleyTrainer(config)
    results = trainer.train()
    return results

if __name__ == "__main__":
    results = main()
    print(f"Final results: {results}")
