#!/usr/bin/env python3
"""
BERKELEY FOUNDATION TRAINING - REAL ROBOT DATA APPROACH
======================================================

Training framework using Berkeley UR5 dataset with identical architecture
to failed DR model for fair comparison and methodology validation.

Key Features:
- Identical model architecture (hash: 25424174) for fair comparison
- Berkeley real robot demonstrations (989 episodes, 68.5GB)
- Same optimizations: mixed precision, gradient accumulation, torch.compile
- Rigorous training monitoring and checkpoint management
- Direct comparison setup vs DR training failure

Expected Results: 3-10x improvement over DR training (0.8% baseline)
Target Performance: >5% success rate (vs 1.4% untrained baseline)

Author: NIVA Training Team
Date: 2025-01-02
Status: Berkeley Foundation Training - Critical Methodology Test
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import hashlib
from pathlib import Path

# Add our framework
sys.path.append('/home/todd/niva-nbot-eval/scripts')
from fair_comparison_training import FairComparisonConfig, StandardizedVisuoMotorPolicy
from berkeley_dataset_parser import BerkeleyDatasetParser, BerkeleyConfig

@dataclass
class BerkeleyTrainingConfig:
    """Configuration for Berkeley foundation training"""
    
    # Model Architecture (IDENTICAL to DR for fair comparison)
    vision_encoder_dim: int = 512
    state_encoder_dim: int = 256  
    language_encoder_dim: int = 256
    fusion_dim: int = 1024
    hidden_dim: int = 512
    action_dim: int = 8  # 7 joint actions + 1 gripper
    num_layers: int = 4
    dropout: float = 0.1
    
    # Berkeley Dataset Configuration
    dataset_path: str = "/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0"
    sequence_length: int = 50  # Truncate long episodes for consistency
    
    # Training Configuration (IDENTICAL to DR)
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 20  # More epochs for real data
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 1000
    
    # Optimization (IDENTICAL to DR)
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_torch_compile: bool = True
    max_grad_norm: float = 1.0
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    pin_memory: bool = True
    
    # Logging and Checkpoints
    save_dir: str = "/home/todd/niva-nbot-eval/berkeley_foundation_training"
    log_interval: int = 50
    checkpoint_interval: int = 500
    
    # Validation
    validate_every_n_epochs: int = 2
    early_stopping_patience: int = 5

class BerkeleyFoundationTrainer:
    """Berkeley real robot data foundation trainer"""
    
    def __init__(self, config: BerkeleyTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        print(f"🎯 BERKELEY FOUNDATION TRAINING")
        print(f"===============================")
        print(f"📂 Dataset: {config.dataset_path}")
        print(f"🤖 Model: Identical to DR (for fair comparison)")
        print(f"🚀 Device: {self.device}")
        print(f"💾 Save directory: {config.save_dir}")
    
    def compute_model_hash(self, model) -> int:
        """Compute model architecture hash for verification"""
        # Get model architecture signature
        total_params = sum(p.numel() for p in model.parameters())
        model_signature = str(model).encode('utf-8')
        hash_obj = hashlib.md5(model_signature)
        return int(hash_obj.hexdigest()[:8], 16)
        
    def setup_model_and_training(self):
        """Setup model, optimizer, and training components"""
        print(f"\n🏗️ SETTING UP TRAINING INFRASTRUCTURE")
        print("=" * 40)
        
        # Create identical model to DR training
        fair_config = FairComparisonConfig()
        self.model = StandardizedVisuoMotorPolicy(
            config=fair_config,
            approach_name="berkeley_foundation"
        ).to(self.device)
        
        # Verify identical architecture
        model_hash = self.compute_model_hash(self.model)
        expected_hash = 25424174  # DR model hash
        
        if model_hash == expected_hash:
            print(f"✅ Model architecture verified identical to DR (hash: {model_hash})")
        else:
            print(f"⚠️ Model architecture differs from DR (hash: {model_hash} vs {expected_hash})")
        
        # Apply optimizations (skip gradient checkpointing if not supported)
        if self.config.use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        else:
            print("⚠️ Gradient checkpointing not available - continuing without")
        
        if self.config.use_torch_compile:
            self.model = torch.compile(self.model)
            print("✅ Model compiled with torch.compile")
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup mixed precision
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
            print("✅ Mixed precision enabled")
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model Statistics:")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Model size: ~{total_params * 4 / 1024**2:.1f}MB")
        
    def setup_data_loaders(self):
        """Setup Berkeley dataset loaders"""
        print(f"\n📊 SETTING UP BERKELEY DATASET")
        print("=" * 35)
        
        # Berkeley dataset configuration
        berkeley_config = BerkeleyConfig(
            dataset_path=self.config.dataset_path,
            batch_size=self.config.batch_size,
            max_sequence_length=self.config.sequence_length,
            use_hand_camera=True,
            use_language=True
        )
        
        # Create dataset parser
        self.dataset_parser = BerkeleyDatasetParser(berkeley_config)
        
        # Create datasets
        self.train_dataset = self.dataset_parser.create_dataset('train')
        
        # Configure for training
        self.train_dataset = self.train_dataset.shuffle(1000)
        self.train_dataset = self.train_dataset.batch(self.config.batch_size)
        self.train_dataset = self.train_dataset.prefetch(4)
        
        # Count episodes for statistics
        train_files = self.dataset_parser._get_tfrecord_files('train')
        print(f"📊 Dataset Statistics:")
        print(f"   • Training files: {len(train_files)}")
        print(f"   • Estimated episodes: ~989")
        print(f"   • Batch size: {self.config.batch_size}")
        print(f"   • Sequence length: {self.config.sequence_length}")
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\n🚀 EPOCH {self.epoch + 1}/{self.config.num_epochs}")
        print("=" * 30)
        
        start_time = time.time()
        
        for batch_idx, tf_batch in enumerate(self.train_dataset):
            try:
                # Convert TensorFlow batch to PyTorch
                batch = self.tf_batch_to_torch(tf_batch)
                
                # Forward pass with gradient accumulation
                if batch_idx % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                
                if self.config.use_mixed_precision:
                    with autocast():
                        loss = self.compute_loss(batch)
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.global_step += 1
                else:
                    loss = self.compute_loss(batch)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        self.global_step += 1
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    avg_loss = epoch_loss / max(1, num_batches)
                    elapsed = time.time() - start_time
                    print(f"   Batch {batch_idx:4d} | Loss: {loss.item():.6f} | Avg: {avg_loss:.6f} | {elapsed:.1f}s")
                
                # Checkpointing
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                
            except Exception as e:
                print(f"   ⚠️ Batch {batch_idx} failed: {e}")
                continue
        
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        epoch_time = time.time() - start_time
        
        print(f"✅ Epoch completed: Loss {avg_epoch_loss:.6f} | Time: {epoch_time:.1f}s")
        
        return {
            'epoch_loss': avg_epoch_loss,
            'num_batches': num_batches,
            'epoch_time': epoch_time
        }
    
    def tf_batch_to_torch(self, tf_batch) -> Dict[str, torch.Tensor]:
        """Convert TensorFlow batch to PyTorch tensors"""
        batch = {}
        
        # Images
        if 'image' in tf_batch:
            images = tf_batch['image'].numpy()
            batch['images'] = torch.from_numpy(images).float().to(self.device)
        
        # Robot states
        if 'robot_state' in tf_batch:
            robot_states = tf_batch['robot_state'].numpy()
            batch['robot_states'] = torch.from_numpy(robot_states).float().to(self.device)
        
        # Actions (target)
        if 'action' in tf_batch:
            actions = tf_batch['action'].numpy()
            batch['actions'] = torch.from_numpy(actions).float().to(self.device)
        
        # Language embeddings
        if 'language_embedding' in tf_batch:
            language = tf_batch['language_embedding'].numpy()
            batch['language_embeddings'] = torch.from_numpy(language).float().to(self.device)
        
        return batch
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss"""
        # Forward pass using batch dictionary (same as DR training)
        predicted_actions = self.model(batch)
        
        target_actions = batch['actions']
        
        # MSE loss (same as DR training for fair comparison)
        loss = nn.MSELoss()(predicted_actions, target_actions)
        
        return loss
    
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
        
        checkpoint_path = os.path.join(self.config.save_dir, f"berkeley_{name}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Execute complete training"""
        print(f"\n🚀 STARTING BERKELEY FOUNDATION TRAINING")
        print("=" * 45)
        
        # Setup training infrastructure
        self.setup_model_and_training()
        self.setup_data_loaders()
        
        # Training metrics
        training_metrics = {
            'epochs': [],
            'losses': [],
            'times': [],
            'model_hash': self.compute_model_hash(self.model)
        }
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch()
            
            # Record metrics
            training_metrics['epochs'].append(epoch + 1)
            training_metrics['losses'].append(epoch_metrics['epoch_loss'])
            training_metrics['times'].append(epoch_metrics['epoch_time'])
            
            # Save epoch checkpoint
            if epoch % 2 == 0:  # Save every 2 epochs
                self.save_checkpoint(f"epoch_{epoch + 1}")
            
            # Early stopping check
            if epoch_metrics['epoch_loss'] < self.best_loss:
                self.best_loss = epoch_metrics['epoch_loss']
                self.patience_counter = 0
                self.save_checkpoint("best_model")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final results
        results_path = os.path.join(self.config.save_dir, "berkeley_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        print(f"\n🎉 BERKELEY FOUNDATION TRAINING COMPLETE")
        print("=" * 45)
        print(f"📊 Final Results:")
        print(f"   • Epochs completed: {len(training_metrics['epochs'])}")
        print(f"   • Best loss: {self.best_loss:.6f}")
        print(f"   • Model hash: {training_metrics['model_hash']}")
        print(f"   • Results saved: {results_path}")
        
        return training_metrics

def main():
    """Main execution for Berkeley foundation training"""
    print("🎯 BERKELEY FOUNDATION TRAINING - CRITICAL METHODOLOGY TEST")
    print("=" * 65)
    
    # Configuration
    config = BerkeleyTrainingConfig()
    
    # Check dataset availability
    if not os.path.exists(config.dataset_path):
        print(f"❌ Berkeley dataset not found: {config.dataset_path}")
        return
    
    # GPU validation
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ No GPU available - training will be slow")
    
    try:
        # Execute training
        trainer = BerkeleyFoundationTrainer(config)
        results = trainer.train()
        
        print(f"\n✅ Berkeley foundation training completed successfully!")
        print(f"🎯 Next step: Evaluate Berkeley model vs DR model vs baseline")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
