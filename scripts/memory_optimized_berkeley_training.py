#!/usr/bin/env python3
"""
MEMORY-OPTIMIZED BERKELEY TRAINING - 77GB DATASET WITH 16GB VRAM
===============================================================

GPU memory-optimized training that can handle the full 77GB Berkeley dataset
with only 16GB VRAM through intelligent streaming and memory management.

Key Optimizations:
- Streaming data pipeline (DRAM → GPU batch-by-batch)
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16) for 50% memory reduction
- CPU pinned memory with async GPU transfers
- Immediate GPU memory cleanup after each batch
- Memory-efficient model architecture

This enables training on massive datasets with limited GPU memory.

Author: NIVA Training Team
Date: 2025-09-02
Status: Production Memory-Optimized Training
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
import gc
import psutil

# Import our Berkeley dataset parser
sys.path.append('/home/todd/niva-nbot-eval/scripts')
from berkeley_dataset_parser import BerkeleyConfig, BerkeleyPyTorchDataset

@dataclass
class MemoryOptimizedConfig:
    """Memory-optimized training configuration"""
    
    # Dataset Configuration
    dataset_path: str = "/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0"
    use_all_files: bool = True
    max_files_train: Optional[int] = None  # Use all files
    max_files_val: Optional[int] = 20
    
    # Memory Optimization - KEY SETTINGS
    batch_size: int = 4           # Small batches that fit in VRAM
    effective_batch_size: int = 32  # Large effective batch via accumulation
    gradient_accumulation_steps: int = 8  # 32/4 = 8 accumulation steps
    max_episodes_in_memory: int = 100  # Limit episode preloading
    
    # Training Configuration
    num_epochs: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    
    # Model Configuration - MEMORY EFFICIENT
    image_size: Tuple[int, int] = (192, 192)  # Smaller images = less VRAM
    max_sequence_length: int = 20             # Shorter sequences = less VRAM
    hidden_dim: int = 256                     # Smaller model = less VRAM
    num_attention_heads: int = 4              # Fewer heads = less VRAM
    num_transformer_layers: int = 2           # Fewer layers = less VRAM
    
    # Memory Optimization Settings
    mixed_precision: bool = True        # FP16 = 50% memory reduction
    pin_memory: bool = True            # Faster CPU→GPU transfer
    non_blocking_transfer: bool = True # Async GPU transfer
    immediate_cleanup: bool = True     # Delete batches immediately
    gradient_checkpointing: bool = True # Trade compute for memory
    
    # Data Loading Optimization
    num_workers: int = 4               # Parallel data loading
    prefetch_factor: int = 2           # CPU prefetching
    
    # Monitoring
    log_frequency: int = 20
    memory_log_frequency: int = 10     # Log GPU memory usage
    
    # Output
    output_dir: str = "/home/todd/niva-nbot-eval/memory_optimized_training"
    model_save_dir: str = "/home/todd/niva-nbot-eval/models"

class MemoryEfficientVisuoMotorPolicy(nn.Module):
    """Memory-efficient visuomotor policy for large dataset training"""
    
    def __init__(self, config: MemoryOptimizedConfig):
        super().__init__()
        self.config = config
        
        # Efficient vision encoder - fewer parameters
        self.vision_encoder = nn.Sequential(
            # Efficient conv layers
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # 32 x 96 x 96
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 x 48 x 48
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 64 x 24 x 24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 12 x 12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 x 6 x 6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((3, 3)),  # 256 x 3 x 3
            nn.Flatten(),  # 2304 features
            
            nn.Linear(2304, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Linear(64, config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Lightweight temporal modeling
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Compact transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 2,  # Smaller feedforward
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, 7)  # 7D action
        )
        
        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            # Apply gradient checkpointing to vision encoder
            self.vision_encoder.requires_grad_(True)
            # We'll apply checkpointing in the forward pass
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🧠 Memory-Efficient Policy:")
        print(f"   Hidden dim: {config.hidden_dim}")
        print(f"   Attention heads: {config.num_attention_heads}")
        print(f"   Transformer layers: {config.num_transformer_layers}")
        print(f"   Gradient checkpointing: {config.gradient_checkpointing}")
        print(f"   Total parameters: {total_params:,}")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Handle single episode dimensions
        images = batch['images']  # [seq, H, W, 3]
        if images.dim() == 4:
            seq_len = images.shape[0]
            images = images.permute(0, 3, 1, 2)  # [seq, 3, H, W]
        else:
            raise ValueError(f"Expected 4D image tensor, got {images.shape}")
        
        # Process images through vision encoder (with optional checkpointing)
        if self.config.gradient_checkpointing and self.training:
            vision_features = torch.utils.checkpoint.checkpoint(
                self.vision_encoder, images, use_reentrant=False
            )
        else:
            vision_features = self.vision_encoder(images)  # [seq, hidden_dim]
        
        # Process robot states
        states = batch['robot_states']  # [seq, 15]
        state_features = self.state_encoder(states)  # [seq, hidden_dim]
        
        # Combine features
        combined_features = vision_features + state_features  # [seq, hidden_dim]
        combined_features = combined_features.unsqueeze(0)  # [1, seq, hidden_dim]
        
        # Apply attention and transformer
        attended, _ = self.temporal_attention(
            combined_features, combined_features, combined_features
        )
        
        temporal_features = self.transformer(attended)  # [1, seq, hidden_dim]
        
        # Use final timestep for action prediction
        final_features = temporal_features[0, -1, :]  # [hidden_dim]
        
        # Predict action
        action = self.action_head(final_features)  # [7]
        
        return action

class MemoryOptimizedTrainer:
    """Memory-optimized trainer for large-scale Berkeley dataset training"""
    
    def __init__(self, config: MemoryOptimizedConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.model_save_dir, exist_ok=True)
        
        print(f"🚀 MEMORY-OPTIMIZED BERKELEY TRAINING")
        print(f"====================================")
        print(f"🔥 Device: {self.device}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"💾 System RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
        print(f"📊 Strategy: Stream 77GB dataset through {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB VRAM")
        print(f"📊 Batch size: {config.batch_size} (effective: {config.effective_batch_size})")
        print(f"📊 Accumulation steps: {config.gradient_accumulation_steps}")
        
        # Initialize memory-efficient model
        self.model = MemoryEfficientVisuoMotorPolicy(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.global_step = 0
        self.memory_stats = []
        
        print(f"✅ Memory-optimized trainer ready")
    
    def log_memory_usage(self, step_name: str):
        """Log detailed memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            # System memory
            ram = psutil.virtual_memory()
            ram_used = ram.used / 1024**3
            ram_percent = ram.percent
            
            memory_info = {
                'step': step_name,
                'gpu_allocated': allocated,
                'gpu_reserved': reserved,
                'gpu_max_allocated': max_allocated,
                'ram_used': ram_used,
                'ram_percent': ram_percent
            }
            
            self.memory_stats.append(memory_info)
            
            print(f"💾 {step_name}: GPU {allocated:.1f}GB/{reserved:.1f}GB | RAM {ram_used:.1f}GB ({ram_percent:.0f}%)")
    
    def setup_data_loaders(self):
        """Setup memory-optimized data loaders"""
        print("🔄 Setting up memory-optimized data loaders...")
        
        # Training dataset with memory limits
        train_config = BerkeleyConfig(
            dataset_path=self.config.dataset_path,
            batch_size=self.config.batch_size,
            max_sequence_length=self.config.max_sequence_length,
            image_size=self.config.image_size,
            use_hand_camera=False,  # Disable to save memory
            use_language=False,     # Disable to save memory
            max_files_per_split=self.config.max_files_train,
            shuffle_buffer_size=500  # Smaller shuffle buffer
        )
        
        # Validation dataset
        val_config = BerkeleyConfig(
            dataset_path=self.config.dataset_path,
            batch_size=self.config.batch_size,
            max_sequence_length=self.config.max_sequence_length,
            image_size=self.config.image_size,
            use_hand_camera=False,
            use_language=False,
            max_files_per_split=self.config.max_files_val,
            shuffle_buffer_size=0
        )
        
        # Create datasets with limited memory usage
        print("📚 Loading training dataset (memory-limited)...")
        self.train_dataset = BerkeleyPyTorchDataset(train_config, split='train')
        
        print("📚 Loading validation dataset...")
        self.val_dataset = BerkeleyPyTorchDataset(val_config, split='test')
        
        # Create memory-optimized data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # Always 1 episode at a time
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,  # Fewer workers for validation
            pin_memory=self.config.pin_memory
        )
        
        print(f"✅ Memory-optimized data loaders ready:")
        print(f"   Training episodes: {len(self.train_dataset)}")
        print(f"   Validation episodes: {len(self.val_dataset)}")
        
        self.log_memory_usage("After data loader setup")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Memory-optimized training epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_count = 0
        epoch_start = time.time()
        
        print(f"\n🔥 Epoch {epoch}/{self.config.num_epochs} - Memory-Optimized Training")
        print("-" * 70)
        
        # Zero gradients at start of accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # CRITICAL: Move data to GPU with non-blocking transfer
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].squeeze(0).to(
                        self.device, 
                        non_blocking=self.config.non_blocking_transfer
                    )
            
            # Skip invalid batches
            if 'actions' not in batch or batch['actions'].numel() == 0:
                continue
            
            # Get target action (final timestep)
            target_action = batch['actions'][-1]  # [7]
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    predicted_action = self.model(batch)
                    loss = self.criterion(predicted_action, target_action)
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                predicted_action = self.model(batch)
                loss = self.criterion(predicted_action, target_action)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # CRITICAL: Immediate cleanup to free GPU memory
            if self.config.immediate_cleanup:
                del batch, predicted_action
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
            accumulation_count += 1
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Optimizer step after accumulation
            if accumulation_count >= self.config.gradient_accumulation_steps:
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulation_count = 0
                num_batches += 1
                self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  [{epoch:2d}][{batch_idx:4d}] Loss: {loss.item() * self.config.gradient_accumulation_steps:.6f} | LR: {current_lr:.2e}")
            
            # Memory logging
            if batch_idx % self.config.memory_log_frequency == 0:
                self.log_memory_usage(f"Batch {batch_idx}")
        
        # Handle remaining gradients
        if accumulation_count > 0:
            if self.config.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start
        
        # Force garbage collection
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return {
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'num_batches': num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """Memory-optimized validation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to GPU
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].squeeze(0).to(self.device, non_blocking=True)
                
                if 'actions' not in batch or batch['actions'].numel() == 0:
                    continue
                
                target_action = batch['actions'][-1]
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast():
                        predicted_action = self.model(batch)
                        loss = self.criterion(predicted_action, target_action)
                else:
                    predicted_action = self.model(batch)
                    loss = self.criterion(predicted_action, target_action)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Immediate cleanup
                del batch, predicted_action
        
        # Cleanup
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'avg_loss': avg_loss}
    
    def train(self):
        """Execute memory-optimized training"""
        print(f"\n🚀 STARTING MEMORY-OPTIMIZED TRAINING")
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Training loop
        for epoch in range(1, self.config.num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Report
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"   Train Loss: {train_metrics['avg_loss']:.6f}")
            print(f"   Val Loss: {val_metrics['avg_loss']:.6f}")
            print(f"   Epoch Time: {train_metrics['epoch_time']:.1f}s")
            print(f"   Batches: {train_metrics['num_batches']}")
            
            # Learning rate step
            self.scheduler.step()
            
            # Memory summary
            self.log_memory_usage(f"End of epoch {epoch}")
        
        # Save final model
        model_path = os.path.join(self.config.model_save_dir, "berkeley_memory_optimized_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'memory_stats': self.memory_stats
        }, model_path)
        
        print(f"\n✅ MEMORY-OPTIMIZED TRAINING COMPLETED!")
        print(f"💾 Model saved: {model_path}")
        
        # Final memory report
        print(f"\n📊 MEMORY USAGE SUMMARY:")
        if self.memory_stats:
            max_gpu = max(stat['gpu_allocated'] for stat in self.memory_stats)
            max_ram = max(stat['ram_used'] for stat in self.memory_stats)
            print(f"   Peak GPU usage: {max_gpu:.1f}GB")
            print(f"   Peak RAM usage: {max_ram:.1f}GB")
            print(f"   Successfully trained on 77GB dataset with {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB VRAM!")

def main():
    """Main execution"""
    print("🚀 MEMORY-OPTIMIZED BERKELEY TRAINING")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    # Memory-optimized configuration
    config = MemoryOptimizedConfig(
        # Core settings
        batch_size=4,
        effective_batch_size=16,
        gradient_accumulation_steps=4,
        
        # Memory limits
        max_episodes_in_memory=50,
        
        # Model efficiency
        image_size=(192, 192),
        max_sequence_length=15,
        hidden_dim=256,
        num_attention_heads=4,
        num_transformer_layers=2,
        
        # Training
        num_epochs=8,
        learning_rate=2e-4,
        
        # Memory optimizations
        mixed_precision=True,
        pin_memory=True,
        non_blocking_transfer=True,
        immediate_cleanup=True,
        gradient_checkpointing=True,
        
        # Monitoring
        log_frequency=25,
        memory_log_frequency=50
    )
    
    trainer = MemoryOptimizedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
