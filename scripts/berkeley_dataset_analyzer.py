#!/usr/bin/env python3

"""
Berkeley Dataset Analyzer: Decode Real TFRecord Format
======================================================

Analyzes the actual Berkeley dataset structure to properly extract:
- Robot states (15D)
- Actions (7D) 
- Episode sequences

This ensures we use REAL robot data for DR training, not synthetic patterns.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

def log(message: str):
    """Enhanced logging with timestamp"""
    import time
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def analyze_tfrecord_structure(tfrecord_path: str, max_records: int = 3):
    """Analyze the structure of Berkeley TFRecord files"""
    log(f"🔍 Analyzing TFRecord structure: {tfrecord_path}")
    
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    
    for i, raw_record in enumerate(dataset.take(max_records)):
        log(f"\n📄 Record {i+1}:")
        
        # Try different parsing approaches
        try:
            # Method 1: Direct Example parsing
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            log(f"  🔧 Features found: {list(example.features.feature.keys())}")
            
            for feature_name, feature in example.features.feature.items():
                if feature.HasField('bytes_list'):
                    log(f"    📦 {feature_name}: bytes_list ({len(feature.bytes_list.value)} items)")
                elif feature.HasField('float_list'):
                    log(f"    📊 {feature_name}: float_list ({len(feature.float_list.value)} items)")
                elif feature.HasField('int64_list'):
                    log(f"    🔢 {feature_name}: int64_list ({len(feature.int64_list.value)} items)")
            
            # Try to extract steps if present
            if 'steps' in example.features.feature:
                steps_feature = example.features.feature['steps']
                if steps_feature.HasField('bytes_list'):
                    log(f"    🔄 Steps: {len(steps_feature.bytes_list.value)} episodes")
                    
                    # Try to parse first step
                    if len(steps_feature.bytes_list.value) > 0:
                        first_step = steps_feature.bytes_list.value[0]
                        step_example = tf.train.Example()
                        step_example.ParseFromString(first_step)
                        log(f"    📋 Step features: {list(step_example.features.feature.keys())}")
                        
                        # Look for observations and actions
                        for step_feature_name in step_example.features.feature.keys():
                            step_feature = step_example.features.feature[step_feature_name]
                            if step_feature.HasField('bytes_list'):
                                log(f"      🔍 {step_feature_name}: bytes_list ({len(step_feature.bytes_list.value)} items)")
                                
                                # Try to parse nested features
                                if len(step_feature.bytes_list.value) > 0:
                                    try:
                                        nested_example = tf.train.Example()
                                        nested_example.ParseFromString(step_feature.bytes_list.value[0])
                                        if nested_example.features.feature:
                                            log(f"        📦 Nested in {step_feature_name}: {list(nested_example.features.feature.keys())}")
                                    except:
                                        pass
                            elif step_feature.HasField('float_list'):
                                log(f"      📊 {step_feature_name}: float_list ({len(step_feature.float_list.value)} values)")
                
        except Exception as e:
            log(f"  ❌ Parsing error: {e}")

def extract_berkeley_episode_data(tfrecord_path: str):
    """Extract actual robot states and actions from Berkeley dataset"""
    log(f"🎯 Extracting episode data from: {tfrecord_path}")
    
    # Use TensorFlow Datasets approach
    try:
        # Create a simple feature description based on the format
        feature_description = {
            'steps': tf.io.VarLenFeature(tf.string)
        }
        
        def parse_tfrecord(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)
        
        dataset = tf.data.TFRecordDataset([tfrecord_path])
        parsed_dataset = dataset.map(parse_tfrecord)
        
        episodes_data = []
        
        for i, parsed_record in enumerate(parsed_dataset.take(2)):  # First 2 episodes
            log(f"\n🔄 Processing episode {i+1}")
            
            steps = parsed_record['steps'].values
            log(f"  📊 Episode has {len(steps)} steps")
            
            robot_states = []
            actions = []
            
            for j, step_bytes in enumerate(steps):
                try:
                    # Parse each step
                    step_example = tf.train.Example()
                    step_example.ParseFromString(step_bytes.numpy())
                    
                    step_features = step_example.features.feature
                    
                    # Extract observation data
                    if 'observation' in step_features:
                        obs_bytes = step_features['observation'].bytes_list.value[0]
                        obs_example = tf.train.Example()
                        obs_example.ParseFromString(obs_bytes)
                        
                        obs_features = obs_example.features.feature
                        
                        # Extract robot state
                        if 'robot_state' in obs_features:
                            robot_state = list(obs_features['robot_state'].float_list.value)
                            robot_states.append(robot_state)
                            if j == 0:
                                log(f"    🤖 Robot state shape: {len(robot_state)}D")
                    
                    # Extract action data
                    if 'action' in step_features:
                        action_bytes = step_features['action'].bytes_list.value[0]
                        action_example = tf.train.Example()
                        action_example.ParseFromString(action_bytes)
                        
                        action_features = action_example.features.feature
                        
                        # Reconstruct 7D action: [x, y, z, rx, ry, rz, gripper]
                        action = [0.0] * 7  # Initialize
                        
                        if 'world_vector' in action_features:
                            world_vec = list(action_features['world_vector'].float_list.value)
                            action[:3] = world_vec  # x, y, z
                        
                        if 'rotation_delta' in action_features:
                            rot_delta = list(action_features['rotation_delta'].float_list.value)
                            action[3:6] = rot_delta  # rx, ry, rz
                        
                        if 'gripper_closedness_action' in action_features:
                            gripper = action_features['gripper_closedness_action'].float_list.value[0]
                            action[6] = gripper  # gripper
                        
                        actions.append(action)
                        if j == 0:
                            log(f"    🎮 Action shape: {len(action)}D")
                
                except Exception as e:
                    if j < 3:  # Only log first few errors
                        log(f"    ⚠️ Step {j} parse error: {e}")
            
            if robot_states and actions:
                episode_data = {
                    'robot_states': np.array(robot_states, dtype=np.float32),
                    'actions': np.array(actions, dtype=np.float32),
                    'episode_length': len(robot_states)
                }
                
                episodes_data.append(episode_data)
                log(f"  ✅ Episode {i+1}: {len(robot_states)} timesteps extracted")
                log(f"    📊 Robot states shape: {episode_data['robot_states'].shape}")
                log(f"    🎮 Actions shape: {episode_data['actions'].shape}")
        
        return episodes_data
    
    except Exception as e:
        log(f"❌ Episode extraction failed: {e}")
        return []

def main():
    """Analyze Berkeley dataset structure"""
    log("🔍 BERKELEY DATASET ANALYZER")
    log("==========================")
    log("🎯 Goal: Understand real TFRecord format for DR training")
    
    dataset_path = Path("/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0")
    
    # Find a training file to analyze
    train_files = list(dataset_path.glob("berkeley_autolab_ur5-train.tfrecord-*"))
    
    if not train_files:
        log("❌ No training files found")
        return
    
    # Analyze first file structure
    first_file = train_files[0]
    log(f"\n📁 Analyzing file: {first_file.name}")
    analyze_tfrecord_structure(str(first_file))
    
    # Try to extract actual data
    log(f"\n🎯 EXTRACTING REAL EPISODE DATA")
    log("=" * 40)
    episodes = extract_berkeley_episode_data(str(first_file))
    
    if episodes:
        log(f"\n✅ EXTRACTION SUCCESSFUL!")
        log(f"📊 Extracted {len(episodes)} episodes")
        
        # Show sample data
        if len(episodes) > 0:
            ep = episodes[0]
            log(f"\n📋 Sample Episode Statistics:")
            log(f"   Robot states: {ep['robot_states'].shape}")
            log(f"   Actions: {ep['actions'].shape}")
            log(f"   Episode length: {ep['episode_length']}")
            
            log(f"\n🤖 Sample robot state (first timestep):")
            log(f"   {ep['robot_states'][0]}")
            
            log(f"\n🎮 Sample action (first timestep):")
            log(f"   {ep['actions'][0]}")
        
        log(f"\n🎯 READY TO IMPLEMENT REAL BERKELEY DR TRAINING!")
    else:
        log(f"\n❌ No episodes extracted - need to debug parser")

if __name__ == "__main__":
    main()
