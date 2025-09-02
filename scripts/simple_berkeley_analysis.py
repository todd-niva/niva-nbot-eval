#!/usr/bin/env python3
"""
SIMPLE BERKELEY DATASET ANALYSIS
================================

Quick analysis of Berkeley dataset quality using the existing infrastructure.
This will give us key insights without complex parsing.

Author: NIVA Training Team
Date: 2025-01-02
Status: Quick Berkeley Dataset Assessment
"""

import os
import sys
import numpy as np
import tensorflow as tf
import json
from typing import Dict, List, Any
from collections import defaultdict

# Add our dataset parser
sys.path.append('/home/todd/niva-nbot-eval/scripts')

def analyze_berkeley_dataset():
    """Simple analysis of Berkeley dataset"""
    
    print("🔍 SIMPLE BERKELEY DATASET ANALYSIS")
    print("=" * 50)
    
    dataset_path = "/mnt/niva_hot/datasets/berkeley_autolab_ur5/0.1.0"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    # Count files
    pattern = "berkeley_autolab_ur5-train.tfrecord-"
    train_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if pattern in file:
                train_files.append(os.path.join(root, file))
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   • Training files found: {len(train_files)}")
    print(f"   • Dataset path: {dataset_path}")
    
    if len(train_files) == 0:
        print("❌ No training files found!")
        return None
    
    # Analyze a few files for structure
    print(f"\n🔍 ANALYZING FILE STRUCTURE:")
    
    sample_files = train_files[:5]  # Analyze first 5 files
    total_episodes = 0
    
    for i, file_path in enumerate(sample_files):
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Count records in file
            record_count = 0
            for _ in tf.data.TFRecordDataset(file_path):
                record_count += 1
            
            total_episodes += record_count
            
            print(f"   File {i+1}: {file_size_mb:.1f}MB, {record_count} episodes")
            
        except Exception as e:
            print(f"   File {i+1}: Error - {e}")
    
    print(f"\n📈 ESTIMATED SCALE:")
    avg_episodes_per_file = total_episodes / len(sample_files) if sample_files else 0
    estimated_total_episodes = avg_episodes_per_file * len(train_files)
    
    print(f"   • Average episodes per file: {avg_episodes_per_file:.1f}")
    print(f"   • Estimated total episodes: {estimated_total_episodes:.0f}")
    print(f"   • Total training files: {len(train_files)}")
    
    # Try to parse one episode to understand structure
    print(f"\n🔬 EPISODE STRUCTURE ANALYSIS:")
    
    try:
        # Take first file and first record
        first_file = train_files[0]
        dataset = tf.data.TFRecordDataset(first_file)
        
        for raw_record in dataset.take(1):
            # Parse the record
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            features = example.features.feature
            
            print(f"   • Features found in episode:")
            for key, value in features.items():
                if value.HasField('bytes_list'):
                    feature_type = "bytes"
                    count = len(value.bytes_list.value)
                elif value.HasField('float_list'):
                    feature_type = "float"
                    count = len(value.float_list.value)
                elif value.HasField('int64_list'):
                    feature_type = "int64" 
                    count = len(value.int64_list.value)
                else:
                    feature_type = "unknown"
                    count = 0
                
                print(f"     - {key}: {feature_type} ({count} values)")
    
    except Exception as e:
        print(f"   ⚠️ Structure analysis failed: {e}")
    
    # Calculate dataset statistics
    total_size_gb = sum(os.path.getsize(f) for f in train_files) / (1024**3)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   • Total size: {total_size_gb:.1f} GB")
    print(f"   • Training files: {len(train_files)}")
    print(f"   • Estimated episodes: {estimated_total_episodes:.0f}")
    print(f"   • Episodes per GB: {estimated_total_episodes / total_size_gb:.0f}")
    
    # Compare to our DR training
    print(f"\n💡 COMPARISON TO DR TRAINING:")
    print(f"   • DR training episodes: 7,500")
    print(f"   • Berkeley episodes (est): {estimated_total_episodes:.0f}")
    print(f"   • Scale advantage: {estimated_total_episodes / 7500:.1f}x more data")
    print(f"   • Data type: Real robot demonstrations vs procedural")
    
    # Generate recommendation
    print(f"\n🎯 INITIAL ASSESSMENT:")
    
    if estimated_total_episodes > 1000 and total_size_gb > 10:
        recommendation = "HIGHLY_PROMISING"
        reasons = [
            f"Large scale: {estimated_total_episodes:.0f} episodes",
            f"Substantial data: {total_size_gb:.1f}GB",
            "Real robot demonstrations (vs synthetic)",
            "UR5 robot compatibility with our system"
        ]
    elif estimated_total_episodes > 100:
        recommendation = "PROMISING"
        reasons = [
            f"Moderate scale: {estimated_total_episodes:.0f} episodes",
            "Real robot data should outperform synthetic",
            "Worth detailed quality analysis"
        ]
    else:
        recommendation = "REQUIRES_INVESTIGATION"
        reasons = [
            f"Small scale: {estimated_total_episodes:.0f} episodes",
            "May need additional datasets from 11.5TB collection"
        ]
    
    print(f"   • Recommendation: {recommendation}")
    for reason in reasons:
        print(f"     - {reason}")
    
    results = {
        'dataset_path': dataset_path,
        'total_files': len(train_files),
        'total_size_gb': total_size_gb,
        'estimated_episodes': estimated_total_episodes,
        'scale_vs_dr': estimated_total_episodes / 7500,
        'recommendation': recommendation,
        'reasons': reasons
    }
    
    return results

def compare_berkeley_to_dr_failure():
    """Compare Berkeley characteristics to our DR training failure"""
    
    print(f"\n🔍 BERKELEY vs DR TRAINING COMPARISON")
    print("=" * 50)
    
    comparison = {
        'data_source': {
            'berkeley': 'Real robot demonstrations with human expertise',
            'dr_training': 'Procedural synthetic episodes'
        },
        'physics_realism': {
            'berkeley': 'Real-world physics constraints and dynamics',
            'dr_training': 'Simplified synthetic physics simulation'
        },
        'action_patterns': {
            'berkeley': 'Authentic robot movements from successful tasks',
            'dr_training': 'Generated patterns optimized for loss, not task success'
        },
        'task_diversity': {
            'berkeley': 'Natural language instructions with varied scenarios',
            'dr_training': '5 complexity levels with artificial variations'
        },
        'success_validation': {
            'berkeley': 'Demonstrated successful task completion',
            'dr_training': 'Assumed success based on procedural generation'
        }
    }
    
    print("🔄 KEY DIFFERENCES:")
    for category, details in comparison.items():
        print(f"\n   {category.upper()}:")
        print(f"     Berkeley: {details['berkeley']}")
        print(f"     DR Training: {details['dr_training']}")
    
    print(f"\n💡 WHY BERKELEY SHOULD PERFORM BETTER:")
    advantages = [
        "Real robot control patterns vs synthetic approximations",
        "Proven successful task completion vs assumed success",
        "Human demonstration quality vs algorithmic generation",
        "Natural physics constraints vs simplified simulation",
        "Language-grounded tasks vs abstract complexity levels"
    ]
    
    for advantage in advantages:
        print(f"   • {advantage}")
    
    return comparison

def main():
    """Main execution"""
    # Analyze Berkeley dataset
    berkeley_results = analyze_berkeley_dataset()
    
    if berkeley_results:
        # Compare to DR training
        comparison = compare_berkeley_to_dr_failure()
        
        # Save results
        output_path = "/home/todd/niva-nbot-eval/analysis_results/simple_berkeley_analysis.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        full_results = {
            'berkeley_analysis': berkeley_results,
            'berkeley_vs_dr_comparison': comparison,
            'timestamp': 'January 2, 2025',
            'next_steps': [
                'Train model on Berkeley dataset using identical architecture to DR',
                'Evaluate Berkeley-trained model using same framework',
                'Compare: Berkeley vs DR vs Baseline performance',
                'If Berkeley succeeds, expand to full 11.5TB dataset collection'
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\n📊 Analysis saved to: {output_path}")
        
        print(f"\n🚀 RECOMMENDED NEXT STEPS:")
        for step in full_results['next_steps']:
            print(f"   1. {step}")

if __name__ == "__main__":
    main()

