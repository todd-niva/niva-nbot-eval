#!/usr/bin/env python3

"""
Pilot Validation Study - Methodology Verification
================================================

Quick validation test with 20 trials per level to verify:
1. Realistic baseline success rates (0.5-5%)
2. Proper failure mode distributions  
3. Literature consistency before full 150-trial evaluation
4. Investor-ready experimental framework

This ensures our methodology is sound before committing to full evaluation.
"""

import os
import sys

# Add the scripts directory to the Python path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

# Import our realistic framework
from phase2_realistic_baseline_framework import RealisticStatisticalFramework

def main():
    """Run pilot validation with 20 trials per level"""
    
    print("🧪 PILOT VALIDATION STUDY")
    print("📊 Methodology verification before full evaluation")
    print("🎯 20 trials per complexity level (100 total trials)")
    
    # Run pilot evaluation
    pilot_evaluator = RealisticStatisticalFramework(trials_per_level=20, random_seed=123)
    results = pilot_evaluator.run_realistic_baseline_evaluation()
    
    # Print validation summary
    print("\n📈 PILOT VALIDATION SUMMARY")
    print("="*60)
    
    statistical_summary = results["statistical_summary"]
    
    for level_key in sorted(statistical_summary.keys()):
        level_data = statistical_summary[level_key]
        level_num = int(level_key.split('_')[1])
        
        success_rate = level_data["success_rate"] 
        expected_rate = level_data["expected_rate"]
        
        print(f"Level {level_num}: {success_rate:.1%} success (expected: {expected_rate:.1%})")
        
        if level_data["primary_failure_mode"]:
            print(f"  Primary failure: {level_data['primary_failure_mode']}")
    
    print("\n✅ Pilot validation complete")
    print("📁 Results saved for methodology review")

if __name__ == "__main__":
    main()
