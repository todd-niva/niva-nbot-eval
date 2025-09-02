#!/usr/bin/env python3

"""
Test Statistical Framework - Quick Validation
============================================

Quick test to validate the statistical framework works before running 500 trials.
Runs 5 trials per level to test the infrastructure.
"""

import os
import sys

# Add the scripts directory to the Python path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

# Import the main statistical evaluator
from phase2_statistical_performance_test import StatisticalPerformanceEvaluator, log

def main():
    """Test the statistical framework with reduced trials"""
    log("🧪 TESTING STATISTICAL FRAMEWORK")
    log("📊 Running 5 trials per level for validation...")
    
    try:
        # Create evaluator with reduced trial count
        evaluator = StatisticalPerformanceEvaluator(trials_per_level=5)
        results = evaluator.run_comprehensive_evaluation()
        
        log("✅ Statistical framework test completed successfully")
        log("🚀 Ready to run full 100-trial evaluation")
        
        return results
        
    except Exception as e:
        log(f"❌ Statistical framework test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
