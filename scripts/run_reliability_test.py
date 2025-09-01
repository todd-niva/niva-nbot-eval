#!/usr/bin/env python3
"""
Script to run the physics-based pick-grab-lift test 10 times and report success rate.
"""

import subprocess
import sys
import time
import os

def run_single_test():
    """Run one instance of the physics test and return success/failure."""
    try:
        cmd = [
            "timeout", "300s", "sg", "docker", "-c",
            'docker exec isaac_stack-isaacsim-1 bash -lc "export RCUTILS_COLORIZED_OUTPUT=0; export OMNI_KIT_SEARCH_PATHS_USER_DISABLED=1; export OMNI_KIT_DISABLE_USER_SITE=1; /isaac-sim/python.sh /ros2_ws/scripts/final_physics_test.py"'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if test passed
        if result.returncode == 0 and "TEST_PASS" in result.stdout:
            return True, result.stdout
        else:
            return False, f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            
    except Exception as e:
        return False, f"Exception running test: {e}"

def main():
    """Run reliability test 10 times."""
    print("=== Physics-Based Pick-Grab-Lift Reliability Test ===")
    print("Running test 10 times to verify 100% success rate...")
    print()
    
    successes = 0
    failures = 0
    results = []
    
    for i in range(1, 11):
        print(f"Test {i}/10: ", end="", flush=True)
        
        success, output = run_single_test()
        
        if success:
            print("PASS")
            successes += 1
            results.append(f"Test {i}: PASS")
        else:
            print("FAIL")
            failures += 1
            results.append(f"Test {i}: FAIL")
            # Print failure details for debugging
            print(f"  Failure details:\n{output}")
        
        # Brief pause between tests to let Isaac Sim fully shut down
        if i < 10:
            time.sleep(2)
    
    print()
    print("=== RESULTS ===")
    for result in results:
        print(result)
    
    print(f"\nSUCCESS RATE: {successes}/10 ({100 * successes / 10:.1f}%)")
    
    if successes == 10:
        print("✅ All tests passed! The physics-based pick-grab-lift is 100% reliable.")
        return 0
    else:
        print(f"❌ {failures} test(s) failed. Reliability needs improvement.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
