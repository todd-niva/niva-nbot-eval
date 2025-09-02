#!/usr/bin/env python3

"""
Inspect Robot USD - Find Available Robot Paths
==============================================

Script to inspect the robot USD file and find available robot articulation paths.
"""

from pxr import Usd

def main():
    """Inspect the robot USD file"""
    usd_path = "/ros2_ws/assets/assets/ur10e_robotiq2f-140/ur10e_robotiq2f-140-topic_based.usd"
    
    print(f"🔍 Inspecting USD file: {usd_path}")
    
    try:
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"❌ Could not open USD file: {usd_path}")
            return
        
        print("\n📋 Available prims in USD file:")
        robot_paths = []
        
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            type_name = prim.GetTypeName()
            
            # Print all prims for inspection
            if len(path.split('/')) <= 3:  # Only show top-level and first sub-level
                print(f"  {path} ({type_name})")
            
            # Look for robot-related paths
            if any(keyword in path.lower() for keyword in ['robot', 'ur10', 'ur_10', 'manipulator', 'arm']):
                robot_paths.append((path, type_name))
        
        print("\n🤖 Robot-related paths found:")
        for path, type_name in robot_paths:
            print(f"  {path} ({type_name})")
        
        # Look specifically for articulation roots
        print("\n🔗 Potential articulation paths:")
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            type_name = prim.GetTypeName()
            
            # Check if this could be an articulation root
            if hasattr(prim, 'GetAttribute'):
                # Check for physics articulation properties
                if prim.HasAPI('PhysicsArticulationRootAPI') or 'Articulation' in type_name:
                    print(f"  🎯 ARTICULATION: {path} ({type_name})")
        
        print(f"\n✅ USD inspection complete")
        
    except Exception as e:
        print(f"❌ Failed to inspect USD: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
