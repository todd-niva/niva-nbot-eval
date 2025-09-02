#!/usr/bin/env python3

from pxr import Usd, UsdGeom, UsdLux

# Load the scene
stage = Usd.Stage.Open('/ros2_ws/assets/evaluation_scene.usd')

print('=== CAMERA ANALYSIS ===')
cameras = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera):
        cameras.append(str(prim.GetPath()))
        print(f'Camera found: {prim.GetPath()}')

print(f'Total cameras: {len(cameras)}')

print('')
print('=== LIGHT ANALYSIS ===')
lights = []
for prim in stage.Traverse():
    if (prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.DomeLight) or 
        prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DistantLight)):
        lights.append(str(prim.GetPath()))
        print(f'Light found: {prim.GetPath()} ({prim.GetTypeName()})')

print(f'Total lights: {len(lights)}')

print('')
print('=== ROBOT ANALYSIS ===')
robots = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if 'Robot' in path or 'UR' in path:
        robots.append(path)

print(f'Robot parts found: {len(robots)}')
if len(robots) > 0:
    print('First few robot parts:')
    for robot in robots[:3]:
        print(f'  {robot}')

print('')
print('=== SUMMARY ===')
print(f'Cameras: {len(cameras)}')
print(f'Lights: {len(lights)}')
print(f'Robot parts: {len(robots)}')
