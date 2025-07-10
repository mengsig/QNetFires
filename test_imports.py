#!/usr/bin/env python3
"""
Simple test script to debug import issues.
"""

import os
import sys

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '.'))
src_path = os.path.join(project_root, 'src')
scripts_path = os.path.join(src_path, 'scripts')

print(f"🔍 Debugging Import Paths:")
print(f"   - Project root: {project_root}")
print(f"   - Src path: {src_path}")
print(f"   - Scripts path: {scripts_path}")

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)
sys.path.insert(0, scripts_path)

print(f"\n📁 Checking file existence:")
required_files = [
    'src/scripts/Simulate.py',
    'src/scripts/FireEnv.py',
    'src/scripts/DQNAgent.py',
    'src/scripts/DomiRankMemoryLoader.py'
]

all_exist = True
for file_path in required_files:
    exists = os.path.exists(file_path)
    print(f"   - {file_path}: {'✅' if exists else '❌'}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Some required files are missing!")
    sys.exit(1)

print(f"\n📦 Testing imports:")

try:
    print("   - Testing Simulate import...")
    from src.scripts.Simulate import Simulate
    print("   ✅ Simulate imported successfully")
except Exception as e:
    print(f"   ❌ Simulate import failed: {e}")

try:
    print("   - Testing FireEnv import...")
    from src.scripts.FireEnv import FireEnv
    print("   ✅ FireEnv imported successfully")
except Exception as e:
    print(f"   ❌ FireEnv import failed: {e}")

try:
    print("   - Testing DQNAgent import...")
    from src.scripts.DQNAgent import DQNAgent
    print("   ✅ DQNAgent imported successfully")
except Exception as e:
    print(f"   ❌ DQNAgent import failed: {e}")

try:
    print("   - Testing DomiRankMemoryLoader import...")
    from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
    print("   ✅ DomiRankMemoryLoader imported successfully")
except Exception as e:
    print(f"   ❌ DomiRankMemoryLoader import failed: {e}")

print(f"\n🎬 All imports successful! You can now run the visualization.")