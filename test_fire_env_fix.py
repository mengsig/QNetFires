#!/usr/bin/env python3
"""
Test script to verify the FireEnv fix works correctly.
"""

import os
import sys
import numpy as np

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '.'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

def test_fire_env_fix():
    """Test that FireEnv works correctly with the fix."""
    print("🔥 Testing FireEnv Fix")
    print("=" * 30)
    
    try:
        from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
        from src.scripts.FireEnv import FireEnv
        
        # Load landscape data
        print("🏞️  Loading landscape data...")
        memory_loader = DomiRankMemoryLoader(grid_size=25)
        
        try:
            landscape_data = memory_loader.load_landscape_data(0)
            print("✅ Landscape data loaded")
        except:
            print("📊 Generating sample landscape data...")
            memory_loader.generate_sample_data(1)
            landscape_data = memory_loader.load_landscape_data(0)
            print("✅ Sample landscape data generated")
        
        # Create FireEnv
        print("🔧 Creating FireEnv...")
        fire_env = FireEnv(
            slope=landscape_data['slp'],
            aspect=landscape_data['asp'],
            dem=landscape_data['dem'],
            cc=landscape_data['cc'],
            cbd=landscape_data['cbd'],
            cbh=landscape_data['cbh'],
            ch=landscape_data['ch'],
            fuel_model=landscape_data['fbfm']
        )
        
        # Configure for fast testing
        fire_env.num_simulations = 2
        fire_env.max_duration = 10
        
        print("✅ FireEnv created")
        
        # Test the fix: reset before using
        print("🔄 Testing reset...")
        obs = fire_env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        
        # Test step
        print("🎯 Testing step...")
        test_action = np.zeros(fire_env.H * fire_env.W)
        test_action[:10] = 1  # Add some fuel breaks
        
        obs, reward, done, info = fire_env.step(test_action)
        print(f"✅ Step successful!")
        print(f"   - Reward: {reward:.1f}")
        print(f"   - Acres burned: {info.get('acres_burned', 'N/A')}")
        print(f"   - Done: {done}")
        
        # Test multiple steps (like in visualization)
        print("🔄 Testing multiple steps...")
        for i in range(3):
            fire_env.reset()  # Reset before each step (like in visualization)
            test_action = np.zeros(fire_env.H * fire_env.W)
            test_action[i*10:(i+1)*10] = 1  # Different fuel breaks each time
            
            obs, reward, done, info = fire_env.step(test_action)
            print(f"   Step {i+1}: reward={reward:.1f}, acres_burned={info.get('acres_burned', 'N/A')}")
        
        print("✅ All tests passed!")
        print("🎬 FireEnv fix is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_fire_env_fix()
    if success:
        print("\n🎉 You can now run the visualization!")
        print("   python launch_visualization.py")
    else:
        print("\n❌ Please fix the remaining issues before running visualization.")