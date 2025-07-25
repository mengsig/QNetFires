#!/usr/bin/env python3
"""
Test script to verify action format works correctly
"""

import sys
import numpy as np
sys.path.append('src')

def test_action_format():
    """Test that actions work correctly with FuelBreakEnv."""
    print("Testing action format...")
    
    # Create a simple mock raster
    mock_raster = {
        'slp': np.random.rand(50, 50).astype(np.float32),
        'asp': np.random.rand(50, 50).astype(np.float32),
        'fbfm': np.random.randint(1, 14, (50, 50)).astype(np.int32),
        'fireline_north': np.zeros((50, 50), dtype=np.int32),
        'fireline_east': np.zeros((50, 50), dtype=np.int32),
        'fireline_south': np.zeros((50, 50), dtype=np.int32),
        'fireline_west': np.zeros((50, 50), dtype=np.int32),
    }
    
    try:
        from Env import FuelBreakEnv
        
        # Create environment
        env = FuelBreakEnv(
            mock_raster,
            break_budget=50,
            break_step=10,
            num_simulations=1,
            seed=42,
        )
        
        obs, info = env.reset()
        print(f"✅ Environment created and reset")
        
        # Test different action formats
        H, W = obs.shape[-2], obs.shape[-1]
        
        # Test 1: int8 action (what environment expects)
        print("\n🧪 Test 1: int8 action")
        action_int8 = np.zeros(H * W, dtype=np.int8)
        action_int8[:5] = 1  # Place 5 fuel breaks
        
        obs, reward, done, truncated, info = env.step(action_int8)
        new_cells = info.get("new_cells", 0)
        print(f"  Action dtype: {action_int8.dtype}, sum: {np.sum(action_int8)}")
        print(f"  Result: new_cells={new_cells}, done={done}")
        
        if new_cells > 0:
            print("  ✅ int8 action works!")
        else:
            print("  ❌ int8 action failed!")
        
        # Test 2: float32 action (converted)
        print("\n🧪 Test 2: float32 action")
        action_float32 = np.zeros(H * W, dtype=np.float32)
        action_float32[5:10] = 1.0  # Place 5 more fuel breaks
        
        obs, reward, done, truncated, info = env.step(action_float32)
        new_cells = info.get("new_cells", 0)
        print(f"  Action dtype: {action_float32.dtype}, sum: {np.sum(action_float32)}")
        print(f"  Result: new_cells={new_cells}, done={done}")
        
        if new_cells > 0:
            print("  ✅ float32 action works!")
        else:
            print("  ❌ float32 action failed!")
            
        print(f"\nTotal fuel breaks used: {env._used}/50")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_action_format()