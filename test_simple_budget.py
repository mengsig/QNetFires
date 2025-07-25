#!/usr/bin/env python3
"""
Simple test to verify budget enforcement in FuelBreakEnv
"""

import sys
import numpy as np
sys.path.append('src')

def test_basic_budget():
    """Test basic budget enforcement without complex wrappers."""
    print("Testing basic budget enforcement...")
    
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
        
        # Test with budget = 50
        budget = 50
        env = FuelBreakEnv(
            mock_raster,
            break_budget=budget,
            break_step=10,  # Max 10 per step
            num_simulations=1,
            seed=42,
        )
        
        obs, info = env.reset()
        print(f"✅ Environment created with budget {budget}")
        
        total_fuel_breaks = 0
        step = 0
        
        while True:
            # Create action to place 10 fuel breaks
            action = np.zeros(50 * 50)
            action[:10] = 1  # Try to place 10
            
            obs, reward, done, truncated, info = env.step(action)
            
            new_cells = info.get("new_cells", 0)
            total_fuel_breaks += new_cells
            step += 1
            
            print(f"  Step {step}: Placed {new_cells} breaks, Total: {total_fuel_breaks}/{budget}, Done: {done}")
            
            if done:
                print(f"✅ Episode completed with {total_fuel_breaks}/{budget} fuel breaks")
                break
                
            if step > 20:  # Safety break
                print(f"❌ Too many steps - budget not enforced!")
                break
        
        # Verify budget was enforced
        if total_fuel_breaks <= budget and done:
            print(f"✅ Budget correctly enforced: {total_fuel_breaks}/{budget}")
        else:
            print(f"❌ Budget NOT enforced: {total_fuel_breaks}/{budget}, Done: {done}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_budget()