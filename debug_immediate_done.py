#!/usr/bin/env python3
"""
Debug script to understand why environments are immediately returning done=True
"""

import sys
import numpy as np
sys.path.append('src')

def test_immediate_done():
    """Test why environments are completing immediately."""
    print("Testing immediate done issue...")
    
    # Try to import and create a simple environment
    try:
        from utils.loadingUtils import RasterManager
        
        # Load a raster
        raster_manager = RasterManager("cropped_raster", 500)
        raster_manager.load_all_rasters()
        raster = raster_manager.all_rasters[0]
        print(f"✅ Loaded raster")
        
        from Env import FuelBreakEnv
        
        # Create environment with same parameters as training script
        budget = 250
        kstep = 10
        sims = 1
        
        print(f"Creating environment with budget={budget}, kstep={kstep}, sims={sims}")
        
        env = FuelBreakEnv(
            raster,
            break_budget=budget,
            break_step=kstep,
            num_simulations=sims,
            seed=42,
        )
        
        print(f"Environment created: budget={env.break_budget}, used={env._used}")
        
        # Reset environment
        obs, info = env.reset()
        print(f"After reset: used={env._used}, obs shape={obs.shape}")
        
        # Try a few steps
        for step in range(5):
            # Create action with 10 fuel breaks
            action = np.zeros(obs.shape[-2] * obs.shape[-1])
            action[:kstep] = 1  # Place 10 fuel breaks
            
            print(f"\nStep {step + 1}:")
            print(f"  Action sum: {np.sum(action)}")
            print(f"  Before step: used={env._used}")
            
            obs, reward, done, truncated, info = env.step(action)
            
            new_cells = info.get("new_cells", 0)
            burned = info.get("burned", 0)
            
            print(f"  After step: used={env._used}, new_cells={new_cells}, burned={burned:.1f}")
            print(f"  Result: reward={reward:.3f}, done={done}, truncated={truncated}")
            
            if done:
                print(f"  Episode completed at step {step + 1}")
                break
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_immediate_done()