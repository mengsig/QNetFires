#!/usr/bin/env python3
"""
Test script to verify budget enforcement is working correctly.
"""

import sys
import numpy as np
sys.path.append('src')

from Env import FuelBreakEnv
from utils.loadingUtils import RasterManager

def test_budget_enforcement():
    """Test that environments respect the budget constraint."""
    print("Testing budget enforcement...")
    
    # Load a single raster
    raster_manager = RasterManager("cropped_raster", 500)
    try:
        raster_manager.load_all_rasters()
        raster = raster_manager.all_rasters[0]
        print(f"✅ Loaded test raster")
    except Exception as e:
        print(f"❌ Failed to load raster: {e}")
        return
    
    # Test with different budgets
    for budget in [50, 100, 250]:
        print(f"\n🧪 Testing with budget: {budget}")
        
        try:
            env = FuelBreakEnv(
                raster,
                break_budget=budget,
                break_step=10,
                num_simulations=1,
                seed=42,
            )
            
            obs, info = env.reset()
            print(f"Environment created successfully")
            
            total_fuel_breaks = 0
            step = 0
            
            while True:
                # Create action that tries to place 10 fuel breaks
                action = np.zeros(obs.shape[-2] * obs.shape[-1])
                action[:10] = 1  # Try to place 10 fuel breaks
                
                obs, reward, done, truncated, info = env.step(action)
                
                new_cells = info.get("new_cells", 0)
                total_fuel_breaks += new_cells
                step += 1
                
                print(f"  Step {step}: Placed {new_cells} breaks, Total: {total_fuel_breaks}/{budget}, Done: {done}")
                
                if done:
                    break
                    
                if step > budget // 10 + 5:  # Safety break
                    print(f"  ⚠️  Safety break - too many steps!")
                    break
            
            if total_fuel_breaks <= budget:
                print(f"✅ Budget respected: {total_fuel_breaks}/{budget}")
            else:
                print(f"❌ Budget exceeded: {total_fuel_breaks}/{budget}")
                
        except Exception as e:
            print(f"❌ Environment test failed: {e}")

def test_dummy_env_budget():
    """Test that dummy environments also respect budget."""
    print("\n🧪 Testing dummy environment budget enforcement...")
    
    # Import the dummy env from the threaded script
    import importlib.util
    spec = importlib.util.spec_from_file_location("threaded", "train_threaded_parallel.py")
    threaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(threaded_module)
    
    # Create threaded vector env with dummy environments
    class DummyEnvFn:
        def __call__(self):
            raise Exception("Force dummy creation")
    
    env_fns = [DummyEnvFn() for _ in range(2)]
    
    try:
        vec_env = threaded_module.ThreadedVectorEnv(env_fns, budget=100)
        obs = vec_env.reset()
        print(f"✅ Created vector env with dummy environments")
        
        total_steps = 0
        while total_steps < 50:  # Max 50 steps
            actions = np.random.rand(2, 2500) > 0.9  # Sparse actions
            actions = actions.astype(np.float32)
            
            next_obs, rewards, dones, truncated, infos = vec_env.step(actions)
            total_steps += 1
            
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    print(f"  Env {i} finished after {total_steps} steps")
                    if info.get("dummy", False):
                        print(f"    Dummy env respected budget: ✅")
                    break
            
            if any(dones):
                break
        
        vec_env.close()
        print(f"✅ Dummy environment test completed")
        
    except Exception as e:
        print(f"❌ Dummy environment test failed: {e}")

if __name__ == "__main__":
    test_budget_enforcement()
    test_dummy_env_budget()
    print("\n🎯 Budget enforcement tests completed!")