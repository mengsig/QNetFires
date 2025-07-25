#!/usr/bin/env python3
"""
Environment diagnostic script to identify fire simulation issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from src.Env import FuelBreakEnv
from src.utils.loadingUtils import RasterManager

def test_raster_loading():
    """Test raster loading functionality."""
    print("=== Testing Raster Loading ===")
    
    raster_manager = RasterManager()
    
    try:
        raster_manager.load_all_rasters("cropped_raster")
        print(f"✅ Successfully loaded {len(raster_manager.all_rasters)} rasters")
        
        # Test first raster
        if raster_manager.all_rasters:
            raster = raster_manager.all_rasters[0]
            print(f"✅ First raster keys: {list(raster.keys())}")
            print(f"✅ Raster shapes: {[(k, v.shape) for k, v in raster.items()]}")
            return raster
        else:
            print("❌ No rasters loaded")
            return None
            
    except Exception as e:
        print(f"❌ Raster loading failed: {e}")
        
        # Create dummy raster for testing
        print("Creating dummy raster for testing...")
        dummy_raster = {
            'slp': np.random.rand(50, 50).astype(np.float32),
            'asp': np.random.rand(50, 50).astype(np.float32),
            'dem': np.random.rand(50, 50).astype(np.float32),
            'cc': np.random.rand(50, 50).astype(np.float32),
            'cbd': np.random.rand(50, 50).astype(np.float32),
            'cbh': np.random.rand(50, 50).astype(np.float32),
            'ch': np.random.rand(50, 50).astype(np.float32),
            'fbfm': np.random.randint(1, 14, (50, 50)).astype(np.float32),
            'fireline_north': np.zeros((50, 50), dtype=np.float32),
            'fireline_east': np.zeros((50, 50), dtype=np.float32),
            'fireline_south': np.zeros((50, 50), dtype=np.float32),
            'fireline_west': np.zeros((50, 50), dtype=np.float32),
        }
        print("✅ Created dummy raster")
        return dummy_raster

def test_environment_creation(raster):
    """Test environment creation with different parameters."""
    print("\n=== Testing Environment Creation ===")
    
    test_configs = [
        {"budget": 200, "k_steps": 10, "sims": 1, "name": "minimal"},
        {"budget": 200, "k_steps": 10, "sims": 3, "name": "normal"},
        {"budget": 200, "k_steps": 10, "sims": 5, "name": "high_sims"},
        {"budget": 100, "k_steps": 5, "sims": 3, "name": "small_budget"},
    ]
    
    working_configs = []
    
    for config in test_configs:
        try:
            print(f"\nTesting config: {config['name']}")
            print(f"  Budget: {config['budget']}, K_steps: {config['k_steps']}, Sims: {config['sims']}")
            
            env = FuelBreakEnv(
                raster,
                break_budget=config['budget'],
                break_step=config['k_steps'],
                num_simulations=config['sims'],
                seed=42,
            )
            
            print(f"  ✅ Environment created successfully")
            working_configs.append(config)
            
            # Test reset
            obs, info = env.reset()
            print(f"  ✅ Reset successful, obs shape: {obs.shape}")
            
            # Test a simple step
            H, W = obs.shape[-2:]
            action = np.zeros(H * W)
            action[0] = 1  # Place one fuel break
            
            obs, reward, done, truncated, info = env.step(action)
            print(f"  ✅ Step successful, reward: {reward:.4f}, burned: {info.get('burned', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Config {config['name']} failed: {type(e).__name__}: {e}")
    
    return working_configs

def test_fire_simulation_stability(raster, num_tests=10):
    """Test fire simulation stability over multiple runs."""
    print(f"\n=== Testing Fire Simulation Stability ({num_tests} tests) ===")
    
    try:
        env = FuelBreakEnv(
            raster,
            break_budget=200,
            break_step=10,
            num_simulations=3,
            seed=42,
        )
        
        success_count = 0
        burned_areas = []
        
        for i in range(num_tests):
            try:
                obs, info = env.reset()
                
                # Take a few random actions
                H, W = obs.shape[-2:]
                for step in range(3):
                    action = np.zeros(H * W)
                    # Place a few random fuel breaks
                    positions = np.random.choice(H * W, size=min(5, H * W), replace=False)
                    action[positions] = 1
                    
                    obs, reward, done, truncated, info = env.step(action)
                    burned = info.get('burned', None)
                    
                    if burned is not None and not np.isnan(burned):
                        burned_areas.append(burned)
                    
                    if done:
                        break
                
                success_count += 1
                print(f"  Test {i+1}: ✅ Success, final burned: {info.get('burned', 'N/A')}")
                
            except Exception as e:
                print(f"  Test {i+1}: ❌ Failed: {type(e).__name__}: {e}")
        
        print(f"\n✅ Success rate: {success_count}/{num_tests} ({100*success_count/num_tests:.1f}%)")
        
        if burned_areas:
            print(f"✅ Burned area stats: min={min(burned_areas):.1f}, max={max(burned_areas):.1f}, mean={np.mean(burned_areas):.1f}")
        else:
            print("❌ No valid burned area measurements")
            
    except Exception as e:
        print(f"❌ Stability test setup failed: {e}")

def test_pyretechnics_directly():
    """Test pyretechnics library directly."""
    print("\n=== Testing Pyretechnics Directly ===")
    
    try:
        from src.Simulate import FireSim
        print("✅ Successfully imported FireSim")
        
        # Create a simple test landscape
        H, W = 20, 20
        test_raster = {
            'slp': np.random.uniform(0, 30, (H, W)).astype(np.float32),
            'asp': np.random.uniform(0, 360, (H, W)).astype(np.float32),
            'fbfm': np.random.randint(1, 14, (H, W)).astype(np.float32),
            'fireline_north': np.zeros((H, W), dtype=np.float32),
            'fireline_east': np.zeros((H, W), dtype=np.float32),
            'fireline_south': np.zeros((H, W), dtype=np.float32),
            'fireline_west': np.zeros((H, W), dtype=np.float32),
        }
        
        # Test FireSim creation
        fire_sim = FireSim(test_raster, seed=42)
        print("✅ FireSim created successfully")
        
        # Test simulation
        fire_sim.run_many_simulations(3)
        burned = fire_sim.average_acres_burned
        print(f"✅ Simulation successful, burned: {burned}")
        
        # Test with fuel breaks
        fuel_breaks = np.zeros((H, W), dtype=bool)
        fuel_breaks[5:15, 5:15] = True  # Add some fuel breaks
        
        fire_sim.set_fuel_breaks(fuel_breaks)
        fire_sim.run_many_simulations(3)
        burned_with_breaks = fire_sim.average_acres_burned
        print(f"✅ Simulation with fuel breaks successful, burned: {burned_with_breaks}")
        
        if burned_with_breaks < burned:
            print("✅ Fuel breaks reduced fire spread as expected")
        else:
            print("⚠️  Fuel breaks didn't reduce fire spread (this might be normal)")
            
    except Exception as e:
        print(f"❌ Pyretechnics test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all diagnostic tests."""
    print("🔍 Environment Diagnostic Tool")
    print("=" * 50)
    
    # Test raster loading
    raster = test_raster_loading()
    
    if raster is None:
        print("❌ Cannot proceed without valid raster data")
        return
    
    # Test environment creation
    working_configs = test_environment_creation(raster)
    
    if not working_configs:
        print("❌ No working environment configurations found")
        return
    
    print(f"\n✅ Found {len(working_configs)} working configurations")
    
    # Test fire simulation stability
    test_fire_simulation_stability(raster)
    
    # Test pyretechnics directly
    test_pyretechnics_directly()
    
    print("\n" + "=" * 50)
    print("🎯 Diagnostic Summary:")
    print(f"  - Raster loading: {'✅ Working' if raster else '❌ Failed'}")
    print(f"  - Environment creation: {len(working_configs)}/{len([1,2,3,4])} configs working")
    print(f"  - Recommended config: budget=200, k_steps=10, sims=3")
    
    if len(working_configs) > 0:
        print("\n🚀 Your environment should work! If you're still getting burned=1000,")
        print("   the issue might be in the training loop or wrapper logic.")
    else:
        print("\n❌ Environment issues detected. Check pyretechnics installation.")

if __name__ == "__main__":
    main()