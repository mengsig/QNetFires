#!/usr/bin/env python3
"""
Test script to verify reward calculation fixes.
"""

import os
import sys
import torch
import numpy as np
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'scripts'))

def test_fire_env_rewards():
    """Test that FireEnv is calculating non-zero rewards."""
    print("🔥 Testing FireEnv Reward Calculation")
    print("-" * 40)
    
    try:
        from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
        from src.scripts.FireEnv import FireEnv
        
        # Generate test landscape
        memory_loader = DomiRankMemoryLoader(grid_size=20)
        memory_loader.generate_sample_data(1)
        landscape_data = memory_loader.load_landscape_data(0)
        
        # Create FireEnv
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
        
        # Configure fire simulation parameters
        fire_env.num_simulations = 10
        fire_env.max_duration = 60
        
        # Test with minimal fuel breaks
        action = np.zeros(400, dtype=int)  # 20x20 = 400
        action[50] = 1  # Add one fuel break
        
        print("Running fire simulation...")
        obs, reward, done, info = fire_env.step(action)
        
        print(f"✅ Reward: {reward:.2f}")
        print(f"✅ Acres burned: {info['acres_burned']:.1f}")
        print(f"✅ Num simulations: {info['num_simulations']}")
        print(f"✅ Max duration: {info['max_duration']}")
        
        if reward == 0.0:
            print("❌ WARNING: Reward is still zero!")
            return False
        else:
            print("✅ SUCCESS: Non-zero reward calculated!")
            return True
            
    except Exception as e:
        print(f"❌ FireEnv test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vectorized_env_config():
    """Test that VectorizedFireEnv uses correct configuration."""
    print("\n🚀 Testing VectorizedFireEnv Configuration")
    print("-" * 40)
    
    try:
        from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
        from src.scripts.VectorizedFireEnv import VectorizedFireEnv
        
        # Generate test landscapes
        memory_loader = DomiRankMemoryLoader(grid_size=15)
        memory_loader.generate_sample_data(2)
        
        landscape_data_list = []
        for i in range(2):
            landscape_data = memory_loader.load_landscape_data(i)
            landscape_data_list.append(landscape_data)
        
        # Test configuration
        test_config = {
            'num_envs': 8,
            'method': 'threading',
            'max_workers': 4,
            'num_simulations': 15,
            'max_duration': 90
        }
        
        print(f"Testing with config: {test_config}")
        
        vec_env = VectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_envs=test_config['num_envs'],
            method=test_config['method'],
            max_workers=test_config['max_workers'],
            num_simulations=test_config['num_simulations'],
            max_duration=test_config['max_duration']
        )
        
        print(f"✅ VectorizedFireEnv created with {vec_env.num_envs} environments")
        
        # Test basic functionality
        observations = vec_env.reset()
        print(f"✅ Reset successful: {len(observations)} observations")
        
        # Test step with simple actions
        actions = [np.zeros(225, dtype=int) for _ in range(vec_env.num_envs)]  # 15x15 = 225
        for i, action in enumerate(actions):
            action[i*10] = 1  # Add different fuel breaks for each env
        
        next_obs, rewards, dones, infos = vec_env.step(actions)
        
        print(f"✅ Step successful:")
        print(f"   - Rewards: {rewards[:3]}...")  # Show first 3
        print(f"   - Non-zero rewards: {sum(1 for r in rewards if r != 0)}/{len(rewards)}")
        
        vec_env.close()
        
        if all(r == 0 for r in rewards):
            print("❌ WARNING: All rewards are zero!")
            return False
        else:
            print("✅ SUCCESS: Some non-zero rewards found!")
            return True
            
    except Exception as e:
        print(f"❌ VectorizedFireEnv test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading and parameter flow."""
    print("\n⚙️  Testing Configuration Loading")
    print("-" * 40)
    
    # Create test config
    test_config = {
        'raster_dir': 'cropped_raster',
        'grid_size': 15,
        'num_landscapes': 2,
        'num_parallel_envs': 6,
        'parallel_method': 'threading',
        'max_workers': 3,
        'memory_simulations': 8,
        'fire_simulation_max_duration': 75
    }
    
    print(f"Test config created:")
    for key, value in test_config.items():
        print(f"   - {key}: {value}")
    
    try:
        from src.scripts.train_dqn_fuel_breaks_parallel import ParallelFuelBreakTrainer
        
        print("\nInitializing trainer with test config...")
        trainer = ParallelFuelBreakTrainer(test_config)
        
        print(f"\n✅ Configuration verification:")
        print(f"   - Landscapes loaded: {len(trainer.landscape_data_list)}")
        print(f"   - Vectorized envs: {trainer.vectorized_env.num_envs}")
        print(f"   - Fire simulations: {trainer.vectorized_env.num_simulations}")
        print(f"   - Max duration: {trainer.vectorized_env.max_duration}")
        
        trainer.vectorized_env.close()
        
        # Check if configuration matches
        config_matches = (
            trainer.vectorized_env.num_envs == test_config['num_parallel_envs'] and
            trainer.vectorized_env.num_simulations == test_config['memory_simulations'] and
            trainer.vectorized_env.max_duration == test_config['fire_simulation_max_duration']
        )
        
        if config_matches:
            print("✅ SUCCESS: Configuration properly applied!")
            return True
        else:
            print("❌ WARNING: Configuration mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Reward and Configuration Fix Test Suite")
    print("=" * 50)
    
    tests = [
        test_fire_env_rewards,
        test_vectorized_env_config,
        test_config_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All fixes verified!")
        print("\n🔧 ISSUES FIXED:")
        print("• FireEnv reward calculation - WORKING")
        print("• VectorizedFireEnv configuration - WORKING")  
        print("• Configuration parameter flow - WORKING")
        print("• Fire simulation parameters - WORKING")
    else:
        print("❌ Some issues remain. Check the failures above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)