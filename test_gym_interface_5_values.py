#!/usr/bin/env python3
"""
Test script to verify the new 5-value gym interface works correctly.
"""

import sys
import os
import traceback
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_gym_interface_5_values():
    """Test that the new 5-value gym interface works correctly."""
    print("🧪 Testing new 5-value gym interface...")
    
    try:
        from src.scripts.GymVectorizedFireEnv import SingleFireEnvWrapper, OptimizedGymVectorizedFireEnv
        
        # Create mock landscape data
        grid_size = 6  # Small for testing
        landscape_data = {
            'slp': np.random.rand(grid_size, grid_size).astype(np.float32),
            'asp': np.random.rand(grid_size, grid_size).astype(np.float32),
            'dem': np.random.rand(grid_size, grid_size).astype(np.float32),
            'cc': np.random.rand(grid_size, grid_size).astype(np.float32),
            'cbd': np.random.rand(grid_size, grid_size).astype(np.float32),
            'cbh': np.random.rand(grid_size, grid_size).astype(np.float32),
            'ch': np.random.rand(grid_size, grid_size).astype(np.float32),
            'fbfm': np.random.rand(grid_size, grid_size).astype(np.float32),
            'fireline_north': np.random.rand(grid_size, grid_size).astype(np.float32),
            'fireline_south': np.random.rand(grid_size, grid_size).astype(np.float32),
            'fireline_east': np.random.rand(grid_size, grid_size).astype(np.float32),
            'fireline_west': np.random.rand(grid_size, grid_size).astype(np.float32),
        }
        
        print("✅ Mock landscape data created")
        
        # Test 1: SingleFireEnvWrapper step method returns 5 values
        print("\n🔧 Testing SingleFireEnvWrapper 5-value interface...")
        
        wrapper = SingleFireEnvWrapper(
            landscape_data=landscape_data,
            env_id=0,
            num_simulations=2,
            max_duration=10
        )
        
        # Test reset
        obs, info = wrapper.reset()
        print(f"✅ Reset works: obs shape {obs.shape}, info keys {list(info.keys())}")
        
        # Test step returns 5 values
        action = 18  # Test action
        result = wrapper.step(action)
        
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            print(f"✅ Step returns 5 values: obs {observation.shape}, reward {reward:.3f}")
            print(f"  terminated: {terminated}, truncated: {truncated}")
            print(f"  info keys: {list(info.keys())}")
            
            # Verify types
            if isinstance(terminated, (bool, np.bool_)):
                print("✅ terminated is boolean")
            else:
                print(f"❌ terminated should be boolean, got {type(terminated)}")
                return False
                
            if isinstance(truncated, (bool, np.bool_)):
                print("✅ truncated is boolean")
            else:
                print(f"❌ truncated should be boolean, got {type(truncated)}")
                return False
                
        else:
            print(f"❌ Step should return 5 values, got {len(result)}")
            return False
        
        # Test 2: OptimizedGymVectorizedFireEnv compatibility
        print("\n🔧 Testing OptimizedGymVectorizedFireEnv compatibility...")
        
        # Create multiple landscapes
        landscape_data_list = [landscape_data.copy() for _ in range(3)]
        
        vectorized_env = OptimizedGymVectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_parallel_envs=2,
            num_simulations=2,
            max_duration=10,
            random_seed=42
        )
        
        # Test reset
        observations = vectorized_env.reset()
        print(f"✅ Vectorized reset works: {len(observations)} observations")
        
        # Test step - should return 4 values for backward compatibility
        actions = [10, 20]  # Actions for 2 environments
        result = vectorized_env.step(actions)
        
        if len(result) == 4:
            obs, rewards, dones, infos = result
            print(f"✅ Vectorized step returns 4 values (backward compatible)")
            print(f"  observations: {len(obs)}, rewards: {len(rewards)}")
            print(f"  dones: {len(dones)}, infos: {len(infos)}")
            
            # Verify dones are boolean
            for i, done in enumerate(dones):
                if isinstance(done, (bool, np.bool_)):
                    print(f"✅ done[{i}] is boolean: {done}")
                else:
                    print(f"❌ done[{i}] should be boolean, got {type(done)}")
                    return False
                    
        else:
            print(f"❌ Vectorized step should return 4 values, got {len(result)}")
            return False
        
        # Test 3: Multiple step calls
        print("\n🔧 Testing multiple step calls...")
        
        for i in range(3):
            action = 5 + i * 10  # Different actions
            obs, reward, terminated, truncated, info = wrapper.step(action)
            print(f"✅ Step {i+1}: action {action}, reward {reward:.3f}, done {terminated}")
        
        # Test 4: Action boundary testing
        print("\n🔧 Testing action boundaries...")
        
        boundary_actions = [0, grid_size * grid_size - 1]
        
        for action in boundary_actions:
            try:
                obs, reward, terminated, truncated, info = wrapper.step(action)
                print(f"✅ Boundary action {action} works: reward {reward:.3f}")
            except Exception as e:
                print(f"❌ Boundary action {action} failed: {e}")
                return False
        
        # Clean up
        vectorized_env.close()
        print("✅ Vectorized environment closed")
        
        return True
        
    except Exception as e:
        print(f"❌ 5-value gym interface test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the 5-value gym interface test."""
    print("🚀 Testing 5-Value Gym Interface Fix")
    print("=" * 50)
    
    success = test_gym_interface_5_values()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 5-value gym interface fix verified!")
        print("\nFixed issues:")
        print("✅ SingleFireEnvWrapper.step() returns 5 values")
        print("✅ terminated/truncated flags properly set")
        print("✅ OptimizedGymVectorizedFireEnv handles 5-value format")
        print("✅ Backward compatibility maintained (4-value return)")
        print("✅ Action boundaries work correctly")
        print("\nYour DQN training should now work with modern gym!")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)