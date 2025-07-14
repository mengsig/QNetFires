#!/usr/bin/env python3
"""
Test script to verify the gym.vector interface fix works correctly.
"""

import sys
import os
import traceback
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_gym_vector_interface():
    """Test that the gym.vector interface works correctly."""
    print("🧪 Testing gym.vector interface fix...")
    
    try:
        from src.scripts.GymVectorizedFireEnv import SingleFireEnvWrapper, OptimizedGymVectorizedFireEnv
        
        # Create mock landscape data
        grid_size = 10  # Small for testing
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
        
        # Test SingleFireEnvWrapper reset method
        try:
            wrapper = SingleFireEnvWrapper(
                landscape_data=landscape_data,
                env_id=0,
                num_simulations=5,  # Small number for testing
                max_duration=30
            )
            print("✅ SingleFireEnvWrapper created")
            
            # Test reset method - should return (observation, info)
            result = wrapper.reset()
            if isinstance(result, tuple) and len(result) == 2:
                observation, info = result
                print(f"✅ reset() returns tuple: observation shape {observation.shape}, info keys {list(info.keys())}")
                
                # Verify observation shape
                expected_shape = (12, grid_size, grid_size)
                if observation.shape == expected_shape:
                    print(f"✅ Observation shape correct: {observation.shape}")
                else:
                    print(f"❌ Observation shape incorrect: got {observation.shape}, expected {expected_shape}")
                    return False
                    
                # Verify info structure
                if 'env_id' in info and 'episode_steps' in info:
                    print("✅ Info dictionary has required keys")
                else:
                    print(f"❌ Info dictionary missing keys: {info}")
                    return False
                    
            else:
                print(f"❌ reset() should return tuple of 2 elements, got: {type(result)}")
                return False
                
        except Exception as e:
            print(f"❌ SingleFireEnvWrapper test failed: {e}")
            traceback.print_exc()
            return False
        
        # Test OptimizedGymVectorizedFireEnv
        try:
            # Create multiple landscape data for vectorized env
            landscape_data_list = [landscape_data.copy() for _ in range(3)]
            
            vectorized_env = OptimizedGymVectorizedFireEnv(
                landscape_data_list=landscape_data_list,
                num_parallel_envs=2,
                num_simulations=5,
                max_duration=30,
                random_seed=42
            )
            print("✅ OptimizedGymVectorizedFireEnv created")
            
            # Test reset method
            observations = vectorized_env.reset()
            print(f"✅ Vectorized reset() successful: {len(observations)} observations")
            
            # Verify observations shape
            if len(observations) == 2:  # num_parallel_envs
                for i, obs in enumerate(observations):
                    expected_shape = (12, grid_size, grid_size)
                    if obs.shape == expected_shape:
                        print(f"✅ Observation {i} shape correct: {obs.shape}")
                    else:
                        print(f"❌ Observation {i} shape incorrect: got {obs.shape}, expected {expected_shape}")
                        return False
            else:
                print(f"❌ Expected 2 observations, got {len(observations)}")
                return False
                
            # Clean up
            vectorized_env.close()
            print("✅ Vectorized environment closed")
            
        except Exception as e:
            print(f"❌ OptimizedGymVectorizedFireEnv test failed: {e}")
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import or general test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the gym.vector interface test."""
    print("🚀 Testing gym.vector Interface Fix")
    print("=" * 40)
    
    success = test_gym_vector_interface()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 gym.vector interface fix verified!")
        print("\nFixed issues:")
        print("✅ SingleFireEnvWrapper.reset() returns (observation, info)")
        print("✅ OptimizedGymVectorizedFireEnv.reset() handles tuple unpacking")
        print("✅ Proper gym.vector.SyncVectorEnv compatibility")
        print("✅ 12-channel observations maintained")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)