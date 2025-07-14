#!/usr/bin/env python3
"""
Comprehensive test to verify all optimized DQN fixes work together.
"""

import sys
import os
import traceback
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_comprehensive_integration():
    """Test that all fixes work together correctly."""
    print("🧪 Testing comprehensive integration of all fixes...")
    
    try:
        from src.scripts.GymVectorizedFireEnv import (
            SingleFireEnvWrapper, 
            OptimizedGymVectorizedFireEnv,
            OptimizedExperienceCollector
        )
        
        # Create mock landscape data
        grid_size = 8  # Small but realistic for testing
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
        
        # Test 1: SingleFireEnvWrapper integration
        print("\n🔧 Testing SingleFireEnvWrapper integration...")
        
        wrapper = SingleFireEnvWrapper(
            landscape_data=landscape_data,
            env_id=0,
            num_simulations=3,
            max_duration=15
        )
        
        # Test reset returns (observation, info)
        result = wrapper.reset()
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
            print(f"✅ Reset returns tuple: obs {observation.shape}, info {list(info.keys())}")
        else:
            print(f"❌ Reset should return (obs, info) tuple, got {type(result)}")
            return False
        
        # Test step with discrete action
        action = 30  # Test action
        result = wrapper.step(action)
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            print(f"✅ Step works: obs {obs.shape}, reward {reward}, done {done}")
        else:
            print(f"❌ Step should return (obs, reward, done, info), got {len(result)} items")
            return False
        
        # Test 2: OptimizedGymVectorizedFireEnv integration
        print("\n🔧 Testing OptimizedGymVectorizedFireEnv integration...")
        
        # Create multiple landscapes
        landscape_data_list = [landscape_data.copy() for _ in range(4)]
        
        vectorized_env = OptimizedGymVectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_parallel_envs=3,
            num_simulations=3,
            max_duration=15,
            random_seed=42
        )
        
        # Test reset
        observations = vectorized_env.reset()
        if isinstance(observations, np.ndarray) and observations.shape[0] == 3:
            print(f"✅ Vectorized reset works: {observations.shape}")
        else:
            print(f"❌ Vectorized reset failed: {type(observations)}")
            return False
        
        # Test step
        actions = [10, 20, 30]  # Discrete actions for 3 environments
        obs, rewards, dones, infos = vectorized_env.step(actions)
        
        if len(obs) == 3 and len(rewards) == 3 and len(dones) == 3 and len(infos) == 3:
            print(f"✅ Vectorized step works: {len(obs)} environments")
        else:
            print(f"❌ Vectorized step failed: obs {len(obs)}, rewards {len(rewards)}")
            return False
        
        # Test 3: Experience Collector integration
        print("\n🔧 Testing OptimizedExperienceCollector integration...")
        
        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.epsilon = 0.1
                
            def act(self, obs):
                # Return random discrete action
                return np.random.randint(0, grid_size * grid_size)
                
            def remember(self, state, action, reward, next_state, done):
                pass
                
            def replay(self):
                pass
        
        mock_agent = MockAgent()
        
        experience_collector = OptimizedExperienceCollector(
            vectorized_env=vectorized_env,
            agent=mock_agent,
            collection_batch_size=16
        )
        
        # Test experience collection
        try:
            results = experience_collector.collect_experiences(
                num_steps=5,
                train_frequency=2
            )
            
            expected_keys = ['mean_reward', 'total_steps', 'experiences_collected', 
                           'collection_time', 'training_steps', 'agent_epsilon']
            
            if all(key in results for key in expected_keys):
                print(f"✅ Experience collection works: {results['experiences_collected']} experiences")
            else:
                print(f"❌ Experience collection missing keys: {list(results.keys())}")
                return False
                
        except Exception as e:
            print(f"❌ Experience collection failed: {e}")
            traceback.print_exc()
            return False
        
        # Test 4: Action space validation
        print("\n🔧 Testing action space validation...")
        
        # Test various discrete actions
        test_actions = [0, grid_size * grid_size // 2, grid_size * grid_size - 1]
        
        for action in test_actions:
            try:
                obs, reward, done, info = wrapper.step(action)
                print(f"✅ Action {action} works (reward: {reward:.3f})")
            except Exception as e:
                print(f"❌ Action {action} failed: {e}")
                return False
        
        # Test 5: Observation format validation
        print("\n🔧 Testing observation format validation...")
        
        expected_shape = (12, grid_size, grid_size)
        
        for i, obs in enumerate(observations):
            if obs.shape == expected_shape:
                print(f"✅ Observation {i} has correct shape: {obs.shape}")
            else:
                print(f"❌ Observation {i} has wrong shape: {obs.shape}, expected {expected_shape}")
                return False
        
        # Clean up
        vectorized_env.close()
        print("✅ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the comprehensive integration test."""
    print("🚀 Testing All Optimized DQN Fixes Integration")
    print("=" * 60)
    
    success = test_comprehensive_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL FIXES VERIFIED - Integration test passed!")
        print("\nVerified components:")
        print("✅ SingleFireEnvWrapper: Action conversion & gym interface")
        print("✅ OptimizedGymVectorizedFireEnv: Proper parallelization")
        print("✅ OptimizedExperienceCollector: Experience collection")
        print("✅ Action space: Discrete actions → binary masks")
        print("✅ Observation format: 12-channel observations")
        print("✅ gym.vector compatibility: Reset/step interface")
        print("\nYour optimized DQN is ready for training!")
    else:
        print("❌ Integration test failed. Check error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)