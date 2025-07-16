#!/usr/bin/env python3
"""
Test script to verify that environments are randomly reshuffled at every episode.
"""

import sys
import os
import traceback
import numpy as np
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_every_episode_reshuffling():
    """Test that environments are randomly selected at every episode."""
    print("🧪 Testing every-episode environment reshuffling...")
    
    try:
        from src.scripts.GymVectorizedFireEnv import OptimizedGymVectorizedFireEnv
        
        # Create multiple different landscapes
        grid_size = 8
        num_landscapes = 10
        num_parallel_envs = 4
        
        landscape_data_list = []
        for i in range(num_landscapes):
            # Create unique landscape data for each landscape
            landscape_data = {
                'slp': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'asp': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'dem': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'cc': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'cbd': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'cbh': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'ch': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'fbfm': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'fireline_north': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'fireline_south': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'fireline_east': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
                'fireline_west': np.random.rand(grid_size, grid_size).astype(np.float32) * (i + 1) / num_landscapes,
            }
            landscape_data_list.append(landscape_data)
        
        print(f"✅ Created {num_landscapes} different landscapes")
        
        # Create vectorized environment
        vectorized_env = OptimizedGymVectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_parallel_envs=num_parallel_envs,
            num_simulations=2,  # Small for testing
            max_duration=10,
            random_seed=None  # Allow true randomness
        )
        
        print(f"✅ Created vectorized environment with {num_parallel_envs} parallel envs")
        
        # Test multiple episodes to verify reshuffling
        print("\n📋 Testing environment reshuffling across episodes...")
        
        episode_selections = []
        num_test_episodes = 8
        
        for episode in range(num_test_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            
            # Reset environment (this should trigger reshuffling)
            observations = vectorized_env.reset()
            
            # Get current landscape selection
            current_selection = vectorized_env.selected_landscape_indices.copy()
            episode_selections.append(current_selection)
            
            # Verify observations
            if observations.shape == (num_parallel_envs, 12, grid_size, grid_size):
                print(f"✅ Observations shape correct: {observations.shape}")
            else:
                print(f"❌ Observations shape incorrect: {observations.shape}")
                return False
            
            # Test a few steps to ensure environment works
            for step in range(2):
                actions = [np.random.randint(0, grid_size * grid_size) for _ in range(num_parallel_envs)]
                obs, rewards, dones, infos = vectorized_env.step(actions)
                
                if len(obs) != num_parallel_envs:
                    print(f"❌ Step {step} failed: wrong number of observations")
                    return False
        
        print(f"\n📊 Analyzing environment diversity across {num_test_episodes} episodes...")
        
        # Analyze diversity
        all_selections = []
        for episode_idx, selection in enumerate(episode_selections):
            all_selections.extend(selection)
            print(f"Episode {episode_idx + 1}: {selection}")
        
        # Check if environments are actually changing
        unique_episode_selections = set(tuple(selection) for selection in episode_selections)
        
        if len(unique_episode_selections) > 1:
            print(f"✅ Environment diversity: {len(unique_episode_selections)} unique episode configurations")
        else:
            print(f"❌ No environment diversity detected: all episodes use same environments")
            return False
        
        # Analyze individual environment usage
        usage_counter = Counter(all_selections)
        print(f"\n📈 Environment usage frequency:")
        for env_idx in range(num_landscapes):
            count = usage_counter.get(env_idx, 0)
            percentage = (count / len(all_selections)) * 100
            print(f"  Landscape {env_idx}: {count} times ({percentage:.1f}%)")
        
        # Check that multiple environments are being used
        environments_used = len(usage_counter)
        if environments_used >= min(num_landscapes, num_parallel_envs):
            print(f"✅ Good diversity: {environments_used} different environments used")
        else:
            print(f"❌ Poor diversity: only {environments_used} different environments used")
            return False
        
        # Get final performance stats
        final_stats = vectorized_env.get_performance_stats()
        print(f"\n📊 Final Performance Stats:")
        print(f"  Environment resets: {final_stats['environment_resets']}")
        print(f"  Unique environments used: {final_stats['unique_environments_used']}")
        print(f"  Diversity coverage: {final_stats['diversity_percentage']:.1f}%")
        print(f"  Total environment selections: {final_stats['total_environment_selections']}")
        
        # Verify that environment resets equals number of test episodes
        if final_stats['environment_resets'] == num_test_episodes:
            print(f"✅ Environment resets match test episodes: {final_stats['environment_resets']}")
        else:
            print(f"❌ Environment resets mismatch: {final_stats['environment_resets']} != {num_test_episodes}")
            return False
        
        # Test that every episode used num_parallel_envs environments
        expected_total_selections = num_test_episodes * num_parallel_envs
        if final_stats['total_environment_selections'] == expected_total_selections:
            print(f"✅ Total environment selections correct: {final_stats['total_environment_selections']}")
        else:
            print(f"❌ Total environment selections incorrect: {final_stats['total_environment_selections']} != {expected_total_selections}")
            return False
        
        # Clean up
        vectorized_env.close()
        print("✅ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Every-episode reshuffling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the every-episode reshuffling test."""
    print("🚀 Testing Every-Episode Environment Reshuffling")
    print("=" * 60)
    
    success = test_every_episode_reshuffling()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Every-episode reshuffling verified!")
        print("\nVerified features:")
        print("✅ Environments randomly selected at every episode")
        print("✅ Environment diversity across episodes")
        print("✅ Proper statistics tracking")
        print("✅ Performance stats accuracy")
        print("✅ Resource management and cleanup")
        print("\nYour agent will now see maximum landscape diversity!")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)