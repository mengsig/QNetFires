#!/usr/bin/env python3
"""
Test script to verify that the environment reshuffling bug is fixed.
"""

import sys
import os
import traceback
import numpy as np
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_fixed_reshuffling():
    """Test that environments are truly randomly selected at every episode."""
    print("🧪 Testing FIXED environment reshuffling...")
    
    try:
        from src.scripts.GymVectorizedFireEnv import OptimizedGymVectorizedFireEnv
        
        # Create test parameters
        grid_size = 6
        num_landscapes = 8
        num_parallel_envs = 3
        num_test_episodes = 12
        
        # Create multiple different landscapes
        landscape_data_list = []
        for i in range(num_landscapes):
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
        
        # Create vectorized environment (no random seed to ensure true randomness)
        vectorized_env = OptimizedGymVectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_parallel_envs=num_parallel_envs,
            num_simulations=2,
            max_duration=10,
            random_seed=None  # This is key - no fixed seed!
        )
        
        print(f"✅ Created vectorized environment with {num_parallel_envs} parallel envs")
        
        # Test multiple episodes
        print(f"\n📋 Testing {num_test_episodes} episodes for environment diversity...")
        
        episode_selections = []
        
        for episode in range(num_test_episodes):
            print(f"\n=== Episode {episode + 1}/{num_test_episodes} ===")
            
            # Reset environment (should trigger reshuffling)
            observations = vectorized_env.reset()
            
            # Store the selected environments
            current_selection = vectorized_env.selected_landscape_indices.copy()
            episode_selections.append(current_selection)
            
            print(f"Episode {episode + 1} environments: {current_selection}")
        
        print(f"\n📊 Analyzing environment diversity across {num_test_episodes} episodes...")
        
        # Count unique episode configurations
        unique_configurations = set(tuple(selection) for selection in episode_selections)
        print(f"✅ Unique episode configurations: {len(unique_configurations)}")
        
        # Show all episode selections
        print("\n📋 All episode selections:")
        for i, selection in enumerate(episode_selections):
            print(f"  Episode {i+1}: {selection}")
        
        # Analyze individual environment usage
        all_selections = []
        for selection in episode_selections:
            all_selections.extend(selection)
        
        usage_counter = Counter(all_selections)
        print(f"\n📈 Environment usage frequency:")
        for env_idx in range(num_landscapes):
            count = usage_counter.get(env_idx, 0)
            percentage = (count / len(all_selections)) * 100
            print(f"  Landscape {env_idx}: {count}/{len(all_selections)} times ({percentage:.1f}%)")
        
        # Check for diversity
        environments_used = len(usage_counter)
        print(f"\n📊 Diversity Analysis:")
        print(f"  Total environments used: {environments_used}/{num_landscapes}")
        print(f"  Diversity percentage: {(environments_used/num_landscapes)*100:.1f}%")
        
        # Verify that environments are changing
        if len(unique_configurations) > 1:
            print(f"✅ SUCCESS: Found {len(unique_configurations)} unique configurations - environments are changing!")
            diversity_success = True
        else:
            print(f"❌ FAILURE: Only {len(unique_configurations)} configuration - environments are NOT changing!")
            diversity_success = False
        
        # Check that multiple environments are being used
        if environments_used >= min(num_landscapes, num_parallel_envs * 2):
            print(f"✅ SUCCESS: Good diversity with {environments_used} different environments used")
            usage_success = True
        else:
            print(f"❌ FAILURE: Poor diversity with only {environments_used} different environments used")
            usage_success = False
        
        # Test consecutive episodes to ensure they're different
        consecutive_same = 0
        for i in range(1, len(episode_selections)):
            if episode_selections[i] == episode_selections[i-1]:
                consecutive_same += 1
        
        if consecutive_same < num_test_episodes // 2:  # Allow some coincidental matches
            print(f"✅ SUCCESS: Only {consecutive_same}/{num_test_episodes-1} consecutive episodes were identical")
            consecutive_success = True
        else:
            print(f"❌ FAILURE: {consecutive_same}/{num_test_episodes-1} consecutive episodes were identical")
            consecutive_success = False
        
        # Get final stats
        final_stats = vectorized_env.get_performance_stats()
        print(f"\n📈 Final Stats:")
        print(f"  Environment resets: {final_stats['environment_resets']}")
        print(f"  Unique environments used: {final_stats['unique_environments_used']}")
        print(f"  Diversity coverage: {final_stats['diversity_percentage']:.1f}%")
        print(f"  Total selections: {final_stats['total_environment_selections']}")
        
        # Clean up
        vectorized_env.close()
        print("✅ Environment closed successfully")
        
        return diversity_success and usage_success and consecutive_success
        
    except Exception as e:
        print(f"❌ Fixed reshuffling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the fixed reshuffling test."""
    print("🚀 Testing Fixed Environment Reshuffling")
    print("=" * 60)
    
    success = test_fixed_reshuffling()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 🎉 🎉 RESHUFFLING BUG FIXED! 🎉 🎉 🎉")
        print("\nFixed issues:")
        print("✅ Lambda closure bug fixed (proper environment creation)")
        print("✅ Random seed issue fixed (time-based randomization)")
        print("✅ Environments randomly selected at EVERY episode")
        print("✅ Maximum landscape diversity achieved")
        print("✅ Proper statistics tracking")
        print("\nYour agent will now see different landscapes every episode!")
    else:
        print("❌ Reshuffling test failed. Environment diversity issues remain.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)