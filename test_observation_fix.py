#!/usr/bin/env python3
"""
Test script to verify the observation fix for the dict issue.
"""

import os
import sys
import numpy as np
import json

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.insert(0, project_root)

from src.scripts.GymVectorizedFireEnv import OptimizedGymVectorizedFireEnv, OptimizedExperienceCollector
from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
from src.scripts.DQNAgent_optimized import OptimizedDQNAgent

def test_observation_fix():
    """Test the observation fix for the dict issue."""
    
    print("🔍 Testing observation fix for dict issue...")
    
    # Configuration
    config = {
        'raster_dir': 'cropped_raster',
        'grid_size': 50,
        'input_channels': 12,
        'num_landscapes': 3,
        'num_parallel_envs': 2,
        'memory_simulations': 10,
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'initial_epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 1000,
        'batch_size': 32
    }
    
    # Load landscape data
    memory_loader = DomiRankMemoryLoader(
        raster_dir=config['raster_dir'],
        grid_size=config['grid_size']
    )
    
    # Generate sample data if needed
    if not os.path.exists(config['raster_dir']):
        print(f"🏞️  Generating {config['num_landscapes']} landscape files...")
        memory_loader.generate_sample_data(config['num_landscapes'])
    
    # Load landscape data
    landscape_data_list = []
    for i in range(config['num_landscapes']):
        try:
            landscape_data = memory_loader.load_landscape_data(i)
            landscape_data_list.append(landscape_data)
            print(f"✅ Loaded landscape {i}")
        except Exception as e:
            print(f"❌ Failed to load landscape {i}: {e}")
            break
    
    if len(landscape_data_list) == 0:
        print("❌ No landscape data available!")
        return
    
    # Initialize vectorized environment
    print(f"\n📊 Initializing vectorized environment with {len(landscape_data_list)} landscapes...")
    vectorized_env = OptimizedGymVectorizedFireEnv(
        landscape_data_list=landscape_data_list,
        num_parallel_envs=config['num_parallel_envs'],
        num_simulations=config['memory_simulations']
    )
    
    # Initialize DQN agent
    print(f"\n🤖 Initializing DQN agent...")
    agent = OptimizedDQNAgent(
        input_channels=config['input_channels'],
        grid_size=config['grid_size'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon=config['initial_epsilon'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size']
    )
    
    # Initialize experience collector
    print(f"\n📦 Initializing experience collector...")
    experience_collector = OptimizedExperienceCollector(
        vectorized_env=vectorized_env,
        agent=agent,
        collection_batch_size=16
    )
    
    # Test reset
    print("\n🔄 Testing reset()...")
    try:
        observations = vectorized_env.reset()
        print(f"✅ Reset successful!")
        print(f"   - Type: {type(observations)}")
        print(f"   - Length: {len(observations)}")
        
        # Examine each observation
        for i, obs in enumerate(observations):
            print(f"   - Observation {i}:")
            print(f"     - Type: {type(obs)}")
            if hasattr(obs, 'shape'):
                print(f"     - Shape: {obs.shape}")
                print(f"     - ✅ Observation is an array with shape!")
            else:
                print(f"     - ❌ Observation has no shape attribute!")
                return False
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test step
    print("\n👟 Testing step()...")
    try:
        # Create dummy actions
        actions = [0, 1]  # Discrete actions
        
        observations, rewards, dones, infos = vectorized_env.step(actions)
        print(f"✅ Step successful!")
        print(f"   - Observations type: {type(observations)}")
        print(f"   - Observations length: {len(observations)}")
        
        # Examine each observation
        for i, obs in enumerate(observations):
            print(f"   - Observation {i}:")
            print(f"     - Type: {type(obs)}")
            if hasattr(obs, 'shape'):
                print(f"     - Shape: {obs.shape}")
                print(f"     - ✅ Observation is an array with shape!")
            else:
                print(f"     - ❌ Observation has no shape attribute!")
                return False
        
        print(f"   - Rewards: {rewards}")
        print(f"   - Dones: {dones}")
        
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test experience collection
    print("\n📦 Testing experience collection...")
    try:
        results = experience_collector.collect_experiences(
            num_steps=5,
            train_frequency=2
        )
        print(f"✅ Experience collection successful!")
        print(f"   - Mean reward: {results['mean_reward']:.3f}")
        print(f"   - Experiences collected: {results['experiences_collected']}")
        print(f"   - Episodes completed: {results['episodes_completed']}")
        print(f"   - Training steps: {results['training_steps']}")
        
    except Exception as e:
        print(f"❌ Experience collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    vectorized_env.close()
    print("\n🧹 Cleanup completed!")
    print("✅ All tests passed! The observation fix is working correctly.")
    return True

if __name__ == "__main__":
    success = test_observation_fix()
    if success:
        print("\n🎉 Test successful! The dict observation issue has been fixed.")
    else:
        print("\n❌ Test failed! The dict observation issue still exists.")
    sys.exit(0 if success else 1)