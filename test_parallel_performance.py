#!/usr/bin/env python3
"""
Test script to verify parallel fire simulation performance improvements.
"""

import os
import sys
import time
import numpy as np
import multiprocessing as mp

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '.'))
sys.path.insert(0, project_root)

try:
    from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
    from src.scripts.VectorizedFireEnv import VectorizedFireEnv
    from src.scripts.FireEnv import FireEnv
    from src.scripts.Simulate import Simulate
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_fire_simulation_performance():
    """Test the performance difference between sequential and parallel fire simulations."""
    print("🔥 Testing Fire Simulation Performance\n")
    
    # Load a sample landscape
    memory_loader = DomiRankMemoryLoader(grid_size=50)
    try:
        landscape_data = memory_loader.load_landscape_data(0)
        print(f"✅ Loaded landscape data")
    except:
        print("📊 Generating sample landscape data...")
        memory_loader.generate_sample_data(1)
        landscape_data = memory_loader.load_landscape_data(0)
        print(f"✅ Generated and loaded landscape data")
    
    # Create a single FireEnv for testing
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
    
    # Test parameters
    num_simulations = 20
    fire_env.num_simulations = num_simulations
    
    print(f"🔧 Test Configuration:")
    print(f"   - Grid size: {landscape_data['slp'].shape}")
    print(f"   - Fire simulations per step: {num_simulations}")
    print(f"   - Available CPU cores: {mp.cpu_count()}")
    
    # Create a test action (some fuel breaks)
    test_action = np.zeros(fire_env.H * fire_env.W)
    # Add some random fuel breaks
    np.random.seed(42)
    num_fuel_breaks = 50
    random_positions = np.random.choice(len(test_action), num_fuel_breaks, replace=False)
    test_action[random_positions] = 1
    
    print(f"\n🧪 Running Performance Tests...")
    print(f"   Test action: {np.sum(test_action)} fuel breaks")
    
    # Test 1: Sequential fire simulations
    print(f"\n📊 Test 1: Sequential Fire Simulations")
    fire_env.reset()
    fire_env.sim._debug_parallel = False
    
    start_time = time.time()
    _, reward_seq, _, info_seq = fire_env.step(test_action)
    sequential_time = time.time() - start_time
    
    print(f"   ⏱️  Sequential time: {sequential_time:.2f}s")
    print(f"   🔥 Acres burned: {info_seq['acres_burned']:.1f}")
    print(f"   💰 Reward: {reward_seq:.1f}")
    
    # Test 2: Parallel fire simulations
    print(f"\n📊 Test 2: Parallel Fire Simulations")
    fire_env.reset()
    fire_env.sim._debug_parallel = True
    
    start_time = time.time()
    _, reward_par, _, info_par = fire_env.step(test_action)
    parallel_time = time.time() - start_time
    
    print(f"   ⏱️  Parallel time: {parallel_time:.2f}s")
    print(f"   🔥 Acres burned: {info_par['acres_burned']:.1f}")
    print(f"   💰 Reward: {reward_par:.1f}")
    
    # Performance comparison
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print(f"\n🚀 Performance Results:")
    print(f"   - Sequential: {sequential_time:.2f}s")
    print(f"   - Parallel:   {parallel_time:.2f}s")
    print(f"   - Speedup:    {speedup:.2f}x")
    
    if speedup > 1.2:
        print(f"   ✅ Parallel execution is {speedup:.2f}x faster!")
    elif speedup > 0.8:
        print(f"   ⚠️  Parallel execution shows minimal improvement ({speedup:.2f}x)")
    else:
        print(f"   ❌ Parallel execution is slower ({speedup:.2f}x)")
    
    print(f"\n📈 Simulation results are consistent: {abs(reward_seq - reward_par) < 1e-6}")

def test_vectorized_env_performance():
    """Test the vectorized environment performance."""
    print(f"\n🌐 Testing Vectorized Environment Performance")
    
    # Load multiple landscapes
    memory_loader = DomiRankMemoryLoader(grid_size=25)  # Smaller grid for faster testing
    
    try:
        landscape_data_list = []
        for i in range(3):
            landscape_data = memory_loader.load_landscape_data(i)
            landscape_data_list.append(landscape_data)
        print(f"✅ Loaded {len(landscape_data_list)} landscapes")
    except:
        print("📊 Generating sample landscape data...")
        memory_loader.generate_sample_data(3)
        landscape_data_list = []
        for i in range(3):
            landscape_data = memory_loader.load_landscape_data(i)
            landscape_data_list.append(landscape_data)
        print(f"✅ Generated and loaded {len(landscape_data_list)} landscapes")
    
    # Test different environment configurations
    test_configs = [
        {"num_envs": 1, "method": "sequential"},
        {"num_envs": 4, "method": "sequential"},
        {"num_envs": 4, "method": "threading"},
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['num_envs']} environments with {config['method']} method")
        
        # Create vectorized environment
        vec_env = VectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_envs=config['num_envs'],
            method=config['method'],
            num_simulations=5,  # Reduced for faster testing
            max_duration=30     # Reduced for faster testing
        )
        
        # Test performance
        start_time = time.time()
        observations = vec_env.reset()
        reset_time = time.time() - start_time
        
        # Create random actions
        actions = []
        for i in range(config['num_envs']):
            action = np.random.randint(0, 2, size=vec_env.envs[i].H * vec_env.envs[i].W)
            actions.append(action)
        
        start_time = time.time()
        next_obs, rewards, dones, infos = vec_env.step(actions)
        step_time = time.time() - start_time
        
        print(f"   ⏱️  Reset time: {reset_time:.2f}s")
        print(f"   ⏱️  Step time:  {step_time:.2f}s")
        print(f"   📊 Rewards: {[f'{r:.1f}' for r in rewards]}")
        
        vec_env.close()

if __name__ == "__main__":
    print("🧪 Parallel Fire Simulation Performance Test\n")
    
    test_fire_simulation_performance()
    test_vectorized_env_performance()
    
    print(f"\n✅ Performance testing completed!")
    print(f"🔧 For full training, use:")
    print(f"   python src/scripts/train_dqn_fuel_breaks_parallel.py")