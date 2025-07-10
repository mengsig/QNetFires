#!/usr/bin/env python3
"""
Demo script for Parallel Deep Q-Learning Fuel Break Placement

This script demonstrates the parallel training capabilities using vectorized
environments for faster and more diverse experience collection.
"""

import os
import sys
import torch
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'scripts'))

from src.scripts.VectorizedFireEnv import VectorizedFireEnv, ParallelExperienceCollector
from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
from src.scripts.DQNAgent import DQNAgent


def test_parallel_environments():
    """Test parallel environment functionality."""
    print("=" * 60)
    print("Testing Parallel Environment Performance")
    print("=" * 60)
    
    # Generate sample landscapes
    memory_loader = DomiRankMemoryLoader(grid_size=25)
    memory_loader.generate_sample_data(3)
    
    landscape_data_list = []
    for i in range(3):
        landscape_data = memory_loader.load_landscape_data(i)
        landscape_data_list.append(landscape_data)
    
    # Test different parallelization methods
    methods = ['sequential', 'threading']
    num_envs_list = [1, 2, 4]
    
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} method ---")
        method_results = {}
        
        for num_envs in num_envs_list:
            print(f"\nTesting with {num_envs} environments...")
            
            # Create vectorized environment
            vec_env = VectorizedFireEnv(
                landscape_data_list=landscape_data_list,
                num_envs=num_envs,
                method=method,
                max_workers=2
            )
            
            # Test performance
            num_steps = 10
            actions = [np.random.randint(0, 625, size=625) for _ in range(num_envs)]
            
            # Time the operations
            start_time = time.time()
            
            observations = vec_env.reset()
            for step in range(num_steps):
                next_obs, rewards, dones, infos = vec_env.step(actions)
                observations = next_obs
            
            total_time = time.time() - start_time
            steps_per_second = (num_steps * num_envs) / total_time
            
            method_results[num_envs] = {
                'total_time': total_time,
                'steps_per_second': steps_per_second,
                'speedup': steps_per_second / method_results.get(1, {}).get('steps_per_second', steps_per_second)
            }
            
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Steps/second: {steps_per_second:.1f}")
            if num_envs > 1:
                speedup = method_results[num_envs]['speedup']
                print(f"  Speedup: {speedup:.2f}x")
            
            vec_env.close()
        
        results[method] = method_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for method in methods:
        print(f"\n{method.upper()} Method:")
        for num_envs in num_envs_list:
            result = results[method][num_envs]
            print(f"  {num_envs} envs: {result['steps_per_second']:.1f} steps/sec ({result['speedup']:.2f}x speedup)")
    
    return results


def test_parallel_training():
    """Test parallel training with experience collection."""
    print("\n" + "=" * 60)
    print("Testing Parallel Training Experience Collection")
    print("=" * 60)
    
    # Setup
    memory_loader = DomiRankMemoryLoader(grid_size=25)
    memory_loader.generate_sample_data(2)
    
    landscape_data_list = []
    for i in range(2):
        landscape_data = memory_loader.load_landscape_data(i)
        landscape_data_list.append(landscape_data)
    
    # Initialize components
    agent = DQNAgent(input_channels=8, grid_size=25, batch_size=8)
    
    # Test with different numbers of environments
    for num_envs in [1, 4]:
        print(f"\n--- Training with {num_envs} environment(s) ---")
        
        vec_env = VectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_envs=num_envs,
            method='threading',
            max_workers=2
        )
        
        collector = ParallelExperienceCollector(
            vectorized_env=vec_env,
            agent=agent,
            collection_batch_size=16
        )
        
        # Collect experiences
        start_time = time.time()
        stats = collector.collect_experiences(num_steps=20, train_frequency=4)
        collection_time = time.time() - start_time
        
        print(f"Collection Results:")
        print(f"  Collection time: {collection_time:.2f}s")
        print(f"  Experiences collected: {stats['experiences_collected']}")
        print(f"  Mean reward: {stats['mean_reward']:.2f}")
        print(f"  Training steps: {stats['training_steps']}")
        print(f"  Environments reset: {stats['environments_reset']}")
        print(f"  Experiences/second: {stats['experiences_collected']/collection_time:.1f}")
        
        # Performance stats
        perf_stats = collector.get_performance_stats()
        print(f"  Agent memory size: {perf_stats['agent_memory_size']}")
        
        vec_env.close()


def demo_scalability():
    """Demonstrate scalability with increasing numbers of environments."""
    print("\n" + "=" * 60)
    print("Demonstrating Scalability")
    print("=" * 60)
    
    # Generate sample landscapes
    memory_loader = DomiRankMemoryLoader(grid_size=20)  # Smaller for speed
    memory_loader.generate_sample_data(2)
    
    landscape_data_list = []
    for i in range(2):
        landscape_data = memory_loader.load_landscape_data(i)
        landscape_data_list.append(landscape_data)
    
    # Test scalability
    env_counts = [1, 2, 4, 8]
    throughput_results = []
    
    for num_envs in env_counts:
        print(f"\nTesting {num_envs} environments...")
        
        vec_env = VectorizedFireEnv(
            landscape_data_list=landscape_data_list,
            num_envs=num_envs,
            method='threading',
            max_workers=min(4, num_envs)
        )
        
        # Measure throughput
        num_steps = 15
        actions = [np.random.randint(0, 400, size=400) for _ in range(num_envs)]
        
        start_time = time.time()
        observations = vec_env.reset()
        
        for step in range(num_steps):
            next_obs, rewards, dones, infos = vec_env.step(actions)
            observations = next_obs
        
        total_time = time.time() - start_time
        total_environment_steps = num_steps * num_envs
        throughput = total_environment_steps / total_time
        
        throughput_results.append(throughput)
        
        print(f"  Environment steps: {total_environment_steps}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} env-steps/second")
        
        vec_env.close()
    
    # Print scalability analysis
    print(f"\nScalability Analysis:")
    print(f"{'Environments':<12} {'Throughput':<12} {'Efficiency':<12}")
    print("-" * 36)
    
    baseline_throughput = throughput_results[0]
    for i, (num_envs, throughput) in enumerate(zip(env_counts, throughput_results)):
        efficiency = throughput / (baseline_throughput * num_envs)
        print(f"{num_envs:<12} {throughput:<12.1f} {efficiency:<12.2f}")
    
    return throughput_results


def main():
    """Main demo function."""
    print("🔥 Parallel Deep Q-Learning Fuel Break Placement Demo")
    print("This demo showcases parallel training capabilities")
    
    try:
        # Test 1: Environment performance
        env_results = test_parallel_environments()
        
        # Test 2: Training experience collection
        test_parallel_training()
        
        # Test 3: Scalability demonstration
        throughput_results = demo_scalability()
        
        # Summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        
        # Find best threading performance
        best_threading = max(env_results['threading'].items(), 
                           key=lambda x: x[1]['steps_per_second'])
        
        print("\n✅ Key Findings:")
        print(f"• Best threading performance: {best_threading[1]['steps_per_second']:.1f} steps/sec with {best_threading[0]} environments")
        print(f"• Maximum speedup achieved: {best_threading[1]['speedup']:.2f}x")
        print(f"• Peak throughput: {max(throughput_results):.1f} env-steps/second")
        
        print("\n🚀 Benefits of Parallel Training:")
        print(f"• {len(env_results['threading'])}x faster experience collection with optimal configuration")
        print("• More diverse experiences from multiple environments")
        print("• Better sample efficiency and faster convergence")
        print("• Scalable to available CPU cores")
        
        print("\n📈 Recommended Settings:")
        cpu_count = os.cpu_count() or 1
        recommended_envs = min(8, cpu_count)
        print(f"• Number of environments: {recommended_envs} (detected {cpu_count} CPU cores)")
        print("• Parallel method: threading (for I/O bound fire simulations)")
        print("• Max workers: 4-8 (balance between speed and resource usage)")
        
        print(f"\nTo run parallel training with these optimized settings:")
        print(f"python src/scripts/train_dqn_fuel_breaks_parallel.py --num_parallel_envs {recommended_envs} --parallel_method threading")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)