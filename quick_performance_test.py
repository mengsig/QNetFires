#!/usr/bin/env python3
"""
Quick Performance Test for Fire Simulation

This script tests the performance difference between the original configuration
and the emergency fix configuration to validate the optimization works.
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def monitor_system_usage():
    """Get current system usage stats."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_percent': memory.percent,
        'thread_count': len(psutil.pids())
    }

def quick_simulation_test(config_file, max_episodes=3):
    """Run a quick test with the given configuration."""
    print(f"\n{'='*60}")
    print(f"Testing configuration: {config_file}")
    print(f"{'='*60}")
    
    # Monitor initial state
    initial_stats = monitor_system_usage()
    print(f"Initial CPU usage: {initial_stats['cpu_percent']:.1f}%")
    print(f"Initial memory usage: {initial_stats['memory_used_gb']:.1f}GB ({initial_stats['memory_percent']:.1f}%)")
    print(f"Initial thread count: {initial_stats['thread_count']}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Print key config parameters
    print(f"\nConfiguration:")
    print(f"  - num_parallel_envs: {config['num_parallel_envs']}")
    print(f"  - max_workers: {config['max_workers']}")
    print(f"  - memory_simulations: {config['memory_simulations']}")
    print(f"  - fire_simulation_max_duration: {config['fire_simulation_max_duration']}")
    
    # Start timer
    start_time = time.time()
    
    try:
        # Import and run the training script
        from src.scripts.train_dqn_fuel_breaks_parallel import ParallelFuelBreakTrainer
        
        # Modify config for quick test
        config['num_episodes'] = max_episodes
        config['steps_per_episode'] = min(5, config.get('steps_per_episode', 5))
        config['pretrain_steps'] = 10
        config['memory_simulations'] = min(3, config.get('memory_simulations', 3))
        
        print(f"\nStarting quick test with {max_episodes} episodes...")
        
        # Create trainer
        trainer = ParallelFuelBreakTrainer(config)
        
        # Monitor during simulation
        peak_stats = {'cpu_percent': 0, 'memory_percent': 0}
        
        # Initialize trainer (this will show if there are immediate issues)
        print("Initializing trainer...")
        trainer.initialize_with_domirank_memories()
        
        # Run a few episodes
        for episode in range(max_episodes):
            episode_start = time.time()
            
            # Monitor system during episode
            current_stats = monitor_system_usage()
            peak_stats['cpu_percent'] = max(peak_stats['cpu_percent'], current_stats['cpu_percent'])
            peak_stats['memory_percent'] = max(peak_stats['memory_percent'], current_stats['memory_percent'])
            
            print(f"\nEpisode {episode + 1}/{max_episodes}")
            print(f"  Current CPU: {current_stats['cpu_percent']:.1f}%")
            print(f"  Current memory: {current_stats['memory_percent']:.1f}%")
            
            # Run collection episode
            results = trainer.run_parallel_collection_episode(config['steps_per_episode'])
            
            episode_time = time.time() - episode_start
            print(f"  Episode time: {episode_time:.2f}s")
            print(f"  Mean reward: {results.get('mean_reward', 0):.2f}")
            print(f"  Experiences/second: {results.get('experiences_per_second', 0):.1f}")
            
            # Break if episode takes too long (sign of problems)
            if episode_time > 60:
                print("⚠️  Episode taking too long - potential performance issue")
                break
        
        # Clean up
        trainer.vectorized_env.close()
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Final stats
    final_stats = monitor_system_usage()
    
    print(f"\n{'='*40}")
    print(f"TEST RESULTS")
    print(f"{'='*40}")
    print(f"Total test time: {total_time:.2f}s")
    print(f"Peak CPU usage: {peak_stats['cpu_percent']:.1f}%")
    print(f"Peak memory usage: {peak_stats['memory_percent']:.1f}%")
    print(f"Final CPU usage: {final_stats['cpu_percent']:.1f}%")
    
    return {
        'config_file': config_file,
        'total_time': total_time,
        'peak_cpu': peak_stats['cpu_percent'],
        'peak_memory': peak_stats['memory_percent'],
        'success': True
    }

def main():
    """Run performance comparison tests."""
    print("🔥 Fire Simulation Performance Test")
    print("This will test your current vs optimized configuration")
    
    # Check which configs exist
    configs_to_test = []
    
    if os.path.exists('workstation_config_EMERGENCY_FIX.json'):
        configs_to_test.append('workstation_config_EMERGENCY_FIX.json')
    
    if os.path.exists('parallel_config.json'):
        configs_to_test.append('parallel_config.json')
    
    if not configs_to_test:
        print("❌ No configuration files found!")
        print("Expected: workstation_config_EMERGENCY_FIX.json or parallel_config.json")
        return
    
    print(f"Found {len(configs_to_test)} configuration(s) to test")
    
    results = []
    
    for config_file in configs_to_test:
        print(f"\n🧪 Testing {config_file}...")
        result = quick_simulation_test(config_file, max_episodes=2)
        if result:
            results.append(result)
        
        # Wait between tests
        time.sleep(2)
    
    # Compare results
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print(f"PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        for i, result in enumerate(results):
            print(f"Config {i+1}: {result['config_file']}")
            print(f"  Time: {result['total_time']:.2f}s")
            print(f"  Peak CPU: {result['peak_cpu']:.1f}%")
            print(f"  Peak Memory: {result['peak_memory']:.1f}%")
        
        if results[0]['total_time'] > results[1]['total_time']:
            speedup = results[0]['total_time'] / results[1]['total_time']
            print(f"\n🚀 Speedup: {speedup:.1f}x faster with optimized config!")
        
    print(f"\n✅ Performance test completed!")
    print(f"💡 Use the configuration with highest CPU usage and fastest time")

if __name__ == "__main__":
    main()