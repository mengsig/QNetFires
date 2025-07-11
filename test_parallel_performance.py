#!/usr/bin/env python3
"""
Performance Testing Script for Parallel Fire Simulations

This script compares the performance of sequential vs parallel fire simulations
to demonstrate the CPU utilization improvements.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

# Add src to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.insert(0, project_root)

from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
from src.scripts.Simulate import Simulate


def test_simulation_performance():
    """Test and compare sequential vs parallel simulation performance."""
    
    print("🚀 Testing Parallel Fire Simulation Performance")
    print(f"System: {cpu_count()} CPU cores available")
    print("=" * 60)
    
    # Generate test landscape
    memory_loader = DomiRankMemoryLoader(grid_size=25)
    print("📝 Generating test landscape...")
    
    if not os.path.exists('cropped_raster'):
        memory_loader.generate_sample_data(1)
    
    landscape_data = memory_loader.load_landscape_data(0)
    
    # Test different simulation counts
    test_cases = [10, 25, 50, 100]
    results = {}
    
    for num_sims in test_cases:
        print(f"\n🔥 Testing {num_sims} simulations...")
        
        # Create simulator instance
        sim = Simulate(
            landscape_data['slp'],
            landscape_data['asp'],
            landscape_data['dem'],
            landscape_data['cc'],
            landscape_data['cbd'],
            landscape_data['cbh'],
            landscape_data['ch'],
            landscape_data['fbfm']
        )
        sim.set_space_time_cubes()
        
        # Test sequential performance
        print("   📊 Testing sequential...")
        sim.set_parallel_simulations(False)
        
        start_time = time.time()
        sim.run_many_simulations(num_sims, max_duration=30)
        sequential_time = time.time() - start_time
        
        print(f"   ⏱️  Sequential: {sequential_time:.2f}s ({num_sims/sequential_time:.1f} sims/sec)")
        
        # Test parallel performance
        print("   🚀 Testing parallel...")
        sim.set_parallel_simulations(True)
        
        start_time = time.time()
        sim.run_many_simulations(num_sims, max_duration=30)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time
        print(f"   ⚡ Parallel: {parallel_time:.2f}s ({num_sims/parallel_time:.1f} sims/sec)")
        print(f"   🎯 Speedup: {speedup:.2f}x")
        
        results[num_sims] = {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'sequential_rate': num_sims / sequential_time,
            'parallel_rate': num_sims / parallel_time
        }
    
    # Plot results
    print("\n📈 Creating performance comparison plots...")
    plot_performance_results(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    avg_speedup = np.mean([r['speedup'] for r in results.values()])
    max_speedup = max([r['speedup'] for r in results.values()])
    
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Maximum speedup: {max_speedup:.2f}x")
    print(f"Theoretical maximum: {cpu_count()}x (CPU cores)")
    
    efficiency = (avg_speedup / cpu_count()) * 100
    print(f"Parallel efficiency: {efficiency:.1f}%")
    
    if avg_speedup > 2.0:
        print("✅ EXCELLENT: Significant performance improvement!")
    elif avg_speedup > 1.5:
        print("✅ GOOD: Notable performance improvement")
    else:
        print("⚠️  MODEST: Some improvement, consider optimizing further")
    
    print("\n💡 Tips for maximum performance:")
    print("   - Use multiprocessing for environment-level parallelism")
    print("   - Adjust simulation counts based on your CPU cores")
    print("   - Monitor CPU usage with 'htop' or 'top' during training")
    print("   - Consider reducing simulation duration for faster iteration")
    
    return results


def plot_performance_results(results):
    """Create plots showing performance comparison."""
    sim_counts = list(results.keys())
    sequential_times = [results[n]['sequential_time'] for n in sim_counts]
    parallel_times = [results[n]['parallel_time'] for n in sim_counts]
    speedups = [results[n]['speedup'] for n in sim_counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Execution time comparison
    ax1.bar(np.array(sim_counts) - 1, sequential_times, width=2, label='Sequential', alpha=0.7)
    ax1.bar(np.array(sim_counts) + 1, parallel_times, width=2, label='Parallel', alpha=0.7)
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup plot
    ax2.plot(sim_counts, speedups, 'o-', linewidth=2, markersize=8, label='Actual Speedup')
    ax2.axhline(y=cpu_count(), color='red', linestyle='--', label=f'Theoretical Max ({cpu_count()}x)')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Parallel Speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parallel_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   📊 Performance plots saved as 'parallel_performance_comparison.png'")


def test_cpu_utilization():
    """Test CPU utilization with different configurations."""
    print("\n🖥️  CPU Utilization Test")
    print("=" * 30)
    print("💡 Monitor CPU usage in another terminal with:")
    print("   htop  # or 'top' on systems without htop")
    print("\nStarting CPU-intensive simulation in 5 seconds...")
    
    for i in range(5, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🔥 Running CPU-intensive simulation...")
    print("   (This should utilize multiple CPU cores)")
    
    # Generate landscape
    memory_loader = DomiRankMemoryLoader(grid_size=30)
    if not os.path.exists('cropped_raster'):
        memory_loader.generate_sample_data(1)
    
    landscape_data = memory_loader.load_landscape_data(0)
    
    # Create simulator
    sim = Simulate(
        landscape_data['slp'],
        landscape_data['asp'],
        landscape_data['dem'],
        landscape_data['cc'],
        landscape_data['cbd'],
        landscape_data['cbh'],
        landscape_data['ch'],
        landscape_data['fbfm']
    )
    sim.set_space_time_cubes()
    sim.set_parallel_simulations(True)
    
    # Run CPU-intensive simulation
    start_time = time.time()
    sim.run_many_simulations(100, max_duration=45)
    elapsed = time.time() - start_time
    
    print(f"✅ Completed in {elapsed:.2f}s")
    print("   Check if all CPU cores were utilized!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test parallel fire simulation performance')
    parser.add_argument('--cpu-test', action='store_true', 
                       help='Run CPU utilization test (monitor with htop)')
    
    args = parser.parse_args()
    
    if args.cpu_test:
        test_cpu_utilization()
    else:
        test_simulation_performance()