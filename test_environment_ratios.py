#!/usr/bin/env python3
"""
Environment-to-Core Ratio Testing Script

This script tests different ratios of parallel environments to CPU cores
to find the optimal configuration for fire simulation training.
"""

import os
import sys
import json
import time
import subprocess
import multiprocessing
from typing import Dict, List, Tuple

def get_cpu_count():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()

def create_test_config(base_config_path: str, num_envs: int, output_suffix: str) -> str:
    """Create a test configuration with specified number of environments."""
    
    # Load base configuration
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Modify for testing
    config['num_parallel_envs'] = num_envs
    config['max_workers'] = num_envs
    config['num_episodes'] = 5  # Quick test
    config['steps_per_episode'] = 10  # Quick test
    config['output_dir'] = f"ratio_test_{output_suffix}"
    config['regenerate_memories'] = False  # Reuse memories for fair comparison
    
    # Save test configuration
    test_config_path = f"test_config_{output_suffix}.json"
    with open(test_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return test_config_path

def run_test(config_path: str, test_name: str) -> Dict:
    """Run a single test and measure performance."""
    print(f"\n🧪 Testing {test_name}...")
    
    # Prepare command
    cmd = [
        'python3', 
        'src/scripts/train_dqn_fuel_breaks_parallel.py',
        '--config', config_path
    ]
    
    # Run test and measure time
    start_time = time.time()
    
    try:
        # Capture output to parse performance metrics
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            # Parse output for performance metrics
            output_lines = result.stdout.split('\n')
            experiences_per_sec = extract_experiences_per_sec(output_lines)
            
            return {
                'success': True,
                'elapsed_time': elapsed_time,
                'experiences_per_sec': experiences_per_sec,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            return {
                'success': False,
                'elapsed_time': elapsed_time,
                'error': result.stderr,
                'stdout': result.stdout
            }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'elapsed_time': 300,
            'error': 'Timeout after 5 minutes'
        }
    except Exception as e:
        return {
            'success': False,
            'elapsed_time': time.time() - start_time,
            'error': str(e)
        }

def extract_experiences_per_sec(output_lines: List[str]) -> float:
    """Extract experiences per second from training output."""
    for line in output_lines:
        if 'Experiences/second:' in line:
            try:
                # Extract number from line like "Experiences/second: 12.5"
                parts = line.split('Experiences/second:')
                if len(parts) > 1:
                    value = float(parts[1].strip())
                    return value
            except (ValueError, IndexError):
                continue
    return 0.0

def test_all_ratios() -> Dict[str, Dict]:
    """Test different environment-to-core ratios."""
    cpu_count = get_cpu_count()
    print(f"🖥️  Detected {cpu_count} CPU cores")
    
    # Define test ratios
    test_ratios = [
        (1, 1),    # 1:1 ratio
        (2, 1),    # 2:1 ratio  
        (3, 1),    # 3:1 ratio
        (4, 1),    # 4:1 ratio
    ]
    
    results = {}
    base_config = 'test_ratios_config.json'
    
    print("\n🧪 RATIO TESTING PLAN:")
    print("=" * 50)
    for ratio_envs, ratio_cores in test_ratios:
        num_envs = ratio_envs * cpu_count
        print(f"  {ratio_envs}:1 ratio → {num_envs} environments")
    
    # Run tests
    for ratio_envs, ratio_cores in test_ratios:
        num_envs = ratio_envs * cpu_count
        test_name = f"{ratio_envs}to1_ratio_{num_envs}envs"
        
        # Create test configuration
        config_path = create_test_config(base_config, num_envs, test_name)
        
        # Run test
        result = run_test(config_path, f"{ratio_envs}:1 ratio ({num_envs} envs)")
        
        # Store results
        results[test_name] = {
            'ratio': f"{ratio_envs}:1",
            'num_environments': num_envs,
            'cpu_cores': cpu_count,
            **result
        }
        
        # Clean up test config
        if os.path.exists(config_path):
            os.remove(config_path)
        
        # Print immediate results
        if result['success']:
            print(f"  ✅ {test_name}: {result['experiences_per_sec']:.1f} exp/sec in {result['elapsed_time']:.1f}s")
        else:
            print(f"  ❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results

def analyze_results(results: Dict[str, Dict]) -> None:
    """Analyze test results and provide recommendations."""
    print("\n" + "=" * 60)
    print("📊 RATIO TESTING RESULTS")
    print("=" * 60)
    
    # Sort by experiences per second
    successful_tests = {k: v for k, v in results.items() if v['success']}
    
    if not successful_tests:
        print("❌ No successful tests!")
        return
    
    sorted_results = sorted(
        successful_tests.items(), 
        key=lambda x: x[1]['experiences_per_sec'], 
        reverse=True
    )
    
    print(f"{'Ratio':<8} {'Envs':<5} {'Exp/Sec':<10} {'Time':<8} {'Efficiency':<12}")
    print("-" * 60)
    
    best_exp_per_sec = 0
    best_ratio = None
    
    for test_name, result in sorted_results:
        exp_per_sec = result['experiences_per_sec']
        elapsed_time = result['elapsed_time']
        ratio = result['ratio']
        num_envs = result['num_environments']
        
        # Calculate efficiency (experiences per second per environment)
        efficiency = exp_per_sec / num_envs if num_envs > 0 else 0
        
        print(f"{ratio:<8} {num_envs:<5} {exp_per_sec:<10.1f} {elapsed_time:<8.1f} {efficiency:<12.3f}")
        
        if exp_per_sec > best_exp_per_sec:
            best_exp_per_sec = exp_per_sec
            best_ratio = ratio
    
    print("\n🏆 RECOMMENDATIONS:")
    print("=" * 30)
    
    if best_ratio:
        print(f"🥇 Best overall: {best_ratio} ratio")
        print(f"   → {best_exp_per_sec:.1f} experiences/second")
        
        # Find the configuration details
        best_config = None
        for test_name, result in results.items():
            if result['success'] and result['ratio'] == best_ratio:
                best_config = result
                break
        
        if best_config:
            print(f"   → {best_config['num_environments']} environments")
            print(f"   → {best_config['cpu_cores']} CPU cores")
    
    # Additional analysis
    print("\n💡 INSIGHTS:")
    ratios_tested = [r['ratio'] for r in successful_tests.values()]
    exp_rates = [r['experiences_per_sec'] for r in successful_tests.values()]
    
    if len(exp_rates) > 1:
        max_rate = max(exp_rates)
        min_rate = min(exp_rates)
        improvement = ((max_rate - min_rate) / min_rate) * 100
        print(f"   • Performance range: {min_rate:.1f} - {max_rate:.1f} exp/sec")
        print(f"   • Best vs worst: {improvement:.1f}% improvement")
    
    print("\n📋 NEXT STEPS:")
    print(f"   1. Update your config to use {best_ratio} ratio")
    print(f"   2. Set num_parallel_envs to {best_config['num_environments'] if best_config else 'TBD'}")
    print("   3. Monitor CPU usage during full training")

def main():
    """Main function to run ratio testing."""
    print("🚀 Environment-to-Core Ratio Testing")
    print("=" * 40)
    
    # Check if base config exists
    if not os.path.exists('test_ratios_config.json'):
        print("❌ test_ratios_config.json not found!")
        print("   Please ensure the configuration file exists.")
        return
    
    # Run tests
    results = test_all_ratios()
    
    # Analyze results
    analyze_results(results)
    
    # Save detailed results
    with open('ratio_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: ratio_test_results.json")

if __name__ == "__main__":
    main()