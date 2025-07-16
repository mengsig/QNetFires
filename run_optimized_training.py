#!/usr/bin/env python3
"""
Demo script to run the optimized DQN training for forest fire prevention.

This script demonstrates how to use the optimized components:
- OptimizedDQNAgent (reduced model size)
- OptimizedGymVectorizedFireEnv (proper gym.vector parallelization)
- Random landscape sampling
- Fixed memory leaks
- Performance monitoring
"""

import sys
import os

# Add src to path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

def main():
    """Run the optimized training."""
    print("🚀 Starting Optimized DQN Training for Forest Fire Prevention")
    print("=" * 60)
    
    # Check if gym is available
    try:
        import gym
        print("✅ gym is available")
    except ImportError:
        print("❌ gym is not available. Installing...")
        os.system("pip install gym")
    
    # Run the optimized training
    print("\n📋 Running optimized training with default configuration...")
    
    # Use the optimized training script
    cmd = "python src/scripts/train_dqn_fuel_breaks_optimized.py --config optimized_config.json"
    
    print(f"🔧 Command: {cmd}")
    print("\n" + "=" * 60)
    
    # Execute the training
    os.system(cmd)

if __name__ == "__main__":
    main()