#!/usr/bin/env python3
"""
Script to verify that configuration loading works correctly.
"""

import json
import sys
import os

def test_config_loading():
    """Test configuration loading and parameter flow."""
    print("🧪 Testing Configuration Loading")
    print("=" * 50)
    
    # Test with the provided config file
    config_files = ['parallel_config.json', 'demo_config.json']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n📄 Testing config file: {config_file}")
            
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                print(f"✅ Successfully loaded {config_file}")
                
                # Check key parameters
                key_params = [
                    'num_parallel_envs', 
                    'memory_simulations', 
                    'num_episodes', 
                    'grid_size',
                    'num_landscapes'
                ]
                
                print(f"📊 Configuration values:")
                for param in key_params:
                    value = config.get(param, 'NOT SET')
                    print(f"   - {param}: {value}")
                
                # Verify expected values for your 50-environment config
                if config.get('num_parallel_envs') == 50:
                    print("✅ Found 50 parallel environments config!")
                elif config.get('num_parallel_envs'):
                    print(f"ℹ️  Found {config.get('num_parallel_envs')} parallel environments")
                else:
                    print("❌ num_parallel_envs not set in config")
                
            except Exception as e:
                print(f"❌ Failed to load {config_file}: {e}")
        else:
            print(f"⚠️  Config file not found: {config_file}")
    
    print(f"\n🎯 To use your config with 50 environments:")
    print(f"python src/scripts/train_dqn_fuel_breaks_parallel.py --config your_config_file.json")
    
    print(f"\n🔧 If still using 4 environments, check:")
    print(f"1. Config file path is correct")
    print(f"2. Config file contains 'num_parallel_envs': 50")
    print(f"3. No command line --num_parallel_envs argument is overriding")


if __name__ == "__main__":
    test_config_loading()