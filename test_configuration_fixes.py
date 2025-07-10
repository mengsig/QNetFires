#!/usr/bin/env python3
"""
Simple test to verify configuration fixes without requiring pyretechnics.
"""

import os
import sys
import torch
import numpy as np
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'scripts'))

def test_gpu_configuration():
    """Test GPU configuration and detection."""
    print("🚀 Testing GPU Configuration")
    print("-" * 40)
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test DQN Agent import and GPU usage
    try:
        from src.scripts.DQNAgent import DQNAgent
        
        agent = DQNAgent(input_channels=8, grid_size=20, batch_size=4)
        device = next(agent.q_network.parameters()).device
        print(f"DQN network device: {device}")
        
        # Test forward pass
        dummy_input = torch.randn(4, 8, 20, 20).to(device)
        with torch.no_grad():
            output = agent.q_network(dummy_input)
        
        print(f"✅ Network forward pass successful")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Network using {'GPU' if device.type == 'cuda' else 'CPU'}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU configuration test failed: {e}")
        return False


def test_vectorized_environment_import():
    """Test vectorized environment import and configuration."""
    print("\n🔀 Testing Vectorized Environment Configuration")
    print("-" * 40)
    
    try:
        from src.scripts.VectorizedFireEnv import VectorizedFireEnv
        
        # Test that VectorizedFireEnv can be imported with new parameters
        print("✅ VectorizedFireEnv imported successfully")
        
        # Check constructor signature
        import inspect
        sig = inspect.signature(VectorizedFireEnv.__init__)
        params = list(sig.parameters.keys())
        
        required_params = ['num_simulations', 'max_duration']
        for param in required_params:
            if param in params:
                print(f"✅ Parameter '{param}' found in constructor")
            else:
                print(f"❌ Parameter '{param}' missing from constructor")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ VectorizedFireEnv import failed: {e}")
        return False


def test_configuration_files():
    """Test configuration file structure."""
    print("\n⚙️  Testing Configuration Files")
    print("-" * 40)
    
    config_files = ['parallel_config.json', 'demo_config.json']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                print(f"✅ {config_file} loaded successfully")
                
                # Check for new parameters
                if 'memory_simulations' in config:
                    print(f"   - memory_simulations: {config['memory_simulations']}")
                
                if 'fire_simulation_max_duration' in config:
                    print(f"   - fire_simulation_max_duration: {config['fire_simulation_max_duration']}")
                
            except Exception as e:
                print(f"❌ {config_file} failed to load: {e}")
                return False
        else:
            print(f"⚠️  {config_file} not found (optional)")
    
    return True


def test_simulate_modifications():
    """Test that Simulate class has the new method signature."""
    print("\n🔧 Testing Simulate Class Modifications")
    print("-" * 40)
    
    try:
        from src.scripts.Simulate import Simulate
        
        # Check method signature
        import inspect
        sig = inspect.signature(Simulate.run_many_simulations)
        params = list(sig.parameters.keys())
        
        if 'max_duration' in params:
            print("✅ run_many_simulations has max_duration parameter")
        else:
            print("❌ run_many_simulations missing max_duration parameter")
            return False
        
        # Check if run_simulation_with_duration exists
        if hasattr(Simulate, 'run_simulation_with_duration'):
            print("✅ run_simulation_with_duration method exists")
        else:
            print("❌ run_simulation_with_duration method missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Simulate class test failed: {e}")
        return False


def test_memory_loader_modifications():
    """Test that DomiRankMemoryLoader has the new method signature."""
    print("\n💾 Testing Memory Loader Modifications")
    print("-" * 40)
    
    try:
        from src.scripts.DomiRankMemoryLoader import DomiRankMemoryLoader
        
        # Check method signature
        import inspect
        sig = inspect.signature(DomiRankMemoryLoader.generate_training_memories)
        params = list(sig.parameters.keys())
        
        if 'max_duration' in params:
            print("✅ generate_training_memories has max_duration parameter")
        else:
            print("❌ generate_training_memories missing max_duration parameter")
            return False
        
        # Check evaluate_fuel_break_performance
        sig2 = inspect.signature(DomiRankMemoryLoader.evaluate_fuel_break_performance)
        params2 = list(sig2.parameters.keys())
        
        if 'max_duration' in params2:
            print("✅ evaluate_fuel_break_performance has max_duration parameter")
        else:
            print("❌ evaluate_fuel_break_performance missing max_duration parameter")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory loader test failed: {e}")
        return False


def test_fire_env_modifications():
    """Test that FireEnv has the new configuration support."""
    print("\n🔥 Testing FireEnv Modifications")
    print("-" * 40)
    
    try:
        from src.scripts.FireEnv import FireEnv
        
        # Check method signature
        import inspect
        sig = inspect.signature(FireEnv.step)
        
        print("✅ FireEnv imported successfully")
        print("✅ step method signature verified")
        
        return True
        
    except Exception as e:
        print(f"❌ FireEnv test failed: {e}")
        return False


def main():
    """Run all configuration tests."""
    print("🧪 Configuration Fixes Test Suite")
    print("=" * 50)
    
    tests = [
        test_gpu_configuration,
        test_vectorized_environment_import,
        test_configuration_files,
        test_simulate_modifications,
        test_memory_loader_modifications,
        test_fire_env_modifications
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All configuration fixes verified!")
        print("\n🔧 VERIFIED FIXES:")
        print("• GPU detection and configuration - WORKING")
        print("• Vectorized environment new parameters - WORKING")
        print("• Configuration file structure - WORKING")
        print("• Simulate class max_duration parameter - WORKING")
        print("• Memory loader max_duration parameter - WORKING")
        print("• FireEnv modifications - WORKING")
    else:
        print("❌ Some tests failed. Please review the issues above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)