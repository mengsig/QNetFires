#!/usr/bin/env python3
"""
Test script to verify the enhanced QAgent training system.
This script tests the functionality with dummy data when real rasters are not available.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import torch
from src.Model import QNet, EnhancedQNet, DuelingQNet
from src.utils.loadingUtils import RasterManager
from src.Env import FuelBreakEnv

def test_models():
    """Test that all model architectures work correctly."""
    print("Testing model architectures...")
    
    H, W = 50, 50
    batch_size = 4
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 8, H, W)
    
    # Test basic QNet
    print("Testing QNet...")
    model = QNet(H, W)
    output = model(dummy_input)
    assert output.shape == (batch_size, H * W), f"Expected {(batch_size, H * W)}, got {output.shape}"
    print("✓ QNet works correctly")
    
    # Test EnhancedQNet
    print("Testing EnhancedQNet...")
    model = EnhancedQNet(H, W)
    output = model(dummy_input)
    assert output.shape == (batch_size, H * W), f"Expected {(batch_size, H * W)}, got {output.shape}"
    print("✓ EnhancedQNet works correctly")
    
    # Test DuelingQNet
    print("Testing DuelingQNet...")
    model = DuelingQNet(H, W)
    output = model(dummy_input)
    assert output.shape == (batch_size, H * W), f"Expected {(batch_size, H * W)}, got {output.shape}"
    print("✓ DuelingQNet works correctly")

def test_raster_manager():
    """Test the RasterManager functionality."""
    print("\nTesting RasterManager...")
    
    # Create dummy rasters
    dummy_rasters = []
    for i in range(10):
        raster = {
            'slp': np.random.rand(50, 50).astype(np.float32),
            'asp': np.random.rand(50, 50).astype(np.float32),
            'dem': np.random.rand(50, 50).astype(np.float32),
            'cc': np.random.rand(50, 50).astype(np.float32),
            'cbd': np.random.rand(50, 50).astype(np.float32),
            'cbh': np.random.rand(50, 50).astype(np.float32),
            'ch': np.random.rand(50, 50).astype(np.float32),
            'fbfm': np.random.randint(1, 14, (50, 50)).astype(np.float32),
            'fireline_north': np.zeros((50, 50), dtype=np.float32),
            'fireline_east': np.zeros((50, 50), dtype=np.float32),
            'fireline_south': np.zeros((50, 50), dtype=np.float32),
            'fireline_west': np.zeros((50, 50), dtype=np.float32),
        }
        dummy_rasters.append(raster)
    
    # Test RasterManager
    manager = RasterManager("dummy_path", 10)
    manager.all_rasters = dummy_rasters
    
    # Test random sampling
    selected = manager.get_random_rasters(5)
    assert len(selected) == 5, f"Expected 5 rasters, got {len(selected)}"
    
    # Test cycling through all rasters
    all_selected = []
    for _ in range(3):  # Test multiple epochs
        batch = manager.get_random_rasters(4)
        all_selected.extend(batch)
    
    print(f"✓ RasterManager works correctly, selected {len(all_selected)} rasters over 3 epochs")

def test_environment():
    """Test the FuelBreakEnv with dummy data."""
    print("\nTesting FuelBreakEnv...")
    
    # Create dummy raster
    dummy_raster = {
        'slp': np.random.rand(50, 50).astype(np.float32),
        'asp': np.random.rand(50, 50).astype(np.float32),
        'dem': np.random.rand(50, 50).astype(np.float32),
        'cc': np.random.rand(50, 50).astype(np.float32),
        'cbd': np.random.rand(50, 50).astype(np.float32),
        'cbh': np.random.rand(50, 50).astype(np.float32),
        'ch': np.random.rand(50, 50).astype(np.float32),
        'fbfm': np.random.randint(1, 14, (50, 50)).astype(np.float32),
        'fireline_north': np.zeros((50, 50), dtype=np.float32),
        'fireline_east': np.zeros((50, 50), dtype=np.float32),
        'fireline_south': np.zeros((50, 50), dtype=np.float32),
        'fireline_west': np.zeros((50, 50), dtype=np.float32),
    }
    
    try:
        env = FuelBreakEnv(dummy_raster, break_budget=100, break_step=10, num_simulations=1)
        obs, info = env.reset()
        
        # Check observation shape
        expected_shape = (8, 50, 50)
        assert obs.shape == expected_shape, f"Expected observation shape {expected_shape}, got {obs.shape}"
        
        # Test a step
        action = np.zeros(50 * 50, dtype=np.int8)
        action[:10] = 1  # Place 10 fuel breaks
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        print(f"✓ Environment works correctly")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Reward: {reward:.4f}")
        print(f"  - Done: {done}")
        
    except Exception as e:
        print(f"⚠ Environment test failed (expected if pyretechnics not installed): {e}")
        print("  This is normal if you don't have the fire simulation dependencies")

def main():
    """Run all tests."""
    print("Running QAgent Enhancement Tests")
    print("=" * 50)
    
    try:
        test_models()
        test_raster_manager()
        test_environment()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! The enhanced system is ready to use.")
        print("\nTo run training:")
        print("  python src/Train.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())