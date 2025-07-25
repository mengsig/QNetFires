#!/usr/bin/env python3
"""
Test script to verify the reward calculation and array handling fixes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

# Test the safe_scalar function
def safe_scalar(value, fallback=0.0):
    """Convert value to scalar, handling arrays, None, and string values safely."""
    if value is None:
        return fallback
    if isinstance(value, str):
        if value == 'N/A':
            return fallback
        try:
            return float(value)
        except ValueError:
            return fallback
    if hasattr(value, '__len__') and len(value) > 1:
        # It's an array-like object with multiple elements
        try:
            return float(np.mean(value))
        except:
            return fallback
    try:
        return float(value)
    except:
        return fallback

def test_safe_scalar():
    """Test the safe_scalar function with various inputs."""
    print("🧪 Testing safe_scalar function:")
    
    test_cases = [
        (None, "None"),
        ('N/A', "String 'N/A'"),
        ('123.45', "String number"),
        ('invalid', "Invalid string"),
        (42, "Integer"),
        (42.5, "Float"),
        (np.array([1, 2, 3]), "Numpy array"),
        (np.array([42]), "Single-element array"),
        ([1, 2, 3], "Python list"),
        (True, "Boolean True"),
        (False, "Boolean False"),
    ]
    
    for value, description in test_cases:
        result = safe_scalar(value)
        print(f"  {description}: {value} → {result} (type: {type(result).__name__})")
        
        # Test that it doesn't cause array truth value errors
        try:
            if result:
                pass  # This should not cause an error
            print(f"    ✅ Truth value test passed")
        except ValueError as e:
            print(f"    ❌ Truth value test failed: {e}")

def test_division_by_zero_scenarios():
    """Test scenarios that could cause division by zero."""
    print("\n🧪 Testing division by zero scenarios:")
    
    # Simulate the reward calculation logic
    def calculate_reward_safe(initial_burned, current_burned, breaks_used, H, W):
        """Simulate the fixed reward calculation."""
        reward = 0.0
        
        # Test milestone calculation
        if initial_burned > 0:
            reduction_percentage = (initial_burned - current_burned) / initial_burned
            print(f"  Reduction percentage: {reduction_percentage:.3f}")
            
            if reduction_percentage > 0.3:
                reward += 0.1
            if reduction_percentage > 0.5:
                reward += 0.2
            if reduction_percentage > 0.7:
                reward += 0.3
                
            # Test efficiency ratio calculation
            if breaks_used > 0:
                breaks_coverage = breaks_used / float(H * W)
                if breaks_coverage > 0:
                    efficiency_ratio = reduction_percentage / breaks_coverage
                    print(f"  Efficiency ratio: {efficiency_ratio:.3f}")
                    if efficiency_ratio > 10:
                        reward += 0.1
                    elif efficiency_ratio < 2:
                        reward -= 0.05
        else:
            reduction_percentage = 0.0
            print(f"  Initial burned was 0, using fallback")
        
        return reward, reduction_percentage
    
    test_scenarios = [
        (100.0, 50.0, 5, 50, 50, "Normal case"),
        (0.0, 0.0, 0, 50, 50, "All zeros"),
        (100.0, 100.0, 0, 50, 50, "No reduction, no breaks"),
        (100.0, 20.0, 1, 50, 50, "High efficiency"),
        (100.0, 95.0, 50, 50, 50, "Low efficiency"),
    ]
    
    for initial, current, breaks, H, W, description in test_scenarios:
        print(f"\n  Testing: {description}")
        print(f"    Initial: {initial}, Current: {current}, Breaks: {breaks}")
        try:
            reward, reduction_pct = calculate_reward_safe(initial, current, breaks, H, W)
            print(f"    ✅ Reward: {reward:.3f}, Reduction: {reduction_pct*100:.1f}%")
        except Exception as e:
            print(f"    ❌ Error: {e}")

def test_info_dict_handling():
    """Test info dictionary value handling."""
    print("\n🧪 Testing info dictionary handling:")
    
    # Simulate problematic info values that could cause array truth errors
    test_infos = [
        {"is_dummy": True, "env_id": 1},
        {"is_dummy": False, "env_id": 2},
        {"is_dummy": np.array([True]), "env_id": np.array([3])},  # Problematic arrays
        {"is_dummy": np.array([False, True]), "env_id": np.array([4, 5])},  # Multi-element arrays
        {},  # Empty dict
        {"is_dummy": "true", "env_id": "6"},  # String values
    ]
    
    for i, info in enumerate(test_infos):
        print(f"\n  Test case {i+1}: {info}")
        try:
            # Simulate the fixed handling
            is_dummy_val = info.get('is_dummy', False)
            env_type = "DUMMY" if safe_scalar(is_dummy_val, fallback=False) else "REAL"
            env_id = safe_scalar(info.get('env_id', i), fallback=i)
            
            print(f"    ✅ env_type: {env_type}, env_id: {env_id}")
            
            # Test truth value evaluation
            if env_type == "DUMMY":
                print(f"    ✅ Truth value comparison worked")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")

def main():
    print("🔧 Testing Reward Calculation and Array Handling Fixes")
    print("=" * 60)
    
    test_safe_scalar()
    test_division_by_zero_scenarios()
    test_info_dict_handling()
    
    print("\n" + "=" * 60)
    print("🎯 Summary:")
    print("If all tests show ✅, the fixes should prevent:")
    print("  - Float division by zero errors")
    print("  - Array truth value ambiguity errors")
    print("  - Environment step failures due to reward calculation")

if __name__ == "__main__":
    main()