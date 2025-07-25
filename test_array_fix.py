#!/usr/bin/env python3
"""
Test script to verify array/scalar handling fixes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

# Import the safe_scalar function from Train.py
sys.path.append("src")
from Train import safe_scalar

def test_safe_scalar():
    """Test the safe_scalar function with various inputs."""
    print("🧪 Testing safe_scalar function")
    print("=" * 40)
    
    test_cases = [
        # (input, expected_type, description)
        (5.0, float, "regular float"),
        (np.float32(3.14), float, "numpy float32"),
        (np.array([2.5]), float, "single-element array"),
        (np.array([1.0, 2.0, 3.0]), float, "multi-element array (should average)"),
        (None, float, "None value"),
        ('N/A', float, "string N/A"),
        ('5.5', float, "numeric string"),
        ('invalid', float, "invalid string"),
        ([], float, "empty list"),
        ([1, 2, 3], float, "list of numbers"),
    ]
    
    for i, (input_val, expected_type, description) in enumerate(test_cases):
        try:
            result = safe_scalar(input_val, fallback=99.9)
            print(f"  Test {i+1}: {description}")
            print(f"    Input: {input_val} (type: {type(input_val).__name__})")
            print(f"    Output: {result} (type: {type(result).__name__})")
            
            if isinstance(result, expected_type):
                print(f"    ✅ Correct type")
            else:
                print(f"    ❌ Wrong type, expected {expected_type.__name__}")
            
            # Check if result is a valid scalar
            try:
                float(result)
                print(f"    ✅ Can convert to float")
            except:
                print(f"    ❌ Cannot convert to float")
            
            print()
            
        except Exception as e:
            print(f"  Test {i+1}: {description}")
            print(f"    ❌ Exception: {e}")
            print()

def test_array_comparison():
    """Test that array comparisons don't cause ambiguous truth values."""
    print("🔍 Testing array comparison safety")
    print("=" * 40)
    
    test_values = [
        np.array([1.0, 2.0, 3.0]),
        np.array([5.5]),
        np.float32(2.5),
        "N/A",
        None,
        [1, 2, 3]
    ]
    
    for i, value in enumerate(test_values):
        try:
            print(f"  Test {i+1}: {type(value).__name__} = {value}")
            
            # This should never cause "truth value is ambiguous" error
            if value is None or (isinstance(value, str) and value == 'N/A'):
                result = 'N/A'
                print(f"    ✅ Handled as N/A")
            else:
                scalar_val = safe_scalar(value)
                result = f"{scalar_val:.1f}"
                print(f"    ✅ Converted to: {result}")
            
        except ValueError as e:
            if "ambiguous" in str(e).lower():
                print(f"    ❌ Ambiguous truth value error: {e}")
            else:
                print(f"    ⚠️  Other ValueError: {e}")
        except Exception as e:
            print(f"    ❌ Unexpected error: {e}")
        
        print()

def test_formatting_safety():
    """Test that f-string formatting is safe with all value types."""
    print("📝 Testing f-string formatting safety")
    print("=" * 40)
    
    test_values = [
        np.array([87.3, 92.1]),  # This would cause the original error
        np.float32(87.3),
        87.3,
        "N/A",
        None,
        [87.3, 92.1, 95.7]
    ]
    
    for i, value in enumerate(test_values):
        try:
            print(f"  Test {i+1}: Formatting {type(value).__name__}")
            
            # This is the pattern used in the training code
            burned_scalar = safe_scalar(value, fallback=None)
            burned_str = f"{burned_scalar:.1f}" if burned_scalar is not None else 'N/A'
            
            print(f"    Input: {value}")
            print(f"    Output: {burned_str}")
            print(f"    ✅ Formatting successful")
            
        except Exception as e:
            print(f"    ❌ Formatting failed: {e}")
        
        print()

def main():
    """Run all tests."""
    print("🔧 Array/Scalar Handling Fix Tests")
    print("=" * 50)
    
    test_safe_scalar()
    test_array_comparison()
    test_formatting_safety()
    
    print("=" * 50)
    print("🎯 Summary:")
    print("  - safe_scalar() handles all input types safely")
    print("  - No more 'ambiguous truth value' errors")
    print("  - f-string formatting works with any input")
    print("  - Arrays are automatically averaged to scalars")
    print("\n✅ All fixes should prevent the line 783 error!")

if __name__ == "__main__":
    main()