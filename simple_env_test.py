#!/usr/bin/env python3
"""
Simplified test to demonstrate the dummy environment issue.
This shows why you were getting identical burned areas.
"""

import numpy as np
from collections import Counter

def simulate_dummy_environment(env_id, steps=5):
    """Simulate what a DummyEnv would produce."""
    print(f"\n--- Dummy Environment {env_id} ---")
    
    # This mimics the old DummyEnv logic
    burned_areas = []
    
    for step in range(steps):
        # Simple mathematical formula (like old DummyEnv)
        base_burned = 160.0
        reduction_per_break = 6.0
        total_breaks_placed = step * 2  # Fixed pattern
        burned_area = max(80.0, base_burned - total_breaks_placed * reduction_per_break)
        
        burned_areas.append(burned_area)
        print(f"  Step {step}: burned_area = {burned_area:.1f}")
    
    return burned_areas

def simulate_real_environment(env_id, steps=5):
    """Simulate what a real environment would produce."""
    print(f"\n--- Real Environment {env_id} ---")
    
    # Set seed for this specific environment
    np.random.seed(env_id)
    
    burned_areas = []
    
    for step in range(steps):
        # More realistic simulation with variation
        base_burned = 170.0 + np.random.uniform(-25, 25)  # Environment variation
        reduction_per_break = 5.0 + np.random.uniform(-1, 1)  # Varied effectiveness
        total_breaks_placed = step * np.random.randint(1, 4)  # Variable breaks per step
        noise = np.random.uniform(-8, 8)  # Realistic noise
        
        burned_area = max(75.0, min(220.0, base_burned - total_breaks_placed * reduction_per_break + noise))
        burned_areas.append(burned_area)
        print(f"  Step {step}: burned_area = {burned_area:.1f}")
    
    return burned_areas

def main():
    print("🔍 Demonstrating the Dummy Environment Problem")
    print("=" * 60)
    
    # Simulate what happens when you have many dummy environments
    print("\n1️⃣  DUMMY ENVIRONMENTS (Old Behavior)")
    print("These use simple math formulas, leading to identical patterns:")
    
    all_dummy_areas = []
    for env_id in range(5):
        dummy_areas = simulate_dummy_environment(env_id)
        all_dummy_areas.extend(dummy_areas)
    
    # Analyze dummy patterns
    print(f"\nDummy environment analysis:")
    print(f"  Total samples: {len(all_dummy_areas)}")
    print(f"  Unique values: {len(set(all_dummy_areas))}")
    print(f"  Range: {min(all_dummy_areas):.1f} - {max(all_dummy_areas):.1f}")
    
    # Count identical values
    rounded_dummy = [round(b, 1) for b in all_dummy_areas]
    dummy_counts = Counter(rounded_dummy)
    print(f"  Most common values:")
    for value, count in dummy_counts.most_common(3):
        percentage = 100 * count / len(all_dummy_areas)
        print(f"    {value:.1f}: {count} times ({percentage:.1f}%)")
        if percentage > 30:
            print(f"      ⚠️  WARNING: This value appears {percentage:.1f}% of the time!")
    
    print("\n" + "=" * 60)
    
    # Simulate what happens with real environments
    print("\n2️⃣  REAL ENVIRONMENTS (Fixed Behavior)")
    print("These use realistic fire simulation with variation:")
    
    all_real_areas = []
    for env_id in range(5):
        real_areas = simulate_real_environment(env_id)
        all_real_areas.extend(real_areas)
    
    # Analyze real patterns
    print(f"\nReal environment analysis:")
    print(f"  Total samples: {len(all_real_areas)}")
    print(f"  Unique values: {len(set([round(b, 1) for b in all_real_areas]))}")
    print(f"  Range: {min(all_real_areas):.1f} - {max(all_real_areas):.1f}")
    print(f"  Standard deviation: {np.std(all_real_areas):.1f}")
    
    # Count identical values
    rounded_real = [round(b, 1) for b in all_real_areas]
    real_counts = Counter(rounded_real)
    print(f"  Most common values:")
    for value, count in real_counts.most_common(3):
        percentage = 100 * count / len(all_real_areas)
        print(f"    {value:.1f}: {count} times ({percentage:.1f}%)")
        if percentage > 30:
            print(f"      ⚠️  WARNING: This value appears {percentage:.1f}% of the time!")
        else:
            print(f"      ✅ Good: Natural variation")
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION")
    print("=" * 60)
    
    # Compare the two approaches
    dummy_std = np.std(all_dummy_areas)
    real_std = np.std(all_real_areas)
    
    print(f"Dummy environments:")
    print(f"  - Standard deviation: {dummy_std:.1f} (low = predictable patterns)")
    print(f"  - Identical values: High frequency")
    print(f"  - Cause: Simple mathematical formulas")
    
    print(f"\nReal environments:")
    print(f"  - Standard deviation: {real_std:.1f} (higher = natural variation)")
    print(f"  - Identical values: Low frequency")  
    print(f"  - Cause: Realistic fire simulation with noise")
    
    print(f"\n🔧 YOUR ISSUE:")
    print(f"  When you saw {len(set(rounded_dummy))} unique values from dummy environments")
    print(f"  vs {len(set(rounded_real))} unique values from real environments,")
    print(f"  it means many of your 64 environments were actually dummy fallbacks!")
    
    print(f"\n✅ SOLUTION:")
    print(f"  The updated code removes dummy fallbacks and forces real environment creation.")
    print(f"  Now you should see varied burned areas like the 'Real Environments' example above.")

if __name__ == "__main__":
    main()