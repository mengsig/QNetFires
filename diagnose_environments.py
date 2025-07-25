#!/usr/bin/env python3
"""
Environment diagnostic script to identify issues with environment creation.
This will help determine why you're getting dummy environments and identical burned areas.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from src.Train import make_env_with_raster
from src.utils.loadingUtils import RasterManager

def test_environment_creation(n_envs=10):
    """Test creating multiple environments to see failure patterns."""
    print(f"🔍 Testing Environment Creation ({n_envs} environments)")
    print("=" * 60)
    
    # Load rasters
    print("Loading rasters...")
    raster_manager = RasterManager()
    
    try:
        raster_manager.load_all_rasters("cropped_raster")
        print(f"✅ Loaded {len(raster_manager.all_rasters)} rasters")
        if len(raster_manager.all_rasters) == 0:
            raise ValueError("No rasters found")
    except Exception as e:
        print(f"❌ Raster loading failed: {e}")
        print("🔧 Creating dummy raster for testing...")
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
        raster_manager.all_rasters = [dummy_raster]
    
    # Test environment creation
    print(f"\nTesting {n_envs} environment creations...")
    
    success_count = 0
    failure_count = 0
    failure_reasons = {}
    
    for i in range(n_envs):
        try:
            raster = raster_manager.get_random_rasters(1)[0]
            
            print(f"\n--- Environment {i+1} ---")
            env_fn = make_env_with_raster(raster, budget=200, kstep=10, sims=2, seed=i)
            env = env_fn()
            
            # Test reset
            obs, info = env.reset()
            print(f"Reset: ✅ obs shape: {obs.shape}")
            
            # Test step
            action = np.zeros(obs.shape[-2] * obs.shape[-1])
            action[0] = 1
            obs2, reward, done, truncated, info = env.step(action)
            
            burned = info.get('burned', 'N/A')
            env_type = "DUMMY" if info.get('dummy', False) else "REAL"
            
            print(f"Step: ✅ reward: {reward:.4f}, burned: {burned:.1f}, type: {env_type}")
            
            if env_type == "REAL":
                success_count += 1
                print(f"Environment {i+1}: ✅ REAL environment working")
            else:
                failure_count += 1
                print(f"Environment {i+1}: ⚠️  DUMMY environment (fallback)")
                
        except Exception as e:
            failure_count += 1
            error_type = type(e).__name__
            if error_type not in failure_reasons:
                failure_reasons[error_type] = 0
            failure_reasons[error_type] += 1
            
            print(f"Environment {i+1}: ❌ FAILED: {error_type}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 ENVIRONMENT CREATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful (REAL environments): {success_count}/{n_envs} ({100*success_count/n_envs:.1f}%)")
    print(f"⚠️  Dummy environments: {failure_count - len(failure_reasons)}")
    print(f"❌ Complete failures: {len(failure_reasons)}")
    
    if failure_reasons:
        print("\nFailure reasons:")
        for reason, count in failure_reasons.items():
            print(f"  - {reason}: {count} times")
    
    # Recommendations
    print("\n🔧 RECOMMENDATIONS:")
    
    if success_count == 0:
        print("❌ CRITICAL: No real environments working!")
        print("  1. Check pyretechnics installation: pip install pyretechnics")
        print("  2. Check raster data in 'cropped_raster' directory")
        print("  3. Try reducing SIMS parameter to 1")
        
    elif success_count < n_envs * 0.5:
        print("⚠️  WARNING: Less than 50% environments working")
        print("  1. Consider reducing SIMS parameter")
        print("  2. Check for memory/resource constraints")
        print("  3. Some rasters might be corrupted")
        
    else:
        print("✅ Good: Most environments working")
        print("  - You should see mostly REAL environments in training")
        print("  - Identical burned areas indicate dummy environments")
    
    return success_count, failure_count

def test_burned_area_patterns():
    """Test for patterns in burned areas that indicate dummy environments."""
    print("\n🔥 Testing Burned Area Patterns")
    print("=" * 60)
    
    # Load rasters
    raster_manager = RasterManager()
    try:
        raster_manager.load_all_rasters("cropped_raster")
        if len(raster_manager.all_rasters) == 0:
            raise ValueError("No rasters")
    except:
        # Use dummy raster
        dummy_raster = {
            'slp': np.random.rand(50, 50).astype(np.float32),
            'asp': np.random.rand(50, 50).astype(np.float32),
            'fbfm': np.random.randint(1, 14, (50, 50)).astype(np.float32),
            'fireline_north': np.zeros((50, 50), dtype=np.float32),
            'fireline_east': np.zeros((50, 50), dtype=np.float32),
            'fireline_south': np.zeros((50, 50), dtype=np.float32),
            'fireline_west': np.zeros((50, 50), dtype=np.float32),
        }
        raster_manager.all_rasters = [dummy_raster]
    
    # Test multiple episodes to see burned area patterns
    burned_areas = []
    env_types = []
    
    for i in range(20):
        try:
            raster = raster_manager.get_random_rasters(1)[0]
            env_fn = make_env_with_raster(raster, budget=200, kstep=10, sims=1, seed=i)
            env = env_fn()
            
            obs, _ = env.reset()
            
            # Simulate a few steps
            for step in range(3):
                action = np.zeros(obs.shape[-2] * obs.shape[-1])
                action[step] = 1  # Place one fuel break per step
                obs, reward, done, truncated, info = env.step(action)
                
                burned = info.get('burned', None)
                env_type = "DUMMY" if info.get('dummy', False) else "REAL"
                
                if burned is not None:
                    burned_areas.append(burned)
                    env_types.append(env_type)
                
                if done:
                    break
                    
        except Exception as e:
            print(f"Episode {i} failed: {e}")
    
    if burned_areas:
        print(f"Collected {len(burned_areas)} burned area samples")
        
        # Analyze patterns
        from collections import Counter
        
        # Round to 1 decimal place for pattern detection
        rounded_areas = [round(b, 1) for b in burned_areas]
        value_counts = Counter(rounded_areas)
        
        print(f"\nBurned area statistics:")
        print(f"  Min: {min(burned_areas):.1f}")
        print(f"  Max: {max(burned_areas):.1f}")
        print(f"  Mean: {np.mean(burned_areas):.1f}")
        print(f"  Std: {np.std(burned_areas):.1f}")
        
        # Check for identical values
        most_common = value_counts.most_common(5)
        print(f"\nMost common burned area values:")
        for value, count in most_common:
            percentage = 100 * count / len(burned_areas)
            print(f"  {value:.1f}: {count} times ({percentage:.1f}%)")
            
            if percentage > 30:
                print(f"    ⚠️  WARNING: {value:.1f} appears {percentage:.1f}% of the time!")
                print(f"    This suggests dummy/fallback environments")
        
        # Check environment types
        real_count = env_types.count("REAL")
        dummy_count = env_types.count("DUMMY")
        print(f"\nEnvironment types:")
        print(f"  REAL: {real_count} ({100*real_count/len(env_types):.1f}%)")
        print(f"  DUMMY: {dummy_count} ({100*dummy_count/len(env_types):.1f}%)")
        
        if dummy_count > real_count:
            print("  ❌ More dummy than real environments!")
        elif dummy_count > 0:
            print("  ⚠️  Some dummy environments detected")
        else:
            print("  ✅ All real environments")
    
    else:
        print("❌ No burned area data collected")

def main():
    """Run all diagnostics."""
    print("🔍 Environment Diagnostic Tool")
    print("This will help identify why you're seeing identical burned areas")
    print("=" * 70)
    
    # Test environment creation
    success_count, failure_count = test_environment_creation(10)
    
    # Test burned area patterns
    test_burned_area_patterns()
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY AND NEXT STEPS")
    print("=" * 70)
    
    if success_count == 0:
        print("❌ CRITICAL ISSUE: No real environments working")
        print("\nIMMEDIATE ACTIONS:")
        print("1. Check pyretechnics installation:")
        print("   pip uninstall pyretechnics")
        print("   pip install pyretechnics")
        print("2. Verify raster data exists: ls -la cropped_raster/")
        print("3. Try train_sync.py with SIMS=1")
        
    elif success_count < 5:
        print("⚠️  PARTIAL ISSUE: Some environments failing")
        print("\nRECOMMENDED ACTIONS:")
        print("1. Reduce SIMS parameter to 1 in training scripts")
        print("2. Reduce N_ENVS to a smaller number (e.g., 8)")
        print("3. Check system resources (memory, CPU)")
        
    else:
        print("✅ ENVIRONMENTS MOSTLY WORKING")
        print("\nIf you still see identical burned areas in training:")
        print("1. The issue might be in the training loop, not environment creation")
        print("2. Check for AsyncVectorEnv hanging/timeout issues")
        print("3. Monitor the environment type indicators in training logs")
    
    print(f"\n🚀 Run training with: python3 train_sync.py")
    print(f"   Look for 'Type=REAL' vs 'Type=DUMMY' in the logs")

if __name__ == "__main__":
    main()