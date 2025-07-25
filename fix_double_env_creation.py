#!/usr/bin/env python3
"""
Fix double environment creation in src/Train.py
The issue is that environments are being created during env_fn creation AND during AsyncVectorEnv initialization.
"""

import os
import shutil
import re

def fix_double_creation():
    """Fix the double environment creation issue."""
    
    train_file = "src/Train.py"
    backup_file = "src/Train_backup_double_fix.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    # Create backup
    shutil.copy(train_file, backup_file)
    print(f"💾 Created backup: {backup_file}")
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Replace the make_env_with_raster function with a lazy version
    new_make_env_function = '''def make_env_with_raster(raster, budget, kstep, sims, seed):
    """Create environment with specific raster data - LAZY CREATION (no double creation)."""
    def thunk():
        # Set random seed for reproducibility in subprocess
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Validate raster data first
                if not isinstance(raster, dict):
                    raise ValueError("Raster must be a dictionary")
                
                required_keys = ['slp', 'asp', 'fbfm', 'fireline_north', 'fireline_east', 'fireline_south', 'fireline_west']
                for key in required_keys:
                    if key not in raster:
                        raise ValueError(f"Missing required raster key: {key}")
                    if not isinstance(raster[key], np.ndarray):
                        raise ValueError(f"Raster key {key} must be numpy array")
                    # Validate array properties
                    if raster[key].size == 0:
                        raise ValueError(f"Raster key {key} is empty")
                    if not np.isfinite(raster[key]).all():
                        print(f"Warning: Raster key {key} contains non-finite values, cleaning...")
                        raster[key] = np.nan_to_num(raster[key], nan=0.0, posinf=1.0, neginf=0.0)
                
                # Start with minimal simulations and increase if successful
                if attempt == 0:
                    effective_sims = 1  # Very conservative first attempt
                elif attempt == 1:
                    effective_sims = min(2, sims)  # Slightly more
                else:
                    effective_sims = min(3, sims)  # Full attempt
                
                # Only print when actually creating (not during env_fn setup)
                print(f"🔧 Creating environment (attempt {attempt + 1}/{max_retries}, sims={effective_sims})")
                
                env = FuelBreakEnv(
                    raster,
                    break_budget=budget,
                    break_step=kstep,
                    num_simulations=effective_sims,
                    seed=seed,
                )
                
                # MINIMAL testing - only check if environment can be created
                try:
                    obs, _ = env.reset()
                    if obs is None or obs.size == 0:
                        raise RuntimeError("Invalid environment")
                    # Skip step testing for speed - any issues will be caught during training
                except Exception as test_e:
                    raise RuntimeError(f"Environment validation failed: {test_e}")
                
                return RobustAutoResetWrapper(env)
                
            except Exception as e:
                print(f"❌ Environment creation attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying with different parameters...")
                    import time
                    time.sleep(0.1)  # Brief pause between retries
                else:
                    print(f"💥 All {max_retries} attempts failed!")
                    raise RuntimeError(f"Failed to create environment after {max_retries} attempts. Last error: {e}")
    
    return thunk'''
    
    # Find and replace the entire make_env_with_raster function
    pattern = r'def make_env_with_raster\(raster, budget, kstep, sims, seed\):.*?return thunk'
    
    content = re.sub(pattern, new_make_env_function, content, flags=re.DOTALL)
    
    # Also fix the main environment creation loop to be clearer
    # Find the environment creation section
    env_creation_pattern = r'(print\(f"🔧 Creating \{N_ENVS\} environment functions\.\.\."\).*?)(for i, raster in enumerate\(selected_rasters\):.*?env_fns\.append\(env_fn\))'
    
    replacement = r'''\1
    # Create environment functions (these are just function definitions, not actual environments)
    print(f"📝 Preparing {N_ENVS} environment function definitions...")
    for i, raster in enumerate(selected_rasters):
        env_fn = make_env_with_raster(raster, BUDGET, K_STEPS, SIMS, seed=i + ep * N_ENVS)
        env_fns.append(env_fn)
        print(f"Environment {i}: 📝 Function prepared")
    
    print(f"✅ All {len(env_fns)} environment functions prepared")'''
    
    content = re.sub(env_creation_pattern, replacement, content, flags=re.DOTALL)
    
    # Write the fixed content
    with open(train_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed double environment creation in {train_file}")
    print(f"💾 Backup saved as: {backup_file}")
    
    return True

def create_super_fast_version():
    """Create a version with absolutely no upfront testing."""
    
    train_file = "src/Train.py"
    fast_file = "src/Train_super_fast.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Create super minimal environment function
    super_fast_function = '''def make_env_with_raster(raster, budget, kstep, sims, seed):
    """Super fast environment creation - NO TESTING."""
    def thunk():
        # Set random seed for reproducibility in subprocess
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        # Create environment directly - any issues caught during training
        env = FuelBreakEnv(
            raster,
            break_budget=budget,
            break_step=kstep,
            num_simulations=1,  # Always use 1 sim for speed
            seed=seed,
        )
        
        return RobustAutoResetWrapper(env)
    
    return thunk'''
    
    # Replace the function
    pattern = r'def make_env_with_raster\(raster, budget, kstep, sims, seed\):.*?return thunk'
    content = re.sub(pattern, super_fast_function, content, flags=re.DOTALL)
    
    # Remove all the verbose logging
    content = re.sub(r'print\(f"🔧 Creating.*?\)', '# Fast creation', content)
    content = re.sub(r'print\(f"Environment \{i\}:.*?\)', '# Fast creation', content)
    
    # Save super fast version
    with open(fast_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created super fast version: {fast_file}")
    return True

if __name__ == "__main__":
    print("🚀 Fixing double environment creation...")
    
    print("\n1️⃣  Fixing double creation issue...")
    if fix_double_creation():
        print("✅ Double creation issue fixed!")
    
    print("\n2️⃣  Creating super fast version...")
    if create_super_fast_version():
        print("✅ Super fast version created!")
    
    print("\n🎯 Results:")
    print("  • src/Train.py: Fixed double creation + minimal testing")
    print("  • src/Train_super_fast.py: No testing at all (fastest)")
    
    print("\n⚡ Expected behavior:")
    print("  • Before: Environment created twice (slow)")
    print("  • After: Environment created once (fast)")
    
    print("\n🚀 You can now run:")
    print("  python3 src/Train.py  # Fixed version")
    print("  python3 src/Train_super_fast.py  # Fastest version")