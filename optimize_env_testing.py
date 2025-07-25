#!/usr/bin/env python3
"""
Environment Testing Optimization Script
Provides multiple options to remove or parallelize environment testing in src/Train.py
"""

import os
import shutil
import re

def remove_testing_completely():
    """Option 1: Remove environment testing completely for maximum speed."""
    
    train_file = "src/Train.py"
    backup_file = "src/Train_no_testing.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Remove the entire testing block
    testing_pattern = r'''# Test the environment with a simple step
                print\(f"Testing environment functionality\.\.\."\)
                obs, info = env\.reset\(\)
                if obs is None or obs\.size == 0:
                    raise RuntimeError\("Environment reset returned empty observation"\)
                
                # Test with minimal action
                test_action = np\.zeros\(obs\.shape\[-2\] \* obs\.shape\[-1\]\)
                test_action\[0\] = 1  # Place one fuel break
                
                obs2, reward, done, truncated, info = env\.step\(test_action\)
                if obs2 is None or obs2\.size == 0:
                    raise RuntimeError\("Environment step returned empty observation"\)
                if not isinstance\(reward, \(int, float, np\.number\)\):
                    raise RuntimeError\(f"Environment step returned invalid reward type: \{type\(reward\)\}"\)
                if info is None or not isinstance\(info, dict\):
                    raise RuntimeError\("Environment step returned invalid info"\)
                
                # Reset after test
                env\.reset\(\)
                print\(f"Environment test successful!"\)'''
    
    # Replace with simple comment
    replacement = '''# Environment testing removed for performance'''
    
    content = re.sub(testing_pattern, replacement, content, flags=re.DOTALL)
    
    # Save optimized version
    with open(backup_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {backup_file} with testing completely removed")
    return True

def create_parallel_testing_version():
    """Option 2: Create version with parallelized environment testing."""
    
    train_file = "src/Train.py"
    parallel_file = "src/Train_parallel_testing.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Add parallel testing function at the top
    parallel_testing_code = '''
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def test_environment_parallel(env_fn, env_id):
    """Test a single environment in parallel."""
    try:
        env = env_fn()
        
        # Quick test
        obs, info = env.reset()
        if obs is None or obs.size == 0:
            return env_id, False, "Empty observation"
        
        # Minimal action test
        test_action = np.zeros(obs.shape[-2] * obs.shape[-1], dtype=np.int8)
        test_action[0] = 1
        
        obs2, reward, done, truncated, info = env.step(test_action)
        if obs2 is None or obs2.size == 0:
            return env_id, False, "Empty step observation"
        
        env.reset()  # Reset after test
        return env_id, True, "Success"
        
    except Exception as e:
        return env_id, False, str(e)

def test_environments_batch(env_fns, max_workers=8):
    """Test multiple environments in parallel."""
    print(f"🧪 Testing {len(env_fns)} environments in parallel with {max_workers} workers...")
    
    successful_envs = []
    failed_envs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all testing tasks
        futures = {
            executor.submit(test_environment_parallel, env_fn, i): i 
            for i, env_fn in enumerate(env_fns)
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            env_id, success, message = future.result()
            
            if success:
                successful_envs.append(env_id)
                print(f"Environment {env_id}: ✅")
            else:
                failed_envs.append((env_id, message))
                print(f"Environment {env_id}: ❌ {message}")
    
    print(f"✅ Testing complete: {len(successful_envs)}/{len(env_fns)} environments successful")
    return successful_envs, failed_envs

'''
    
    # Insert the parallel testing code after imports
    import_pattern = r'(from src\.utils\.loadingUtils import RasterManager.*?\n)'
    content = re.sub(import_pattern, r'\1' + parallel_testing_code, content, flags=re.DOTALL)
    
    # Replace the individual testing with batch testing in main()
    # Find where environments are created
    env_creation_pattern = r'(env_fns = \[.*?\])'
    replacement = r'''\1
    
    # Parallel environment testing (much faster than sequential)
    if len(env_fns) > 4:  # Only for multiple environments
        successful_envs, failed_envs = test_environments_batch(env_fns, max_workers=min(8, len(env_fns)))
        
        if len(successful_envs) < len(env_fns) // 2:
            print(f"⚠️  Too many failed environments ({len(failed_envs)}/{len(env_fns)})")
            print("Continuing with successful environments only...")
            
        # Filter to only successful environments
        env_fns = [env_fns[i] for i in successful_envs]
        N_ENVS = len(env_fns)
        print(f"🎯 Using {N_ENVS} successfully tested environments")'''
    
    content = re.sub(env_creation_pattern, replacement, content, flags=re.DOTALL)
    
    # Remove individual testing from make_env_with_raster
    testing_pattern = r'''# Test the environment with a simple step.*?print\(f"Environment test successful!"\)'''
    content = re.sub(testing_pattern, '# Individual testing removed - using batch testing instead', content, flags=re.DOTALL)
    
    # Save parallel version
    with open(parallel_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {parallel_file} with parallelized testing")
    return True

def create_minimal_testing_version():
    """Option 3: Keep minimal testing but make it much faster."""
    
    train_file = "src/Train.py"
    minimal_file = "src/Train_minimal_testing.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Replace comprehensive testing with minimal testing
    old_testing = r'''# Test the environment with a simple step
                print\(f"Testing environment functionality\.\.\."\)
                obs, info = env\.reset\(\)
                if obs is None or obs\.size == 0:
                    raise RuntimeError\("Environment reset returned empty observation"\)
                
                # Test with minimal action
                test_action = np\.zeros\(obs\.shape\[-2\] \* obs\.shape\[-1\]\)
                test_action\[0\] = 1  # Place one fuel break
                
                obs2, reward, done, truncated, info = env\.step\(test_action\)
                if obs2 is None or obs2\.size == 0:
                    raise RuntimeError\("Environment step returned empty observation"\)
                if not isinstance\(reward, \(int, float, np\.number\)\):
                    raise RuntimeError\(f"Environment step returned invalid reward type: \{type\(reward\)\}"\)
                if info is None or not isinstance\(info, dict\):
                    raise RuntimeError\("Environment step returned invalid info"\)
                
                # Reset after test
                env\.reset\(\)
                print\(f"Environment test successful!"\)'''
    
    # Replace with minimal testing
    new_testing = '''# Minimal environment test (fast)
                try:
                    obs, _ = env.reset()
                    if obs is None or obs.size == 0:
                        raise RuntimeError("Invalid environment")
                    # Skip step testing for speed
                except Exception as test_e:
                    raise RuntimeError(f"Environment test failed: {test_e}")'''
    
    content = re.sub(old_testing, new_testing, content, flags=re.DOTALL)
    
    # Save minimal version
    with open(minimal_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {minimal_file} with minimal testing")
    return True

def create_lazy_testing_version():
    """Option 4: Lazy testing - only test environments when they fail."""
    
    train_file = "src/Train.py"
    lazy_file = "src/Train_lazy_testing.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Remove upfront testing
    testing_pattern = r'''# Test the environment with a simple step.*?print\(f"Environment test successful!"\)'''
    content = re.sub(testing_pattern, '# Lazy testing - environments tested only when used', content, flags=re.DOTALL)
    
    # Add lazy testing wrapper
    lazy_wrapper_code = '''
class LazyTestWrapper:
    """Wrapper that tests environment only on first use."""
    
    def __init__(self, env_fn):
        self.env_fn = env_fn
        self.env = None
        self.tested = False
    
    def _ensure_env(self):
        if self.env is None:
            self.env = self.env_fn()
            
        if not self.tested:
            try:
                # Quick test on first use
                obs, _ = self.env.reset()
                if obs is None or obs.size == 0:
                    raise RuntimeError("Invalid environment")
                self.tested = True
            except Exception as e:
                raise RuntimeError(f"Lazy test failed: {e}")
    
    def reset(self):
        self._ensure_env()
        return self.env.reset()
    
    def step(self, action):
        self._ensure_env()
        return self.env.step(action)
    
    def close(self):
        if self.env is not None:
            self.env.close()

'''
    
    # Insert lazy wrapper after imports
    import_pattern = r'(from src\.utils\.loadingUtils import RasterManager.*?\n)'
    content = re.sub(import_pattern, r'\1' + lazy_wrapper_code, content, flags=re.DOTALL)
    
    # Modify environment creation to use lazy wrapper
    return_pattern = r'return RobustAutoResetWrapper\(env\)'
    content = re.sub(return_pattern, 'return LazyTestWrapper(lambda: RobustAutoResetWrapper(env))', content)
    
    # Save lazy version
    with open(lazy_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {lazy_file} with lazy testing")
    return True

def main():
    """Create all optimized versions."""
    print("🚀 Creating optimized environment testing versions...")
    
    print("\n1️⃣  Option 1: Complete Testing Removal")
    remove_testing_completely()
    
    print("\n2️⃣  Option 2: Parallel Testing")
    create_parallel_testing_version()
    
    print("\n3️⃣  Option 3: Minimal Testing")
    create_minimal_testing_version()
    
    print("\n4️⃣  Option 4: Lazy Testing")
    create_lazy_testing_version()
    
    print("\n✅ All optimized versions created!")
    print("\n🎯 Recommendations:")
    print("  • Fastest: src/Train_no_testing.py (no testing)")
    print("  • Balanced: src/Train_parallel_testing.py (parallel testing)")
    print("  • Safe: src/Train_minimal_testing.py (minimal testing)")
    print("  • Smart: src/Train_lazy_testing.py (test only when needed)")
    
    print("\n⚡ Speed Comparison (32 environments):")
    print("  • Original: ~60-120 seconds testing")
    print("  • No Testing: ~0 seconds")
    print("  • Parallel: ~8-15 seconds")
    print("  • Minimal: ~5-10 seconds")
    print("  • Lazy: ~0 seconds upfront")

if __name__ == "__main__":
    main()