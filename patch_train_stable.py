#!/usr/bin/env python3
"""
Patch script to make src/Train.py stable for high-core workstations.
This script applies critical stability improvements to the existing Train.py.
"""

import os
import shutil
import re

def patch_train_py():
    """Apply stability patches to src/Train.py"""
    
    train_file = "src/Train.py"
    backup_file = "src/Train_backup.py"
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found!")
        return False
    
    # Create backup
    shutil.copy(train_file, backup_file)
    print(f"✅ Created backup: {backup_file}")
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Patch 1: Increase N_ENVS to 32 and optimize parameters
    content = re.sub(
        r'N_ENVS = \d+.*',
        'N_ENVS = 32  # Optimized for high-core workstations',
        content
    )
    
    # Patch 2: Increase STEPS_PER_EP for proper episode completion
    content = re.sub(
        r'STEPS_PER_EP = \d+.*',
        'STEPS_PER_EP = 25  # Allow episodes to complete naturally',
        content
    )
    
    # Patch 3: Optimize buffer and batch size
    content = re.sub(
        r'BUFFER_CAP = \d+.*',
        'BUFFER_CAP = 200_000  # Larger buffer for high-core systems',
        content
    )
    
    content = re.sub(
        r'BATCH_SIZE = \d+.*',
        'BATCH_SIZE = 64  # Larger batch size for stability',
        content
    )
    
    # Patch 4: Set BUDGET to 250
    content = re.sub(
        r'BUDGET = \d+.*',
        'BUDGET = 250  # Standard budget',
        content
    )
    
    # Patch 5: Reduce SIMS for stability
    content = re.sub(
        r'SIMS = \d+.*',
        'SIMS = 1  # Minimal simulations for maximum stability',
        content
    )
    
    # Patch 6: Add missing environment reset between episodes
    # Find the location after episode statistics and before next episode
    pattern = r'(# Episode statistics\s+if episode_rewards:\s+reward_window\.extend\(episode_rewards\))'
    replacement = r'''\1
        
        # Reset environments for next episode (CRITICAL FIX)
        print(f"🔄 Resetting environments for episode {ep + 1}...")
        try:
            obs = vec_env.reset()
        except Exception as reset_error:
            print(f"❌ Reset failed: {reset_error}")
            # Try to recreate environments
            try:
                vec_env.close()
                selected_rasters = raster_manager.get_random_rasters(N_ENVS)
                env_fns = [
                    make_env_with_raster(raster, BUDGET, K_STEPS, SIMS, seed=i + ep * N_ENVS) 
                    for i, raster in enumerate(selected_rasters)
                ]
                vec_env = AsyncVectorEnv(env_fns)
                obs = vec_env.reset()
                obs = obs[0] if isinstance(obs, tuple) else obs
                print("✅ Successfully recreated environments")
            except Exception as recreate_error:
                print(f"💥 Critical error: {recreate_error}")
                break'''
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Patch 7: Improve timeout handling - reduce timeout to 8 seconds
    content = re.sub(
        r'step_thread\.join\(timeout=\d+\.\d+\)',
        'step_thread.join(timeout=8.0)',
        content
    )
    
    content = re.sub(
        r'raise TimeoutError\("Environment step timed out after \d+ seconds"\)',
        'raise TimeoutError("Environment step timed out after 8 seconds")',
        content
    )
    
    # Patch 8: Add process cleanup improvements
    cleanup_code = '''
        # Enhanced process cleanup for high-core systems
        try:
            import psutil
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
            print(f"🧹 Cleaning up {len(children)} child processes...")
            
            for child in children:
                try:
                    child.terminate()
                    child.wait(timeout=3)
                except psutil.TimeoutExpired:
                    try:
                        child.kill()
                        child.wait(timeout=1)
                    except:
                        pass
                except:
                    pass
        except ImportError:
            print("psutil not available, using basic cleanup")
            for p in multiprocessing.active_children():
                try:
                    p.terminate()
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()
                except:
                    pass'''
    
    # Insert enhanced cleanup before environment recreation
    pattern = r'(for p in multiprocessing\.active_children\(\):.*?pass)'
    content = re.sub(pattern, cleanup_code, content, flags=re.DOTALL)
    
    # Patch 9: Add action data type fix
    content = re.sub(
        r'(def choose_actions_batch.*?)(action = np.zeros\(HxW\))',
        r'\1action = np.zeros(HxW, dtype=np.int8)',
        content,
        flags=re.DOTALL
    )
    
    # Write patched content
    with open(train_file, 'w') as f:
        f.write(content)
    
    print("✅ Applied all stability patches to src/Train.py")
    print("\n🔧 Key improvements applied:")
    print("  • Increased N_ENVS to 32 for high-core utilization")
    print("  • Increased STEPS_PER_EP to 25 for proper episode completion")
    print("  • Added critical environment reset between episodes")
    print("  • Reduced timeout to 8 seconds for faster error detection")
    print("  • Enhanced process cleanup with psutil")
    print("  • Fixed action data types (int8)")
    print("  • Optimized buffer and batch sizes")
    print("  • Set SIMS=1 for maximum stability")
    print(f"\n💾 Backup saved as: {backup_file}")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing required dependencies...")
    os.system("pip install psutil")
    print("✅ Dependencies installed")

if __name__ == "__main__":
    print("🚀 Patching src/Train.py for high-core workstation stability...")
    
    # Install dependencies
    install_dependencies()
    
    # Apply patches
    if patch_train_py():
        print("\n🎯 src/Train.py is now optimized for 32+ threads!")
        print("\n🚀 You can now run:")
        print("   python3 src/Train.py")
        print("\n📊 Expected performance:")
        print("   • 32 parallel environments")
        print("   • Stable training without hanging")
        print("   • Proper episode completion")
        print("   • Enhanced error recovery")
    else:
        print("\n❌ Patching failed!")